#!/usr/bin/env python3
"""
Independent check of a generated helmet: is the lattice confined to the inside
of the helmet?

    python tools/check_output.py final_helmet_auxetic.stl fitted_head.stl fitted_helmet.stl
    python tools/check_output.py final_helmet_auxetic.stl fitted_head.stl fitted_helmet.stl --png slices.png

Get fitted_head.stl / fitted_helmet.stl from `me323 --export-fit`: they are the
exact meshes the generator received, so nothing here re-derives the fit.

Needs numpy, scipy, open3d (and matplotlib for --png). Deliberately uses
Open3D's ray tracer rather than libigl, so it does not check the C++ output
with the same code that produced it.

Every solid point of the output is sorted into:
    skin     within reach of the helmet shell surface
    liner    within reach of the head surface (the head-side liner)
    lattice  inside the helmet cavity (mapped outward from the head centre)
    OUTSIDE  none of the above: material outside the helmet. Step 1 requires ~0.
Tolerances scale with the head size, matching the generator's own thicknesses
(skin <= 2.25 voxels, liner <= clearance + 2 voxels, voxel ~ head height / 120).
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy import ndimage


def scene(mesh):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return s


def tensor(a):
    return o3d.core.Tensor(np.ascontiguousarray(a, dtype=np.float32))


def cavity_map(helmet_scene, centre, res_deg=0.5, max_vent=0.05):
    """Independent re-implementation of HelmetCavity: radius map, vents filled."""
    nt = int(round(180 / res_deg)); nph = 2 * nt; step = np.pi / nt
    th = (np.arange(nt) + 0.5) * step; ph = (np.arange(nph) + 0.5) * step
    TT, PP = np.meshgrid(th, ph, indexing="ij")
    D = np.stack([np.sin(TT) * np.cos(PP), np.cos(TT), np.sin(TT) * np.sin(PP)], -1).reshape(-1, 3)
    R = helmet_scene.cast_rays(tensor(np.hstack([np.tile(centre, (len(D), 1)), D])))["t_hit"].numpy()
    R = R.astype(float).reshape(nt, nph)
    R[~np.isfinite(R)] = np.nan

    def wrap(i, j):
        if i < 0:
            i, j = -i - 1, j + nph // 2
        elif i >= nt:
            i, j = 2 * nt - i - 1, j + nph // 2
        return i, j % nph

    miss = np.isnan(R)
    lab, nlab = ndimage.label(miss, structure=np.ones((3, 3)))
    # Azimuth wraps, and each pole row joins itself half a turn round. Merge
    # labels with union-find: relabelling pair by pair can point a later pair at
    # a label that was already merged away, splitting a hole that should be one.
    parent = list(range(nlab + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = [(lab[i, 0], lab[i, -1]) for i in range(nt)]
    pairs += [(lab[i, 0], lab[i + d, -1]) for i in range(nt) for d in (-1, 1) if 0 <= i + d < nt]
    pairs += [(lab[0, j], lab[0, (j + nph // 2 + d) % nph]) for j in range(nph) for d in (-1, 0, 1)]
    pairs += [(lab[-1, j], lab[-1, (j + nph // 2 + d) % nph]) for j in range(nph) for d in (-1, 0, 1)]
    for a, b in pairs:
        if a and b:
            parent[find(a)] = find(b)
    lab = np.vectorize(find)(lab) * (lab > 0)
    w = np.sin(TT)
    vent = np.zeros_like(miss)
    for k in np.unique(lab[lab > 0]):
        if w[lab == k].sum() / w.sum() <= max_vent:
            vent |= lab == k
    if vent.any():
        cells = np.argwhere(vent)
        idx = -np.ones((nt, nph), int)
        idx[vent] = np.arange(len(cells))
        rows, cols, vals, b = [], [], [], np.zeros(len(cells))
        for k, (i, j) in enumerate(cells):
            deg = 0.0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = wrap(i + di, j + dj)
                if vent[ii, jj]:
                    rows.append(k); cols.append(idx[ii, jj]); vals.append(-1.0); deg += 1
                elif not np.isnan(R[ii, jj]):
                    b[k] += R[ii, jj]; deg += 1
            rows.append(k); cols.append(k); vals.append(deg)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(len(cells),) * 2).tocsc()
        R[vent] = spl.spsolve(A, b)
    return R, step


def inner_radius(R, step, centre, P):
    """Bilinear lookup; NaN where any corner looks through the opening."""
    nt, nph = R.shape
    d = P - centre
    r = np.linalg.norm(d, axis=1)
    u = d / np.maximum(r, 1e-300)[:, None]
    theta = np.arccos(np.clip(u[:, 1], -1, 1))
    phi = np.mod(np.arctan2(u[:, 2], u[:, 0]), 2 * np.pi)
    ti, pj = theta / step - 0.5, phi / step - 0.5
    i0, j0 = np.floor(ti).astype(int), np.floor(pj).astype(int)
    fi, fj = ti - i0, pj - j0

    def at(i, j):
        i, j = i.copy(), j.copy()
        n = i < 0; i[n] = -i[n] - 1; j[n] += nph // 2
        s = i >= nt; i[s] = 2 * nt - i[s] - 1; j[s] += nph // 2
        return R[i, j % nph]

    q00, q10, q01, q11 = at(i0, j0), at(i0 + 1, j0), at(i0, j0 + 1), at(i0 + 1, j0 + 1)
    return r, (1 - fi) * (1 - fj) * q00 + fi * (1 - fj) * q10 + (1 - fi) * fj * q01 + fi * fj * q11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("fitted_head")
    ap.add_argument("fitted_helmet")
    ap.add_argument("--png", default=None, help="write side/front slice pictures here")
    args = ap.parse_args()

    out = o3d.io.read_triangle_mesh(args.output)
    head = o3d.io.read_triangle_mesh(args.fitted_head)
    helm = o3d.io.read_triangle_mesh(args.fitted_helmet)
    for m in (out, head, helm):
        m.remove_duplicated_vertices()
    Vh = np.asarray(head.vertices)
    centre = (Vh.min(0) + Vh.max(0)) / 2               # the generator's reference point
    H = np.ptp(Vh[:, 1])
    v = H / 120.0                                       # nominal generator voxel
    skin_tol, liner_tol, cav_tol = 3.0 * v, 5.0 * v, 1.0 * v

    S_out, S_head, S_helm = scene(out), scene(head), scene(helm)
    R, step = cavity_map(S_helm, centre)

    Vo = np.asarray(out.vertices)
    g = 1.25 * v
    axes = [np.arange(Vo.min(0)[i] - g, Vo.max(0)[i] + g, g) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    solid = S_out.compute_occupancy(tensor(G)).numpy() > 0.5
    P = G[solid]
    d_skin = S_helm.compute_distance(tensor(P)).numpy()
    d_head = S_head.compute_distance(tensor(P)).numpy()
    r, rin = inner_radius(R, step, centre, P)
    in_cavity = np.isfinite(rin) & (r < rin + cav_tol)

    is_skin = d_skin <= skin_tol
    is_liner = ~is_skin & (d_head <= liner_tol)
    is_lattice = ~is_skin & ~is_liner & in_cavity
    is_outside = ~is_skin & ~is_liner & ~in_cavity
    cell = g ** 3
    vol = lambda m: m.sum() * cell

    labels, counts, _ = out.cluster_connected_triangles()
    counts = np.asarray(counts)
    tiny = int((counts < 0.001 * counts.max()).sum())

    unit = "L" if H < 2.0 else "model units^3"          # metres in, litres out
    k = 1e3 if H < 2.0 else 1.0
    print(f"output: {len(out.triangles):,} triangles, {len(counts)} connected pieces "
          f"({tiny} tiny fragments)")
    print(f"solid material, sampled every {g*1000 if H < 2 else g:.2f}{' mm' if H < 2 else ''}:")
    print(f"   helmet skin              {vol(is_skin)*k:8.3f} {unit}")
    print(f"   head-side liner          {vol(is_liner)*k:8.3f} {unit}"
          f"   (of which outside the helmet: {vol(is_liner & ~in_cavity)*k:.3f} -- step 2)")
    print(f"   lattice inside helmet    {vol(is_lattice)*k:8.3f} {unit}")
    print(f"   OUTSIDE the helmet       {vol(is_outside)*k:8.3f} {unit}")
    total = vol(solid) if solid.any() else 1.0
    frac = vol(is_outside) / total
    verdict = "PASS" if frac < 0.001 else "FAIL"
    print(f"step 1 (lattice only inside the helmet): {verdict} "
          f"-- {100*frac:.3f}% of all material lies outside the helmet")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as mpatches
        cls = np.zeros(len(G), int)
        cls[np.where(solid)[0][is_skin]] = 1
        cls[np.where(solid)[0][is_liner]] = 2
        cls[np.where(solid)[0][is_lattice]] = 3
        cls[np.where(solid)[0][is_outside]] = 4
        C = cls.reshape([len(a) for a in axes])
        cols = ["#ffffff", "#24426e", "#c3b8a8", "#f08c28", "#d7191c"]
        fig, axs = plt.subplots(1, 2, figsize=(14, 6.6))
        ix = np.argmin(np.abs(axes[0] - centre[0]))
        iz = np.argmin(np.abs(axes[2] - centre[2]))
        for ax, img, ext, title in (
                (axs[0], C[ix, :, :], (axes[2], axes[1]), "SIDE: front-to-back slice (face on the right)"),
                (axs[1], C[:, :, iz].T, (axes[0], axes[1]), "FRONT: side-to-side slice")):
            ax.imshow(img, origin="lower", cmap=ListedColormap(cols), vmin=0, vmax=4,
                      interpolation="nearest",
                      extent=[ext[0][0] * 1000, ext[0][-1] * 1000, ext[1][0] * 1000, ext[1][-1] * 1000])
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("mm")
            ax.set_aspect("equal")
        fig.legend(handles=[mpatches.Patch(color=cols[1], label="helmet skin"),
                            mpatches.Patch(color=cols[2], label="head-side liner"),
                            mpatches.Patch(color=cols[3], label="lattice inside the helmet"),
                            mpatches.Patch(color=cols[4], label="material OUTSIDE the helmet")],
                   loc="lower center", ncol=4, frameon=False)
        fig.suptitle(f"{os.path.basename(args.output)}: step 1 {verdict}", fontsize=12)
        plt.tight_layout(rect=(0, 0.06, 1, 0.96))
        plt.savefig(args.png, dpi=110)
        print(f"slices written to {args.png}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
