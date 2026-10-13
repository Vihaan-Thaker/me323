#!/usr/bin/env python3
"""
Independent check of a generated helmet: is the lattice confined to the inside
of the helmet (step 1), and is the bottom open so it can be put on (step 2)?

    python tools/check_output.py final_helmet_auxetic.stl fitted_head.stl fitted_helmet.stl
    python tools/check_output.py final_helmet_auxetic.stl fitted_head.stl fitted_helmet.stl --png slices.png

Get fitted_head.stl / fitted_helmet.stl from `me323 --export-fit`: they are the
exact meshes the generator received, so nothing here re-derives the fit. The
head keeps the lid marker of data/head.stl ("ME323-HEAD v1 lid-faces <first>
<count>" in its STL header), which says which triangles only close the scan.

Needs numpy, scipy, open3d (and matplotlib for --png). Deliberately uses
Open3D's ray tracer rather than libigl, so it does not check the C++ output
with the same code that produced it.

Every solid point of the output is sorted into:
    skin     within reach of the helmet shell surface
    liner    within reach of the head surface (the head-side liner)
    lattice  inside the helmet cavity (mapped outward from the head centre)
    OUTSIDE  none of the above: material outside the helmet.
Tolerances scale with the head size, matching the generator's own thicknesses
(skin <= 2.25 voxels, liner <= clearance + 2 voxels, voxel ~ head height / 120).

Step 1 (lattice only inside the helmet): OUTSIDE below 0.1% of all material.
Step 2 (bottom open, can be put on), each below 0.1% of all material:
    a. liner or lattice in directions where the head was not scanned (seen
       from the head centre, only its lid is there);
    b. liner in directions the helmet does not cover (seen from the head
       centre; a vent counts as covered, and under one the liner may stand
       above the bridged surface, as it must not be holed there);
    c. for the best straight lift-off direction (tilts -45..45 and sideways
       -10..10 degrees, 5-degree steps), material the head passes through as
       the helmet lifts off. Shell within the head clearance is a fit problem
       (the stock helmet sits too close), so it is reported apart.
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy import ndimage

STL_TRIANGLE = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
LIMIT = 0.001


def scene(mesh):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return s


def tensor(a):
    return o3d.core.Tensor(np.ascontiguousarray(a, dtype=np.float32))


def inside(s, P):
    """Inside test by majority over 3 rays. Open3D's one-ray parity (also
    behind compute_signed_distance) flips for points whose ray grazes an edge,
    e.g. points exactly in the plane of the head's extreme vertex."""
    return s.compute_occupancy(tensor(P), nsamples=3).numpy() > 0.5


def signed_distance(s, P):
    d = s.compute_distance(tensor(P)).numpy()
    return np.where(inside(s, P), -d, d)


def read_binary_stl(path):
    """Header bytes and the triangles in file order, (n, 3, 3). Read by hand:
    the lid marker refers to file order, which a mesh loader may not keep."""
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[80:84], "little") if len(raw) >= 84 else -1
    if len(raw) != 84 + 50 * n:
        sys.exit(f"{path} is not a binary STL")
    return raw[:80], np.frombuffer(raw, STL_TRIANGLE, n, 84)["corners"].astype(float)


def head_lid(header, tri, path):
    """Per-triangle lid mask from the marker, or None for a complete head.
    Same checks as the generator: a fan around one point nothing else touches."""
    text = header.split(b"\0")[0].decode("ascii", "replace")
    if not text.startswith("ME323-HEAD"):
        return None
    parts = text.split()
    if len(parts) != 5 or parts[1] != "v1" or parts[2] != "lid-faces":
        sys.exit(f"{path}: unrecognised marker {text!r}")
    first, count = int(parts[3]), int(parts[4])
    if count < 3 or first < 0 or first + count > len(tri):
        sys.exit(f"{path}: marker names triangles past the end of the mesh")
    lid = tri[first:first + count]
    shared = [c for c in lid[0] if (lid == c).all(axis=2).any(axis=1).all()]
    if len(shared) != 1:
        sys.exit(f"{path}: the marked triangles are not a fan around one point")
    mask = np.zeros(len(tri), bool)
    mask[first:first + count] = True
    if (tri[~mask] == shared[0]).all(axis=2).any():
        sys.exit(f"{path}: an unmarked triangle touches the lid's apex")
    return mask


def soup_mesh(tri):
    """Triangle i of the mesh is triangle i of the file (Open3D ray hits report
    it as primitive id i)."""
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(tri.reshape(-1, 3)),
        o3d.utility.Vector3iVector(np.arange(3 * len(tri)).reshape(-1, 3)))


def direction_map(res_deg=0.5):
    """Cell centres of the direction map: polar angle from +Y, azimuth atan2(z, x)."""
    nt = int(round(180 / res_deg)); nph = 2 * nt; step = np.pi / nt
    th = (np.arange(nt) + 0.5) * step; ph = (np.arange(nph) + 0.5) * step
    TT, PP = np.meshgrid(th, ph, indexing="ij")
    D = np.stack([np.sin(TT) * np.cos(PP), np.cos(TT), np.sin(TT) * np.sin(PP)], -1).reshape(-1, 3)
    return nt, nph, step, TT, D


def first_hits(s, centre, D):
    return s.cast_rays(tensor(np.hstack([np.tile(centre, (len(D), 1)), D])))


def cavity_map(helmet_scene, centre, res_deg=0.5, max_vent=0.05, return_vents=False):
    """Independent re-implementation of HelmetCavity: radius map, vents filled
    (and, if asked, which cells were vents)."""
    nt, nph, step, TT, D = direction_map(res_deg)
    R = first_hits(helmet_scene, centre, D)["t_hit"].numpy().astype(float).reshape(nt, nph)
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
    return (R, step, vent) if return_vents else (R, step)


def unscanned_map(head_scene, lid, centre, res_deg=0.5):
    """True where the ray from the centre leaves the head through its lid."""
    nt, nph, _, _, D = direction_map(res_deg)
    hit = first_hits(head_scene, centre, D)
    if not np.isfinite(hit["t_hit"].numpy()).all():
        sys.exit("a ray from the head centre misses the head: the head mesh has a hole")
    return lid[hit["primitive_ids"].numpy()].reshape(nt, nph)


def bilinear(shape, step, centre, P):
    """Distance from the centre, the four surrounding map cells and their weights."""
    nt, nph = shape
    d = P - centre
    r = np.linalg.norm(d, axis=1)
    u = d / np.maximum(r, 1e-300)[:, None]
    theta = np.arccos(np.clip(u[:, 1], -1, 1))
    phi = np.mod(np.arctan2(u[:, 2], u[:, 0]), 2 * np.pi)
    ti, pj = theta / step - 0.5, phi / step - 0.5
    i0, j0 = np.floor(ti).astype(int), np.floor(pj).astype(int)
    fi, fj = ti - i0, pj - j0

    def cell(i, j):
        i, j = i.copy(), j.copy()
        n = i < 0; i[n] = -i[n] - 1; j[n] += nph // 2
        s = i >= nt; i[s] = 2 * nt - i[s] - 1; j[s] += nph // 2
        return i, j % nph

    corners = [cell(i0, j0), cell(i0 + 1, j0), cell(i0, j0 + 1), cell(i0 + 1, j0 + 1)]
    weights = [(1 - fi) * (1 - fj), fi * (1 - fj), (1 - fi) * fj, fi * fj]
    return r, corners, weights


def lift_off_direction(tilt, side):
    t, q = np.radians(tilt), np.radians(side)
    d = np.array([np.sin(q) * np.cos(t), np.cos(q) * np.cos(t), np.sin(t)])
    return d / np.linalg.norm(d)


def describe(tilt, side):
    if tilt == 0 and side == 0:
        return "straight up"
    parts = []
    if tilt:
        parts.append(f"{abs(tilt)} deg toward " + ("-Z (back)" if tilt < 0 else "+Z (face)"))
    if side:
        parts.append(f"{abs(side)} deg toward " + ("-X" if side < 0 else "+X"))
    return "up, tilted " + " and ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("fitted_head")
    ap.add_argument("fitted_helmet")
    ap.add_argument("--png", default=None, help="write side/front slice pictures here")
    args = ap.parse_args()

    out = o3d.io.read_triangle_mesh(args.output)
    helm = o3d.io.read_triangle_mesh(args.fitted_helmet)
    for m in (out, helm):
        m.remove_duplicated_vertices()
    header, Th = read_binary_stl(args.fitted_head)
    lid = head_lid(header, Th, args.fitted_head)
    head = soup_mesh(Th)
    Vh = Th.reshape(-1, 3)
    centre = (Vh.min(0) + Vh.max(0)) / 2               # the generator's reference point
    H = np.ptp(Vh[:, 1])
    v = H / 120.0                                       # nominal generator voxel
    skin_tol, liner_tol, cav_tol = 3.0 * v, 5.0 * v, 1.0 * v

    S_out, S_head, S_helm = scene(out), scene(head), scene(helm)
    R, step, vents = cavity_map(S_helm, centre, return_vents=True)
    if lid is not None:
        if not inside(S_head, centre[None])[0]:
            sys.exit("the head centre is not inside the head")
        U = unscanned_map(S_head, lid, centre)

    Vo = np.asarray(out.vertices)
    g = 1.25 * v
    axes = [np.arange(Vo.min(0)[i] - g, Vo.max(0)[i] + g, g) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    solid = inside(S_out, G)
    P = G[solid]
    d_skin = S_helm.compute_distance(tensor(P)).numpy()
    sd_head = signed_distance(S_head, P)
    d_head = np.abs(sd_head)
    r, corners, weights = bilinear(R.shape, step, centre, P)
    rin = sum(w * R[c] for c, w in zip(corners, weights))   # NaN where any corner is open
    in_cavity = np.isfinite(rin) & (r < rin + cav_tol)
    # Unscanned / uncovered only where all four surrounding cells are: the
    # generator stops at the first, so its own boundary layer never counts.
    unscanned = (np.logical_and.reduce([U[c] for c in corners]) if lid is not None
                 else np.zeros(len(P), bool))
    uncovered = np.logical_and.reduce([np.isnan(R[c]) for c in corners])

    is_skin = d_skin <= skin_tol
    is_liner = ~is_skin & (d_head <= liner_tol)
    is_lattice = ~is_skin & ~is_liner & in_cavity
    is_outside = ~is_skin & ~is_liner & ~in_cavity
    touching = is_skin & (sd_head < v)                 # shell inside the head clearance
    cell = g ** 3
    vol = lambda m: m.sum() * cell

    # Lift-off: rays from every judged point along each candidate direction.
    judged = np.where(~touching)[0]
    results = []
    for tilt in range(-45, 46, 5):
        for side in range(-10, 11, 5):
            d = lift_off_direction(tilt, side)
            t = S_head.cast_rays(tensor(np.hstack([P[judged], np.tile(d, (len(judged), 1))])))["t_hit"].numpy()
            blocked = np.zeros(len(P), bool)
            blocked[judged[np.isfinite(t)]] = True
            results.append((blocked.sum(), np.degrees(np.arccos(d[1])), tilt, side, blocked))
    best = min(results, key=lambda x: (x[0], x[1]))
    up = next(x for x in results if x[2] == 0 and x[3] == 0)
    in_way = best[4]

    labels, counts, _ = out.cluster_connected_triangles()
    counts = np.asarray(counts)
    tiny = int((counts < 0.001 * counts.max()).sum())

    unit = "L" if H < 2.0 else "model units^3"          # metres in, litres out
    k = 1e3 if H < 2.0 else 1.0
    total = vol(solid) if solid.any() else 1.0
    pct = lambda m: 100 * vol(m) / total
    print(f"output: {len(out.triangles):,} triangles, {len(counts)} connected pieces "
          f"({tiny} tiny fragments; largest piece {100 * counts.max() / counts.sum():.1f}% of triangles)")
    print(f"solid material, sampled every {g*1000 if H < 2 else g:.2f}{' mm' if H < 2 else ''}:")
    print(f"   helmet skin              {vol(is_skin)*k:8.3f} {unit}")
    print(f"   head-side liner          {vol(is_liner)*k:8.3f} {unit}")
    print(f"   lattice inside helmet    {vol(is_lattice)*k:8.3f} {unit}")
    print(f"   OUTSIDE the helmet       {vol(is_outside)*k:8.3f} {unit}")
    under_vent = np.logical_and.reduce([vents[c] for c in corners])
    print(f"   (liner under the vents   {vol(is_liner & under_vent)*k:8.3f} {unit}; vents are bridged, not open)")

    frac1 = vol(is_outside) / total
    v1 = "PASS" if frac1 < LIMIT else "FAIL"
    print(f"step 1 (lattice only inside the helmet): {v1} "
          f"-- {100*frac1:.3f}% of all material lies outside the helmet")

    a = ~is_skin & unscanned
    b = is_liner & uncovered
    ok = [vol(a) / total < LIMIT, vol(b) / total < LIMIT, vol(in_way) / total < LIMIT]
    v2 = "PASS" if all(ok) else "FAIL"
    print(f"step 2 (bottom open, can be put on): {v2}")
    if lid is None:
        print("   a. head has no lid marker: treated as complete, nothing unscanned")
    else:
        print(f"   a. liner/lattice where the head was not scanned:  {pct(a):6.3f}%  {'ok' if ok[0] else 'FAIL'}"
              f"   (shell there: {pct(is_skin & unscanned):.2f}%, allowed)")
    print(f"   b. liner where the helmet does not cover:         {pct(b):6.3f}%  {'ok' if ok[1] else 'FAIL'}")
    print(f"   c. in the head's way, best lift-off ({describe(best[2], best[3])}): "
          f"{pct(in_way):6.3f}%  {'ok' if ok[2] else 'FAIL'}"
          f"   (of it shell {pct(in_way & is_skin):.3f}%; straight up: {pct(up[4]):.2f}%)")
    if touching.any():
        amount = f"{vol(touching)*1e6:.2f} cm^3" if H < 2 else f"{vol(touching):.3g} {unit}"
        print(f"   note: shell within the head clearance (fit too tight there, not judged): {amount}")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as mpatches
        cls = np.zeros(len(G), int)
        where = np.where(solid)[0]
        cls[where[is_skin]] = 1
        cls[where[is_liner]] = 2
        cls[where[is_lattice]] = 3
        cls[where[is_outside]] = 4
        cls[where[a]] = 5
        cls[where[in_way]] = 6
        C = cls.reshape([len(x) for x in axes])
        cols = ["#ffffff", "#24426e", "#c3b8a8", "#f08c28", "#d7191c", "#7b3294", "#e7298a"]
        fig, axs = plt.subplots(1, 2, figsize=(14, 6.6))
        ix = np.argmin(np.abs(axes[0] - centre[0]))
        iz = np.argmin(np.abs(axes[2] - centre[2]))
        for ax, img, ext, title in (
                (axs[0], C[ix, :, :], (axes[2], axes[1]), "SIDE: front-to-back slice (face on the right)"),
                (axs[1], C[:, :, iz].T, (axes[0], axes[1]), "FRONT: side-to-side slice")):
            ax.imshow(img, origin="lower", cmap=ListedColormap(cols), vmin=0, vmax=6,
                      interpolation="nearest",
                      extent=[ext[0][0] * 1000, ext[0][-1] * 1000, ext[1][0] * 1000, ext[1][-1] * 1000])
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("mm")
            ax.set_aspect("equal")
        fig.legend(handles=[mpatches.Patch(color=cols[1], label="helmet skin"),
                            mpatches.Patch(color=cols[2], label="head-side liner"),
                            mpatches.Patch(color=cols[3], label="lattice inside the helmet"),
                            mpatches.Patch(color=cols[4], label="material OUTSIDE the helmet"),
                            mpatches.Patch(color=cols[5], label="liner/lattice where head not scanned"),
                            mpatches.Patch(color=cols[6], label="in the head's way lifting off")],
                   loc="lower center", ncol=3, frameon=False)
        fig.suptitle(f"{os.path.basename(args.output)}: step 1 {v1}, step 2 {v2}", fontsize=12)
        plt.tight_layout(rect=(0, 0.09, 1, 0.96))
        plt.savefig(args.png, dpi=110)
        print(f"slices written to {args.png}")
    return 0 if v1 == "PASS" and v2 == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
