#include "helmet_cavity.h"

#include <igl/AABB.h>
#include <igl/Hit.h>
#include <igl/winding_number.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <queue>
#include <string>
#include <vector>

namespace
{

constexpr double pi = 3.14159265358979323846;

std::string percent(double fraction)
{
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f", 100.0 * fraction);
    return text;
}

// Unit vector through the centre of map cell (i, j).
Eigen::RowVector3d cell_direction(double step, int i, int j)
{
    const double theta = (i + 0.5) * step;
    const double phi = (j + 0.5) * step;
    return Eigen::RowVector3d(
        std::sin(theta) * std::cos(phi), std::cos(theta),
        std::sin(theta) * std::sin(phi));
}

} // namespace

double HelmetCavity::radius_at(int i, int j) const
{
    // Stepping off either pole lands on the same ring, half a turn round.
    if (i < 0)
    {
        i = -i - 1;
        j += n_phi / 2;
    }
    else if (i >= n_theta)
    {
        i = 2 * n_theta - i - 1;
        j += n_phi / 2;
    }
    j %= n_phi;
    if (j < 0)
    {
        j += n_phi;
    }
    return radius[static_cast<std::size_t>(i) * n_phi + j];
}

namespace
{

// The map cells around the direction of p, for bilinear interpolation.
struct Corners
{
    double r;            // distance of p from the centre
    double fi;
    double fj;
    double q[4];         // radii at (i0, j0), (i0+1, j0), (i0, j0+1), (i0+1, j0+1)

    bool any_open() const
    {
        return std::isnan(q[0]) || std::isnan(q[1]) || std::isnan(q[2]) ||
               std::isnan(q[3]);
    }
};

Corners corners_of(const HelmetCavity& cavity, const Eigen::RowVector3d& p)
{
    const Eigen::RowVector3d d = p - cavity.centre;
    Corners c{};
    c.r = d.norm();
    // At the centre itself the direction is undefined; it lies inside the head,
    // so any inside answer will do.
    const Eigen::RowVector3d u = c.r > 0.0 ? Eigen::RowVector3d(d / c.r)
                                           : Eigen::RowVector3d(0.0, 1.0, 0.0);
    const double theta = std::acos(std::clamp(u.y(), -1.0, 1.0));
    double phi = std::atan2(u.z(), u.x());
    if (phi < 0.0)
    {
        phi += 2.0 * pi;
    }

    // Bilinear between the four surrounding cell centres.
    const double ti = theta / cavity.step - 0.5;
    const double pj = phi / cavity.step - 0.5;
    const int i0 = static_cast<int>(std::floor(ti));
    const int j0 = static_cast<int>(std::floor(pj));
    c.fi = ti - i0;
    c.fj = pj - j0;
    c.q[0] = cavity.radius_at(i0, j0);
    c.q[1] = cavity.radius_at(i0 + 1, j0);
    c.q[2] = cavity.radius_at(i0, j0 + 1);
    c.q[3] = cavity.radius_at(i0 + 1, j0 + 1);
    return c;
}

} // namespace

double HelmetCavity::radial_distance(const Eigen::RowVector3d& p, double margin) const
{
    const Corners c = corners_of(*this, p);
    // Any corner looking through the opening: treat the point as outside, so
    // lattice stops a fraction of a cell short of the rim instead of guessing.
    if (c.any_open())
    {
        return open_distance;
    }
    const double inner = (1.0 - c.fi) * (1.0 - c.fj) * c.q[0] + c.fi * (1.0 - c.fj) * c.q[1]
        + (1.0 - c.fi) * c.fj * c.q[2] + c.fi * c.fj * c.q[3];
    return c.r - (inner - margin);
}

bool HelmetCavity::covers(const Eigen::RowVector3d& p) const
{
    return !corners_of(*this, p).any_open();
}

bool build_helmet_cavity(
    const Eigen::MatrixXd& helmet_vertices,
    const Eigen::MatrixXi& helmet_faces,
    const Eigen::RowVector3d& centre,
    HelmetCavity& cavity,
    std::string& error,
    double resolution_degrees,
    double max_vent_fraction)
{
    if (helmet_vertices.rows() == 0 || helmet_faces.rows() == 0 ||
        helmet_vertices.cols() != 3 || helmet_faces.cols() != 3 ||
        !helmet_vertices.allFinite() || !centre.allFinite())
    {
        error = "Helmet cavity: empty or invalid helmet mesh.";
        return false;
    }
    if (!(resolution_degrees > 0.0) || 180.0 / resolution_degrees > 4000.0)
    {
        error = "Helmet cavity: invalid map resolution.";
        return false;
    }

    const Eigen::RowVector3d vmin = helmet_vertices.colwise().minCoeff();
    const Eigen::RowVector3d vmax = helmet_vertices.colwise().maxCoeff();
    const double diagonal = (vmax - vmin).norm();
    if (!(diagonal > 0.0) || !std::isfinite(diagonal))
    {
        error = "Helmet cavity: the helmet has zero size.";
        return false;
    }

    // libigl's ray-triangle test rejects a hit when |det| < 1e-6 in absolute
    // terms, and |det| <= |e1 x e2| scales with triangle size squared. With the
    // helmet in metres, 51% of its triangles fall below that and become
    // invisible to rays: only 96.4% of directions then met the inner face first,
    // against 100% at a millimetre scale. So rays are cast against a copy scaled
    // to a fixed size, whatever units the model is in, and distances are scaled
    // back afterwards.
    const double scale = 1000.0 / diagonal;
    const Eigen::MatrixXd scaled = helmet_vertices * scale;
    const Eigen::RowVector3d origin = centre * scale;
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(scaled, helmet_faces);

    cavity = HelmetCavity{};
    cavity.centre = centre;
    cavity.n_theta = static_cast<int>(std::lround(180.0 / resolution_degrees));
    cavity.n_phi = 2 * cavity.n_theta;            // even, as crossing a pole needs
    cavity.step = pi / cavity.n_theta;
    cavity.open_distance = 10.0 * diagonal;
    const std::size_t cells = static_cast<std::size_t>(cavity.n_theta) * cavity.n_phi;
    cavity.radius.assign(cells, std::numeric_limits<double>::quiet_NaN());

    std::vector<double> weight(cells);            // solid angle ~ sin(theta)
    double total_weight = 0.0;
    double seen_weight = 0.0;
    std::size_t hits = 0;
    std::size_t facing_centre = 0;
    for (int i = 0; i < cavity.n_theta; ++i)
    {
        const double theta = (i + 0.5) * cavity.step;
        for (int j = 0; j < cavity.n_phi; ++j)
        {
            const std::size_t cell = static_cast<std::size_t>(i) * cavity.n_phi + j;
            weight[cell] = std::sin(theta);
            total_weight += weight[cell];

            const Eigen::RowVector3d direction = cell_direction(cavity.step, i, j);
            igl::Hit<double> hit;
            if (!tree.intersect_ray(scaled, helmet_faces, origin, direction, hit))
            {
                continue;
            }
            cavity.radius[cell] = hit.t / scale;  // direction is unit length
            seen_weight += weight[cell];
            ++hits;
            const Eigen::RowVector3d a = scaled.row(helmet_faces(hit.id, 0));
            const Eigen::RowVector3d b = scaled.row(helmet_faces(hit.id, 1));
            const Eigen::RowVector3d c = scaled.row(helmet_faces(hit.id, 2));
            if ((b - a).cross(c - a).dot(direction) < 0.0)
            {
                ++facing_centre;
            }
        }
    }

    if (hits == 0)
    {
        error = "Helmet cavity: no direction from the head centre meets the helmet; "
                "the head centre is not under it. Adjust the fit.";
        return false;
    }
    // Star-shape check. The first hit should always be the inner face, whose
    // normal faces the centre if triangles wind outward, away from it if they
    // wind inward, so accept agreement either way.
    const double facing = static_cast<double>(facing_centre) / hits;
    cavity.inner_face_fraction = std::max(facing, 1.0 - facing);
    cavity.seen_fraction = seen_weight / total_weight;
    if (cavity.inner_face_fraction < 0.99)
    {
        error = "Helmet cavity: only " + percent(cavity.inner_face_fraction)
            + "% of rays from the head centre meet the same side of the shell "
              "first, so its inside can't be mapped from there. Either the helmet "
              "is not over the head (adjust the fit) or its triangles are wound "
              "inconsistently (repair the mesh).";
        return false;
    }

    // Holes: connected regions of unseen directions. 8-neighbourhood, azimuth
    // wraps, and a pole is crossed by continuing half a turn round.
    const auto neighbours8 = [&](int i, int j, auto&& visit)
    {
        for (int di = -1; di <= 1; ++di)
        {
            for (int dj = -1; dj <= 1; ++dj)
            {
                if (di != 0 || dj != 0)
                {
                    visit(i + di, j + dj);
                }
            }
        }
    };
    const auto wrap = [&](int& i, int& j)
    {
        if (i < 0)
        {
            i = -i - 1;
            j += cavity.n_phi / 2;
        }
        else if (i >= cavity.n_theta)
        {
            i = 2 * cavity.n_theta - i - 1;
            j += cavity.n_phi / 2;
        }
        j %= cavity.n_phi;
        if (j < 0)
        {
            j += cavity.n_phi;
        }
    };

    std::vector<int> component(cells, -1);
    std::vector<std::vector<std::size_t>> holes;
    std::vector<double> hole_weight;
    for (std::size_t start = 0; start < cells; ++start)
    {
        if (!std::isnan(cavity.radius[start]) || component[start] >= 0)
        {
            continue;
        }
        const int id = static_cast<int>(holes.size());
        holes.emplace_back();
        hole_weight.push_back(0.0);
        std::queue<std::size_t> queue;
        queue.push(start);
        component[start] = id;
        while (!queue.empty())
        {
            const std::size_t cell = queue.front();
            queue.pop();
            holes[id].push_back(cell);
            hole_weight[id] += weight[cell];
            const int ci = static_cast<int>(cell / cavity.n_phi);
            const int cj = static_cast<int>(cell % cavity.n_phi);
            neighbours8(ci, cj, [&](int ni, int nj)
            {
                wrap(ni, nj);
                const std::size_t next = static_cast<std::size_t>(ni) * cavity.n_phi + nj;
                if (std::isnan(cavity.radius[next]) && component[next] < 0)
                {
                    component[next] = id;
                    queue.push(next);
                }
            });
        }
    }

    // Small holes are vents; bridge them with a harmonic fill, solved exactly.
    // (400 rounds of neighbour averaging were still 7.4 mm short across the
    // 26-degree crown vent at this resolution.)
    std::vector<int> unknown(cells, -1);
    std::vector<std::size_t> vent_cells;
    double open_weight = 0.0;
    for (std::size_t h = 0; h < holes.size(); ++h)
    {
        const double fraction = hole_weight[h] / total_weight;
        if (fraction <= max_vent_fraction)
        {
            ++cavity.vents_filled;
            cavity.largest_vent_fraction = std::max(cavity.largest_vent_fraction, fraction);
            for (const std::size_t cell : holes[h])
            {
                unknown[cell] = static_cast<int>(vent_cells.size());
                vent_cells.push_back(cell);
            }
        }
        else
        {
            open_weight += hole_weight[h];
        }
    }
    cavity.open_fraction = open_weight / total_weight;

    if (!vent_cells.empty())
    {
        // Discrete Laplace equation on the vent cells, 4-neighbourhood, with the
        // seen radii around each vent as boundary values. Every vent touches seen
        // cells (anything touching the opening would belong to the opening's
        // hole), so the matrix is symmetric positive definite.
        const int n = static_cast<int>(vent_cells.size());
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(static_cast<std::size_t>(n) * 5);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
        for (int k = 0; k < n; ++k)
        {
            const int ci = static_cast<int>(vent_cells[k] / cavity.n_phi);
            const int cj = static_cast<int>(vent_cells[k] % cavity.n_phi);
            double diagonal_entry = 0.0;
            const int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (const auto& offset : offsets)
            {
                int ni = ci + offset[0];
                int nj = cj + offset[1];
                wrap(ni, nj);
                const std::size_t next = static_cast<std::size_t>(ni) * cavity.n_phi + nj;
                if (unknown[next] >= 0)
                {
                    entries.emplace_back(k, unknown[next], -1.0);
                    diagonal_entry += 1.0;
                }
                else if (!std::isnan(cavity.radius[next]))
                {
                    rhs[k] += cavity.radius[next];
                    diagonal_entry += 1.0;
                }
            }
            entries.emplace_back(k, k, diagonal_entry);
        }
        Eigen::SparseMatrix<double> laplacian(n, n);
        laplacian.setFromTriplets(entries.begin(), entries.end());
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(laplacian);
        const Eigen::VectorXd filled = solver.solve(rhs);
        if (solver.info() != Eigen::Success || !filled.allFinite())
        {
            error = "Helmet cavity: bridging the vents failed (singular system).";
            return false;
        }
        for (int k = 0; k < n; ++k)
        {
            cavity.radius[vent_cells[k]] = filled[k];
        }
    }
    return true;
}

bool exclude_unscanned_directions(
    HelmetCavity& cavity,
    const Eigen::MatrixXd& head_vertices,
    const Eigen::MatrixXi& head_faces,
    const std::vector<bool>& lid,
    std::string& error)
{
    if (head_vertices.rows() == 0 || head_faces.rows() == 0 ||
        head_vertices.cols() != 3 || head_faces.cols() != 3 ||
        !head_vertices.allFinite() ||
        lid.size() != static_cast<std::size_t>(head_faces.rows()))
    {
        error = "Unscanned directions: empty or invalid head mesh or lid list.";
        return false;
    }
    if (cavity.radius.empty() || cavity.n_theta <= 0 ||
        cavity.radius.size() != static_cast<std::size_t>(cavity.n_theta) * cavity.n_phi)
    {
        error = "Unscanned directions: build the helmet cavity first.";
        return false;
    }
    if (std::find(lid.begin(), lid.end(), true) == lid.end())
    {
        error = "Unscanned directions: no triangle is marked as lid.";
        return false;
    }

    // Every ray from a point outside the head also meets the head, so hits
    // alone prove nothing; the winding number does (+-1 inside, 0 outside).
    const double winding =
        igl::winding_number(head_vertices, head_faces, cavity.centre);
    if (!(std::abs(winding) > 0.5))
    {
        char text[32];
        std::snprintf(text, sizeof(text), "%.2f", winding);
        error = std::string("Unscanned directions: the head centre is not inside "
                            "the closed head mesh (winding number ") + text + ").";
        return false;
    }

    // Same reason as in build_helmet_cavity: cast against a copy scaled to a
    // fixed size, so libigl's absolute ray epsilon can't hide small triangles.
    const Eigen::RowVector3d vmin = head_vertices.colwise().minCoeff();
    const Eigen::RowVector3d vmax = head_vertices.colwise().maxCoeff();
    const double diagonal = (vmax - vmin).norm();
    if (!(diagonal > 0.0) || !std::isfinite(diagonal))
    {
        error = "Unscanned directions: the head has zero size.";
        return false;
    }
    const double scale = 1000.0 / diagonal;
    const Eigen::MatrixXd scaled = head_vertices * scale;
    const Eigen::RowVector3d origin = cavity.centre * scale;
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(scaled, head_faces);

    double total_weight = 0.0;
    double unscanned_weight = 0.0;
    double covered_weight = 0.0;
    std::size_t still_covered = 0;
    for (int i = 0; i < cavity.n_theta; ++i)
    {
        const double weight = std::sin((i + 0.5) * cavity.step);
        for (int j = 0; j < cavity.n_phi; ++j)
        {
            const std::size_t cell = static_cast<std::size_t>(i) * cavity.n_phi + j;
            total_weight += weight;
            igl::Hit<double> hit;
            if (!tree.intersect_ray(scaled, head_faces, origin,
                                    cell_direction(cavity.step, i, j), hit))
            {
                error = "Unscanned directions: a ray from the head centre leaves "
                        "without meeting the head, so the head mesh has a hole.";
                return false;
            }
            if (lid[static_cast<std::size_t>(hit.id)])
            {
                unscanned_weight += weight;
                if (!std::isnan(cavity.radius[cell]))
                {
                    covered_weight += weight;
                    cavity.radius[cell] = std::numeric_limits<double>::quiet_NaN();
                }
            }
            else if (!std::isnan(cavity.radius[cell]))
            {
                ++still_covered;
            }
        }
    }
    cavity.unscanned_fraction = unscanned_weight / total_weight;
    cavity.unscanned_covered_fraction = covered_weight / total_weight;
    if (still_covered == 0)
    {
        error = "Unscanned directions: no direction is both scanned and under the "
                "helmet, so there is nowhere to build. Check the fit.";
        return false;
    }
    return true;
}
