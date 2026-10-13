#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

// The space enclosed by a helmet shell, described as a radius map seen from a
// point inside it (the head's centre): for every direction, how far away the
// shell's inner face is.
//
// Directions that see no shell are holes in the map. Small holes are vents and
// get bridged by a smooth (harmonic) fill, so lattice continues underneath
// them. The one large hole is the opening the head goes through; it stays open.
//
// This replaces deciding "inside the helmet" from the sign of a distance to the
// helmet mesh. The shell is open (rim, vents), so a signed distance to it has
// no meaningful inside, and combined with the helmet's bounding box it filled
// the whole box with lattice.
//
// The map needs the shell to be star-shaped from the centre: every ray must
// meet the inner face before the outer one. build_helmet_cavity() checks this
// and fails rather than return a wrong map.
struct HelmetCavity
{
    Eigen::RowVector3d centre = Eigen::RowVector3d::Zero();

    // Cell (i, j) covers polar angle [i, i+1) * step measured from +Y (up) and
    // azimuth [j, j+1) * step measured as atan2(z, x) in [0, 2 pi).
    int n_theta = 0;
    int n_phi = 0;
    double step = 0.0;

    // Distance from the centre to the inner face, n_theta * n_phi, row-major
    // in theta. NaN where the direction looks out through the opening.
    std::vector<double> radius;

    // Returned for directions that look through the opening: a large finite
    // distance (never infinity, which would poison marching-tetrahedra
    // interpolation).
    double open_distance = 0.0;

    // Diagnostics, as fractions of all directions (solid-angle weighted).
    double inner_face_fraction = 0.0;
    double seen_fraction = 0.0;
    double open_fraction = 0.0;
    double largest_vent_fraction = 0.0;
    int vents_filled = 0;

    // |p - centre| - (r_inner(direction) - margin): negative inside the cavity,
    // stopping `margin` short of the inner face. Measured along the ray rather
    // than to the nearest shell point: under a bridged vent the nearest shell
    // point is the vent's edge, which made the alternative jump by up to 200 mm
    // across the boundary on the real helmet; this one changes smoothly.
    double radial_distance(const Eigen::RowVector3d& p, double margin) const;

    double radius_at(int i, int j) const;   // wraps j; i past a pole continues across it
};

bool build_helmet_cavity(
    const Eigen::MatrixXd& helmet_vertices,
    const Eigen::MatrixXi& helmet_faces,
    const Eigen::RowVector3d& centre,
    HelmetCavity& cavity,
    std::string& error,
    double resolution_degrees = 0.5,
    double max_vent_fraction = 0.05);
