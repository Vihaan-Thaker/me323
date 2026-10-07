#pragma once

#include <Eigen/Core>

#include <string>

enum class LatticeType
{
    Auxetic,
    Gyroid
};

struct GeneratedMesh
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    double voxel_size = 0.0;
    double cell_size = 0.0;
};

bool generate_final_helmet(
    const Eigen::MatrixXd& head_vertices,
    const Eigen::MatrixXi& head_faces,
    const Eigen::MatrixXd& helmet_vertices,
    const Eigen::MatrixXi& helmet_faces,
    LatticeType lattice_type,
    GeneratedMesh& output,
    std::string& error);
