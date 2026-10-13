#include "head_lid.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

namespace
{

constexpr std::size_t header_size = 80;
constexpr const char* marker = "ME323-HEAD";

// A point shared by this many triangles is the apex of a fan, not ordinary
// mesh: scan vertices have about 6, and the GrabCAD head's busiest has 23.
constexpr int fan_valence = 64;

using Corner = std::array<double, 3>;

Corner corner(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces,
              Eigen::Index face, int which)
{
    const Eigen::Index v = faces(face, which);
    return {vertices(v, 0), vertices(v, 1), vertices(v, 2)};
}

// Exact comparison on purpose: the STL stores each corner as float32, so every
// copy of the apex reads back to the same double.
bool has_corner(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces,
                Eigen::Index face, const Corner& point)
{
    for (int which = 0; which < 3; ++which)
    {
        if (corner(vertices, faces, face, which) == point)
        {
            return true;
        }
    }
    return false;
}

} // namespace

bool read_head_lid(
    const std::string& path,
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    HeadLid& lid,
    std::string& error)
{
    lid = HeadLid{};
    if (vertices.cols() != 3 || faces.cols() != 3 || faces.rows() == 0 ||
        faces.minCoeff() < 0 || faces.maxCoeff() >= vertices.rows())
    {
        error = "Head lid: the head mesh is empty or invalid.";
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        error = "Head lid: could not open " + path + " to read its header.";
        return false;
    }
    std::string header(header_size, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header_size));
    if (file.gcount() != static_cast<std::streamsize>(header_size))
    {
        error = "Head lid: " + path + " is shorter than an STL header.";
        return false;
    }
    lid.header = header;

    if (header.compare(0, std::strlen(marker), marker) != 0)
    {
        lid.note = path + " has no lid marker: treated as a complete head.";
        std::map<Corner, int> uses;
        for (Eigen::Index face = 0; face < faces.rows(); ++face)
        {
            for (int which = 0; which < 3; ++which)
            {
                ++uses[corner(vertices, faces, face, which)];
            }
        }
        int busiest = 0;
        for (const auto& entry : uses)
        {
            busiest = std::max(busiest, entry.second);
        }
        if (busiest >= fan_valence)
        {
            lid.note += " WARNING: one point is shared by " + std::to_string(busiest)
                + " triangles, which looks like the lid closing a scan. If this is a"
                  " scan, re-export it with sfm/export_for_me323.py so the lid is"
                  " marked; otherwise material may be built against the lid.";
        }
        return true;
    }

    const std::string text(header.c_str());   // up to the first NUL
    int version = 0;
    long long first = -1;
    long long count = -1;
    char extra = 0;
    const int fields = std::sscanf(
        text.c_str(), "ME323-HEAD v%d lid-faces %lld %lld %c",
        &version, &first, &count, &extra);
    if (fields != 3 || version != 1)
    {
        error = "Head lid: unrecognised marker in " + path + ": \"" + text + "\".";
        return false;
    }
    if (count < 3 || first < 0 || first + count > faces.rows())
    {
        error = "Head lid: the marker in " + path + " names triangles " +
            std::to_string(first) + ".." + std::to_string(first + count - 1) +
            ", but the mesh has " + std::to_string(faces.rows()) +
            ". The header belongs to a different file; re-export the head.";
        return false;
    }

    // The fan's apex: the corner of the first marked triangle that all of them
    // share.
    const Eigen::Index begin = static_cast<Eigen::Index>(first);
    const Eigen::Index end = static_cast<Eigen::Index>(first + count);
    bool found = false;
    Corner apex{};
    for (int which = 0; which < 3 && !found; ++which)
    {
        apex = corner(vertices, faces, begin, which);
        found = true;
        for (Eigen::Index face = begin + 1; face < end && found; ++face)
        {
            found = has_corner(vertices, faces, face, apex);
        }
    }
    if (!found)
    {
        error = "Head lid: the triangles the marker in " + path + " names do not"
                " form a fan around one point. The header belongs to a different"
                " file; re-export the head.";
        return false;
    }
    for (Eigen::Index face = 0; face < faces.rows(); ++face)
    {
        if ((face < begin || face >= end) && has_corner(vertices, faces, face, apex))
        {
            error = "Head lid: triangle " + std::to_string(face) + " in " + path +
                " touches the lid's apex but is not marked as lid. The header"
                " belongs to a different file; re-export the head.";
            return false;
        }
    }

    lid.faces.assign(static_cast<std::size_t>(faces.rows()), false);
    std::fill(lid.faces.begin() + begin, lid.faces.begin() + end, true);
    lid.note = path + ": triangles " + std::to_string(first) + ".." +
        std::to_string(first + count - 1) + " (" + std::to_string(count) +
        ") are the lid closing the scan, not head surface.";
    return true;
}

bool write_stl_header(
    const std::string& path,
    const std::string& header,
    std::string& error)
{
    if (header.size() != header_size)
    {
        error = "STL header must be exactly 80 bytes.";
        return false;
    }
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file)
    {
        error = "Could not open " + path + " to write its header.";
        return false;
    }
    file.seekp(0);
    file.write(header.data(), static_cast<std::streamsize>(header_size));
    file.flush();
    if (!file)
    {
        error = "Could not write the header of " + path + ".";
        return false;
    }
    return true;
}
