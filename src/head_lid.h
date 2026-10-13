#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

// Which triangles of the head mesh are its lid: the fan that
// sfm/export_for_me323.py adds to close the scan's open rim, so the head has a
// well-defined inside for the signed distance. The lid is not head surface.
// Beyond it lies head nobody measured (forehead below the cap, ears, neck), so
// nothing may be built against it.
//
// The exporter records the lid in the binary STL's 80-byte header as
//     ME323-HEAD v1 lid-faces <first> <count>
// meaning triangles [first, first + count) in file order. igl::readSTL keeps
// file order (it reads a triangle soup without merging vertices) and
// igl::writeSTL writes faces in order, so the range stays valid for copies of
// the head written after fitting, such as --export-fit's fitted_head.stl.
struct HeadLid
{
    std::vector<bool> faces;   // per triangle; empty when the file has no marker
    std::string header;        // the file's 80 header bytes, to copy into exports
    std::string note;          // one line for the log
};

// Reads the header of `path` and checks a marked range against the mesh read
// from that file: every marked triangle shares one corner (the fan's apex) and
// no other triangle touches it. No marker means a complete head (the GrabCAD
// model, for one): `faces` stays empty. A marker that doesn't fit the mesh is
// an error, since it means the header and the triangles came from different
// files.
bool read_head_lid(
    const std::string& path,
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    HeadLid& lid,
    std::string& error);

// Overwrites the 80-byte header of a binary STL, e.g. one igl::writeSTL wrote.
bool write_stl_header(
    const std::string& path,
    const std::string& header,
    std::string& error);
