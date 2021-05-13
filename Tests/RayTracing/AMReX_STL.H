#ifndef AMREX_STL_H_
#define AMREX_STL_H_

#include <AMReX_Print.H>
#include <AMReX_Primitives.H>

#include <string>
#include <vector>

namespace amrex {
namespace STL {

struct BinaryTriangle
{
    float normal[3];
    float p0[3];
    float p1[3];
    float p2[3];
    std::uint16_t attribute_byte_count;
};

std::vector<Triangle> read_stl (const std::string& filename) noexcept;

std::vector<Triangle> read_ascii_stl (const std::string& filename) noexcept;

std::vector<Triangle> read_binary_stl (const std::string& filename) noexcept;

namespace detail {
    std::vector<std::string> split (std::string s, std::string delimiter) noexcept;
}
}
}

#endif
