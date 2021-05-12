#ifndef AMREX_STL_H_
#define AMREX_STL_H_

#include <AMReX_Print.H>
#include <AMReX_Primitives.H>

#include <string>
#include <vector>

namespace amrex {
namespace STL {

std::vector<Triangle> read_ascii_stl (const std::string& filename) noexcept;

namespace detail {
    std::vector<std::string> split (std::string s, std::string delimiter) noexcept;
}
}
}

#endif
