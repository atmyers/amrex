#include <AMReX_Primitives.H>

namespace amrex {

std::ostream& operator<< (std::ostream& os, const Ray& ray) noexcept {
    os << "Origin: " << ray.origin << " Dest: " << ray.dest;
    return os;
}

std::ostream& operator<< (std::ostream& os, const AABB& box) noexcept
{
    os << "LE: " << box.left_edge << " RE: " << box.right_edge;
    return os;
}

std::ostream& operator<< (std::ostream& os, const Triangle& tri) noexcept
{
    os << "P0: " << tri.p0 << " " << "P1: " << tri.p1 << " " << "P2: " << tri.p2;
    return os;
}
}
