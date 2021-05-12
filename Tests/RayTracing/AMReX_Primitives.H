#ifndef AMREX_PRIMITIVES_H_
#define AMREX_PRIMITIVES_H_

#include <AMReX_REAL.H>
#include <AMReX_Dim3.H>
#include <AMReX_Algorithm.H>

#include <iostream>

namespace amrex
{

struct Ray {
    XDim3 origin, dest;
};
std::ostream& operator<< (std::ostream& os, const Ray& ray) noexcept;

struct AABB {
    XDim3 left_edge;
    XDim3 right_edge;
};
std::ostream& operator<< (std::ostream& os, const AABB& box) noexcept;

struct Triangle
{
    XDim3 p0, p1, p2;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    XDim3 getCentroid () const noexcept
    {
        return XDim3{(p0.x + p1.x + p2.x) / Real(3.0),
                     (p0.y + p1.y + p2.y) / Real(3.0),
                     (p0.z + p1.z + p2.z) / Real(3.0)};
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    AABB getAABB () const noexcept {
        return AABB{elemwiseMin(p0, p1, p2), elemwiseMax(p0, p1, p2)};
    }
};
std::ostream& operator<< (std::ostream& os, const Triangle& tri) noexcept;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real dot (const XDim3& a, const XDim3& b) noexcept { return a.x*b.x + a.y*b.y + a.z*b.z;}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
XDim3 cross (const XDim3& a, const XDim3& b) noexcept {
    return XDim3{a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}

}

#endif
