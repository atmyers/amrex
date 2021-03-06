#ifndef AMREX_INT_H_
#define AMREX_INT_H_

#ifdef __cplusplus

#include <type_traits>
#include <cinttypes>

using amrex_long = int64_t;
namespace amrex {
    using Long = amrex_long;
}

#else

#ifdef AMREX_TYPECHECK
#define __attribute__(x) 
#define __extension__ 
#define __restrict 
#endif

#include <inttypes.h>

typedef int64_t amrex_long;

#endif

#endif
