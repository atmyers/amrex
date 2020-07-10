#ifndef AMREX_GPUALLOCATORS_H_
#define AMREX_GPUALLOCATORS_H_

#include <map>
#include <memory>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include <AMReX_Print.H>
#include <AMReX_Arena.H>
#include <AMReX_GpuDevice.H>

#ifdef AMREX_USE_CUDA
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#endif // AMREX_USE_CUDA

namespace amrex {

    template <typename T>
    struct RunOnGpu : std::false_type {};

    struct ArenaAllocatorTraits {
        typedef std::true_type propagate_on_container_copy_assignment;
        typedef std::true_type propagate_on_container_move_assignment;
        typedef std::true_type propagate_on_container_swap;
        typedef std::true_type is_always_equal;
    };

    template<typename T>
    class ArenaAllocator
        : public ArenaAllocatorTraits
    {
    public :

        using value_type = T;

        inline value_type* allocate(std::size_t n)
        {
	    value_type* result = nullptr;
            result = (value_type*) The_Arena()->alloc(n * sizeof(T));
	    return result;
        }

        inline void deallocate(value_type* ptr, std::size_t)
        {
            The_Arena()->free(ptr);
        }

        inline std::pair<value_type*, std::size_t>
        grow (value_type* ptr, std::size_t min_size, std::size_t max_size)
        {
            auto r = The_Arena()->grow(ptr, min_size*sizeof(T), max_size*sizeof(T));
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }

        inline std::pair<value_type*, std::size_t>
        shrink (value_type* ptr, std::size_t desired_size)
        {
            auto r = The_Arena()->shrink(ptr, desired_size*sizeof(T));
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }
    };

    template<typename T>
    class DeviceArenaAllocator
        : public ArenaAllocatorTraits
    {
    public :

        using value_type = T;

        inline value_type* allocate(std::size_t n)
        {
	    value_type* result = nullptr;
            result = (value_type*) The_Device_Arena()->alloc(n * sizeof(T));
	    return result;
        }

        inline void deallocate(value_type* ptr, std::size_t)
        {
            The_Device_Arena()->free(ptr);
        }

        inline std::pair<value_type*, std::size_t>
        grow (value_type* ptr, std::size_t min_size, std::size_t max_size)
        {
            auto r = The_Device_Arena()->grow(ptr, min_size*sizeof(T), max_size*sizeof(T));
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }

        inline std::pair<value_type*, std::size_t>
        shrink (value_type* ptr, std::size_t desired_size)
        {
            auto r = The_Device_Arena()->shrink(ptr, desired_size*sizeof(T));
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }
    };

    template<typename T>
    class PinnedArenaAllocator
        : public ArenaAllocatorTraits
    {
    public :

        using value_type = T;

        inline value_type* allocate(std::size_t n)
        {
	    value_type* result = nullptr;
            result = (value_type*) The_Pinned_Arena()->alloc(n * sizeof(T));
	    return result;
        }

        inline void deallocate(value_type* ptr, std::size_t)
        {
            The_Pinned_Arena()->free(ptr);
        }

        inline std::pair<value_type*, std::size_t>
        grow (value_type* ptr, std::size_t min_size, std::size_t max_size)
        {
            auto r = The_Pinned_Arena()->grow(ptr, min_size*sizeof(T), max_size*sizeof(T));
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }

        inline std::pair<value_type*, std::size_t>
        shrink (value_type* ptr, std::size_t desired_size)
        {
            auto r = The_Pinned_Arena()->shrink(ptr, desired_size*sizeof(T));
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }
    };

    template<typename T>
    class ManagedArenaAllocator
        : public ArenaAllocatorTraits
    {
    public :

        using value_type = T;

        inline value_type* allocate(std::size_t n)
        {
	    value_type* result = nullptr;
            result = (value_type*) The_Managed_Arena()->alloc(n * sizeof(T));
	    return result;
        }

        inline void deallocate(value_type* ptr, std::size_t)
        {
            The_Managed_Arena()->free(ptr);
        }

        inline std::pair<value_type*, std::size_t>
        grow (value_type* ptr, std::size_t min_size, std::size_t max_size)
        {
            auto r = The_Managed_Arena()->grow(ptr, min_size*sizeof(T), max_size*sizeof(T));
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }

        inline std::pair<value_type*, std::size_t>
        shrink (value_type* ptr, std::size_t desired_size)
        {
            auto r = The_Managed_Arena()->shrink(ptr, desired_size*sizeof(T));
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }
    };

    template<typename T>
    class PolymorphicAllocator
    {
    public :

        using value_type = T;

        PolymorphicAllocator () : m_use_gpu_aware_mpi(ParallelDescriptor::UseGpuAwareMpi()) {}

        inline value_type* allocate(std::size_t n)
        {
	    value_type* result = nullptr;
            if (m_use_gpu_aware_mpi)
            {
                result = (value_type*) The_Device_Arena()->alloc(n * sizeof(T));
            }
            else
            {
                result = (value_type*) The_Pinned_Arena()->alloc(n * sizeof(T));
            }
	    return result;
        }

        inline void deallocate(value_type* ptr, std::size_t)
        {
            if (m_use_gpu_aware_mpi)
            {
                The_Device_Arena()->free(ptr);
            }
            else
            {
                The_Pinned_Arena()->free(ptr);
            }
        }

        bool m_use_gpu_aware_mpi;

        template <class U, class V>
        friend bool
        operator== (PolymorphicAllocator<U> const& a, PolymorphicAllocator<V> const& b) noexcept
        {
            return a.m_use_gpu_aware_mpi == b.m_use_gpu_aware_mpi;
        }

        template <class U, class V>
        friend bool
        operator!= (PolymorphicAllocator<U> const& a, PolymorphicAllocator<V> const& b) noexcept
        {
            return a.m_use_gpu_aware_mpi != b.m_use_gpu_aware_mpi;
        }

        inline std::pair<value_type*, std::size_t>
        grow (value_type* ptr, std::size_t min_size, std::size_t max_size)
        {
            std::pair<void*, std::size_t> r;
            if (ParallelDescriptor::UseGpuAwareMpi()) {
                r = The_Device_Arena()->grow(ptr, min_size*sizeof(T), max_size*sizeof(T));
            } else {
                r = The_Pinned_Arena()->grow(ptr, min_size*sizeof(T), max_size*sizeof(T));
            }
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }

        inline std::pair<value_type*, std::size_t>
        shrink (value_type* ptr, std::size_t desired_size)
        {
            std::pair<void*, std::size_t> r;
            if (ParallelDescriptor::UseGpuAwareMpi()) {
                r = The_Device_Arena()->shrink(ptr, desired_size*sizeof(T));
            } else {
                r = The_Pinned_Arena()->shrink(ptr, desired_size*sizeof(T));
            }
            return std::make_pair((value_type*)(r.first), r.second/sizeof(T));
        }
    };

#ifdef AMREX_USE_GPU
    template <class T>
    using DefaultAllocator = amrex::ArenaAllocator<T>;

    template <typename T>
    struct RunOnGpu<ArenaAllocator<T> > : std::true_type {};

    template <typename T>
    struct RunOnGpu<DeviceArenaAllocator<T> > : std::true_type {};

    template <typename T>
    struct RunOnGpu<ManagedArenaAllocator<T> > : std::true_type {};

#endif // AMREX_USE_GPU

    template <typename T>
    class StdAllocator
        : public std::allocator<T>
    {
    public:
        using value_type = T;

        inline std::pair<value_type*, std::size_t>
        grow (value_type* ptr, std::size_t /*min_size*/, std::size_t max_size)
        {
            return std::make_pair(std::allocator<T>::allocate(max_size), max_size);
        }

        inline std::pair<value_type*, std::size_t>
        shrink (value_type* ptr, std::size_t desired_size)
        {
            return std::make_pair(std::allocator<T>::allocate(desired_size), desired_size);
        }
    };

#ifdef AMREX_USE_GPU
    template <class T>
    using DefaultAllocator = amrex::ArenaAllocator<T>;
#else
    template <class T>
    using DefaultAllocator = amrex::StdAllocator<T>;
#endif // AMREX_USE_GPU
} // namespace amrex

#endif // AMREX_GPUALLOCATORS_H_
