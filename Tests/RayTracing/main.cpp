#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_IntVect.H>
#include <AMReX_Geometry.H>
#include <AMReX_Print.H>
#include <AMReX_Primitives.H>
#include <AMReX_STL.H>
#include <AMReX_Morton.H>
#include <AMReX_GpuContainers.H>

#include <string>
#include <vector>
#include <cstdint>
#include <numeric>
#include <algorithm>

#ifdef AMREX_USE_CUB
#include <cub/cub.cuh>
#endif

using namespace amrex;

struct TestParams
{
    IntVect size;
    int max_grid_size;
    int nlevs;
};

void testRay ();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    amrex::Print() << "Running ray tracing test. \n";
    testRay();

    amrex::Finalize();
}

void get_test_params (TestParams& params, const std::string& prefix)
{
    ParmParse pp(prefix);
    pp.get("size", params.size);
    pp.get("max_grid_size", params.max_grid_size);
    pp.get("nlevs", params.nlevs);
}

void getPermutationSequence (Gpu::DeviceVector<int>& indices_in,
                             const Gpu::DeviceVector<std::uint32_t>& codes_in) noexcept {
    const std::size_t num_items = codes_in.size();
    indices_in.resize(num_items);
#ifdef AMREX_USE_GPU
    auto indices_ptr = indices_in.dataPtr();
    amrex::ParallelFor(num_items, [=] AMREX_GPU_DEVICE (int i) noexcept { indices_ptr[i] = i; });

    Gpu::DeviceVector<int> indices_out(num_items);
    Gpu::DeviceVector<std::uint32_t> codes_out(num_items);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    std::size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    codes_in.dataPtr(), codes_out.dataPtr(),
                                    indices_in.dataPtr(), indices_out.dataPtr(), num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    codes_in.dataPtr(), codes_out.dataPtr(),
                                    indices_in.dataPtr(), indices_out.dataPtr(), num_items);

    Gpu::copy(Gpu::deviceToDevice, indices_out.begin(), indices_out.end(), indices_in.begin());
#else
    std::iota(indices_in.begin(), indices_in.end(), 0);
    {
        std::vector<std::pair<std::uint32_t,int>> pairs;
        for (std::size_t i = 0; i < num_items; ++i) {
            pairs.push_back(std::make_pair(codes_in[i], indices_in[i]));
        }
        std::sort(pairs.begin(), pairs.end());
        for (std::size_t i = 0; i < num_items; i++) {
            indices_in[i] = pairs[i].second;
        }
    }
#endif
}

void testRay ()
{
    BL_PROFILE("testRay");
    TestParams params;
    get_test_params(params, "ray");

    int is_per[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; i++)
        is_per[i] = false;

    Vector<IntVect> rr(params.nlevs-1);
    for (int lev = 1; lev < params.nlevs; lev++)
        rr[lev-1] = IntVect(AMREX_D_DECL(2,2,2));

    RealBox real_box;
    for (int n = 0; n < BL_SPACEDIM; n++)
    {
        real_box.setLo(n, -1001.);
        real_box.setHi(n,  1001.);
    }

    IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
    IntVect domain_hi(AMREX_D_DECL(params.size[0]-1,params.size[1]-1,params.size[2]-1));
    const Box base_domain(domain_lo, domain_hi);

    Vector<Geometry> geom(params.nlevs);
    geom[0].define(base_domain, &real_box, CoordSys::cartesian, is_per);
    for (int lev = 1; lev < params.nlevs; lev++) {
        geom[lev].define(amrex::refine(geom[lev-1].Domain(), rr[lev-1]),
                         &real_box, CoordSys::cartesian, is_per);
    }

    Vector<BoxArray> ba(params.nlevs);
    Vector<DistributionMapping> dm(params.nlevs);
    IntVect lo = IntVect(AMREX_D_DECL(0, 0, 0));
    IntVect size = params.size;
    for (int lev = 0; lev < params.nlevs; ++lev) {
        ba[lev].define(Box(lo, lo+params.size-1));
        ba[lev].maxSize(params.max_grid_size);
        dm[lev].define(ba[lev]);
        lo += size/2;
        size *= 2;
    }

    auto triangles = STL::read_ascii_stl("mesh/Sphericon.stl");
    amrex::Print() << "Have " << triangles.size() << " triangles total. \n";

    // copy triangles to device
    Gpu::DeviceVector<Triangle> triangles_d(triangles.size());
    Gpu::copy(Gpu::hostToDevice, triangles.begin(), triangles.end(), triangles_d.begin());

    // compute morton codes and bounding boxes for each primitive
    Gpu::DeviceVector<std::uint32_t> morton_codes_d(triangles.size());
    Gpu::DeviceVector<AABB> bboxes_d(triangles.size());
    auto plo = geom[0].ProbLoArray();
    auto phi = geom[0].ProbHiArray();
    auto code_ptr = morton_codes_d.dataPtr();
    auto bbox_ptr = bboxes_d.dataPtr();
    auto tris_ptr = triangles_d.dataPtr();
    amrex::ParallelFor(triangles.size(),
                       [=] AMREX_GPU_DEVICE (int i) noexcept {
                           const auto& tri = tris_ptr[i];
                           code_ptr[i] = Morton::get32BitCode(tri.getCentroid(), plo, phi);
                           bbox_ptr[i] = tri.getAABB();
                       });

    // Find ordering for the primitives that put them in morton order
    Gpu::DeviceVector<int> indices_d;
    getPermutationSequence(indices_d, morton_codes_d);

    // copy off and print the codes / permutation array
    std::vector<int> indices(indices_d.size());
    Gpu::copy(Gpu::deviceToHost, indices_d.begin(), indices_d.end(), indices.begin());

    std::vector<std::uint32_t> morton_codes(morton_codes_d.size());
    Gpu::copy(Gpu::deviceToHost, morton_codes_d.begin(), morton_codes_d.end(), morton_codes.begin());

    for (std::size_t i = 0; i < triangles.size(); ++i) {
        amrex::Print() << morton_codes[indices[i]] << " ";
    }
    amrex::Print() << "\n";

    // the way this test is set up, if we make it here we pass
    amrex::Print() << "pass \n";
}
