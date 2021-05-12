#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_IntVect.H>
#include <AMReX_Geometry.H>
#include <AMReX_Print.H>
#include <AMReX_Primitives.H>
#include <AMReX_STL.H>
#include <AMReX_Morton.H>

#include <string>
#include <vector>
#include <cstdint>
#include <numeric>
#include <algorithm>

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

void get_test_params(TestParams& params, const std::string& prefix)
{
    ParmParse pp(prefix);
    pp.get("size", params.size);
    pp.get("max_grid_size", params.max_grid_size);
    pp.get("nlevs", params.nlevs);
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

    // compute morton codes and bounding boxes for each primitive
    std::vector<std::uint32_t> morton_codes;
    std::vector<AABB> bboxes;
    auto plo = geom[0].ProbLoArray();
    auto phi = geom[0].ProbHiArray();
    for (const auto& tri : triangles) {
        morton_codes.push_back(Morton::get32BitCode(tri.getCentroid(), plo, phi));
        bboxes.emplace_back(tri.getAABB());
    }

    // compute ordering that puts primitives in morton order
    std::vector<int> indices(morton_codes.size());
    std::iota(indices.begin(), indices.end(), 0);
    {
        std::vector<std::pair<std::uint32_t,int>> pairs;
        for (std::size_t i = 0; i < morton_codes.size(); ++i) {
            pairs.push_back(std::make_pair(morton_codes[i], indices[i]));
        }
        std::sort(pairs.begin(), pairs.end());
        for (std::size_t i = 0; i < morton_codes.size(); i++) {
            indices[i] = pairs[i].second;
        }
    }

    for (int i = 0; i < triangles.size(); ++i) {
        amrex::Print() << morton_codes[indices[i]] << " ";
    }
    amrex:Print() << "\n";

    // the way this test is set up, if we make it here we pass
    amrex::Print() << "pass \n";
}
