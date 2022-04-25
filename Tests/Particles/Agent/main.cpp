#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>

using namespace amrex;

class AgentContainer
    : public amrex::ParticleContainer<0, 0, 0, 1>
{

public:

    AgentContainer (const Vector<amrex::Geometry>            & a_geom,
                    const Vector<amrex::DistributionMapping> & a_dmap,
                    const Vector<amrex::BoxArray>            & a_ba,
                    const Vector<amrex::IntVect>             & a_rr)
        : amrex::ParticleContainer<0, 0, 0, 1>(a_geom, a_dmap, a_ba, a_rr)
    {}

    void initAgents ()
    {
        BL_PROFILE("initAgents");

        this->InitRandom(1000, 451, {{},{},{},{0}}, false);

        for (int lev = 0; lev <= finestLevel(); ++lev)
        {
            auto& plev  = GetParticles(lev);
            for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
            {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                auto& ptile = plev[std::make_pair(gid, tid)];
                auto& soa   = ptile.GetStructOfArrays();
                const size_t np = soa.numParticles();
                auto d_ptr = soa.GetIntData(0).data();

                amrex::ParallelForRNG( np,
                [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
                {
                    if (amrex::Random(engine) < 0.05) {
                        d_ptr[i] = 1;
                    }
                });
            }
        }
    }

    void moveAgents ()
    {
        BL_PROFILE("moveAgents");

        for (int lev = 0; lev <= finestLevel(); ++lev)
        {
            const auto dx = Geom(lev).CellSizeArray();
            auto& plev  = GetParticles(lev);

            for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
            {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                auto& ptile = plev[std::make_pair(gid, tid)];
                auto& aos   = ptile.GetArrayOfStructs();
                ParticleType* pstruct = &(aos[0]);
                const size_t np = aos.numParticles();

                amrex::ParallelForRNG( np,
                [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
                {
                    ParticleType& p = pstruct[i];

                    p.pos(0) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[0]);
#if AMREX_SPACEDIM > 1
                    p.pos(1) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[1]);
#endif
#if AMREX_SPACEDIM > 2
                    p.pos(2) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[2]);
#endif
                });
            }
        }
    }

   void interactAgents ()
   {
       BL_PROFILE("interactAgents");

       IntVect bin_size = {AMREX_D_DECL(4, 4, 4)};
       for (int lev = 0; lev < numLevels(); ++lev)
       {
           const Geometry& geom = Geom(lev);
           const auto dxi = geom.InvCellSizeArray();
           const auto plo = geom.ProbLoArray();
           const auto domain = geom.Domain();

           for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
           {
               amrex::DenseBins<ParticleType> bins;
               auto& ptile = ParticlesAt(lev, mfi);
               auto& aos   = ptile.GetArrayOfStructs();
               const size_t np = aos.numParticles();
               auto pstruct_ptr = aos().dataPtr();

               const Box& box = mfi.validbox();

               int ntiles = numTilesInBox(box, true, bin_size);

               bins.build(np, pstruct_ptr, ntiles, GetParticleBin{plo, dxi, domain, bin_size, box});
               auto inds = bins.permutationPtr();
               auto offsets = bins.offsetsPtr();

               auto& soa   = ptile.GetStructOfArrays();
               auto d_ptr = soa.GetIntData(0).data();

               amrex::ParallelForRNG( bins.numBins(),
               [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
               {
                   auto cell_start = offsets[i_cell];
                   auto cell_stop  = offsets[i_cell+1];
                   for (unsigned int i = cell_start; i < cell_stop; ++i) {
                       auto pindex = inds[i];
                       if (d_ptr[pindex] == 1) {
                           for (unsigned int j = cell_start; j < cell_stop; ++j) {
                               if (i == j) { continue; }
                               auto pindex2 = inds[j];
                               if (amrex::Random(engine) < 0.5) {
                                   d_ptr[pindex2] = 1;
                               }
                           }
                       }
                   }
               });
               amrex::Gpu::synchronize();
           }
       }
   }
};

struct TestParams
{
    IntVect size;
    int max_grid_size;
    int nsteps;
    int nlevs;
};

void runAgent();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    runAgent();

    amrex::Finalize();
}

void get_test_params(TestParams& params, const std::string& prefix)
{
    ParmParse pp(prefix);
    params.nlevs = 1;
    pp.get("size", params.size);
    pp.get("max_grid_size", params.max_grid_size);
    pp.get("nsteps", params.nsteps);
}

void runAgent ()
{
    BL_PROFILE("runAgent");
    TestParams params;
    get_test_params(params, "agent");

    int is_per[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; i++)
        is_per[i] = true;

    Vector<IntVect> rr(params.nlevs-1);
    for (int lev = 1; lev < params.nlevs; lev++)
        rr[lev-1] = IntVect(AMREX_D_DECL(2,2,2));

    RealBox real_box;
    for (int n = 0; n < BL_SPACEDIM; n++)
    {
        real_box.setLo(n, 0.0);
        real_box.setHi(n, 1024.0);
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
    for (int lev = 0; lev < params.nlevs; ++lev)
    {
        ba[lev].define(Box(lo, lo+params.size-1));
        ba[lev].maxSize(params.max_grid_size);
        dm[lev].define(ba[lev]);
        lo += size/2;
        size *= 2;
    }

    AgentContainer pc(geom, dm, ba, rr);

    pc.initAgents();

    for (int i = 0; i < params.nsteps; ++i)
    {
        pc.interactAgents();
        pc.moveAgents();
        pc.Redistribute();
        pc.WriteAsciiFile(amrex::Concatenate("particles", i, 5));
    }
}
