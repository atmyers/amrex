#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>

using namespace amrex;

static constexpr int NSR = 6;
static constexpr int NSI = 1;
static constexpr int NAR = 1;
static constexpr int NAI = 1;

// Number of containers to redistribute together. Each gets a different
// number of runtime real/int components, exercising the fact that
// RedistributeMultiple allows this to vary across the batch.
static constexpr int NUM_CONTAINERS = 4;

void get_position_unit_cell (Real* r, const IntVect& nppc, int i_part)
{
    int nx = nppc[0];
#if AMREX_SPACEDIM > 1
    int ny = nppc[1];
#else
    int ny = 1;
#endif
#if AMREX_SPACEDIM > 2
    int nz = nppc[2];
#else
    int nz = 1;
#endif

    int ix_part = i_part/(ny * nz);
    int iy_part = (i_part % (ny * nz)) % ny;
    int iz_part = (i_part % (ny * nz)) / ny;

    r[0] = (0.5+ix_part)/nx;
    r[1] = (0.5+iy_part)/ny;
    r[2] = (0.5+iz_part)/nz;
}

class TestParticleContainer
    : public amrex::ParticleContainer<NSR, NSI, NAR, NAI>
{

public:

    TestParticleContainer (const Vector<amrex::Geometry>            & a_geom,
                           const Vector<amrex::DistributionMapping> & a_dmap,
                           const Vector<amrex::BoxArray>            & a_ba,
                           const Vector<amrex::IntVect>             & a_rr,
                           int a_num_runtime_real, int a_num_runtime_int)
        : amrex::ParticleContainer<NSR, NSI, NAR, NAI>(a_geom, a_dmap, a_ba, a_rr)
    {
        for (int i = 0; i < a_num_runtime_real; ++i) { AddRealComp(true); }
        for (int i = 0; i < a_num_runtime_int; ++i) { AddIntComp(true); }
    }

    void InitParticles (const amrex::IntVect& a_num_particles_per_cell)
    {
        BL_PROFILE("InitParticles");

        const int lev = 0;
        const Real* dx = Geom(lev).CellSize();
        const Real* plo = Geom(lev).ProbLo();

        const int num_ppc = AMREX_D_TERM( a_num_particles_per_cell[0],
                                         *a_num_particles_per_cell[1],
                                         *a_num_particles_per_cell[2]);

        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            DefineAndReturnParticleTile(lev, mfi.index(), mfi.LocalTileIndex());
        }

        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            const Box& tile_box  = mfi.tilebox();

            Gpu::HostVector<ParticleType> host_particles;
            std::array<Gpu::HostVector<ParticleReal>, NAR> host_real;
            std::array<Gpu::HostVector<int>, NAI> host_int;
            std::vector<Gpu::HostVector<ParticleReal> > host_runtime_real(NumRuntimeRealComps());
            std::vector<Gpu::HostVector<int> > host_runtime_int(NumRuntimeIntComps());

            for (IntVect iv = tile_box.smallEnd(); iv <= tile_box.bigEnd(); tile_box.next(iv))
            {
                for (int i_part=0; i_part<num_ppc; i_part++) {
                    Real r[3];
                    get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                    ParticleType p;
                    p.id()  = ParticleType::NextID();
                    p.cpu() = ParallelDescriptor::MyProc();
                    p.pos(0) = static_cast<ParticleReal>(plo[0] + (iv[0] + r[0])*dx[0]);
#if AMREX_SPACEDIM > 1
                    p.pos(1) = static_cast<ParticleReal>(plo[1] + (iv[1] + r[1])*dx[1]);
#endif
#if AMREX_SPACEDIM > 2
                    p.pos(2) = static_cast<ParticleReal>(plo[2] + (iv[2] + r[2])*dx[2]);
#endif

                    for (int i = 0; i < NSR; ++i) { p.rdata(i) = ParticleReal(p.id()); }
                    for (int i = 0; i < NSI; ++i) { p.idata(i) = int(p.id()); }

                    host_particles.push_back(p);
                    for (int i = 0; i < NAR; ++i) { host_real[i].push_back(ParticleReal(p.id())); }
                    for (int i = 0; i < NAI; ++i) { host_int[i].push_back(int(p.id())); }
                    for (int i = 0; i < NumRuntimeRealComps(); ++i) { host_runtime_real[i].push_back(ParticleReal(p.id())); }
                    for (int i = 0; i < NumRuntimeIntComps(); ++i) { host_runtime_int[i].push_back(int(p.id())); }
                }
            }

            auto& particle_tile = DefineAndReturnParticleTile(lev, mfi.index(), mfi.LocalTileIndex());
            auto old_size = particle_tile.GetArrayOfStructs().size();
            auto new_size = old_size + host_particles.size();
            particle_tile.resize(new_size);

            Gpu::copyAsync(Gpu::hostToDevice,
                           host_particles.begin(),
                           host_particles.end(),
                           particle_tile.GetArrayOfStructs().begin() + old_size);

            auto& soa = particle_tile.GetStructOfArrays();
            for (int i = 0; i < NAR; ++i) {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_real[i].begin(), host_real[i].end(),
                               soa.GetRealData(i).begin() + old_size);
            }
            for (int i = 0; i < NAI; ++i) {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_int[i].begin(), host_int[i].end(),
                               soa.GetIntData(i).begin() + old_size);
            }
            for (int i = 0; i < NumRuntimeRealComps(); ++i) {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_runtime_real[i].begin(), host_runtime_real[i].end(),
                               soa.GetRealData(NAR+i).begin() + old_size);
            }
            for (int i = 0; i < NumRuntimeIntComps(); ++i) {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_runtime_int[i].begin(), host_runtime_int[i].end(),
                               soa.GetIntData(NAI+i).begin() + old_size);
            }

            Gpu::streamSynchronize();
        }

        // Local redistribute to seat particles on their initial ranks
        Redistribute(0, finestLevel(), 0, 1);
    }

    void moveParticles (const IntVect& move_dir, int do_random)
    {
        BL_PROFILE("TestParticleContainer::moveParticles");

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
                ParticleType* pstruct = aos.data();
                const size_t np = aos.numParticles();

                if (do_random == 0)
                {
                    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept
                    {
                        ParticleType& p = pstruct[i];
                        p.pos(0) += static_cast<ParticleReal>(move_dir[0]*dx[0]);
#if AMREX_SPACEDIM > 1
                        p.pos(1) += static_cast<ParticleReal>(move_dir[1]*dx[1]);
#endif
#if AMREX_SPACEDIM > 2
                        p.pos(2) += static_cast<ParticleReal>(move_dir[2]*dx[2]);
#endif
                    });
                }
                else
                {
                    amrex::ParallelForRNG(np,
                    [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
                    {
                        ParticleType& p = pstruct[i];
                        p.pos(0) += static_cast<ParticleReal>((2*amrex::Random(engine)-1)*move_dir[0]*dx[0]);
#if AMREX_SPACEDIM > 1
                        p.pos(1) += static_cast<ParticleReal>((2*amrex::Random(engine)-1)*move_dir[1]*dx[1]);
#endif
#if AMREX_SPACEDIM > 2
                        p.pos(2) += static_cast<ParticleReal>((2*amrex::Random(engine)-1)*move_dir[2]*dx[2]);
#endif
                    });
                }
            }
        }
    }

    void checkAnswer () const
    {
        BL_PROFILE("TestParticleContainer::checkAnswer");

        AMREX_ALWAYS_ASSERT(OK());

        int num_rr = NumRuntimeRealComps();
        int num_ii = NumRuntimeIntComps();

        for (int lev = 0; lev <= finestLevel(); ++lev)
        {
            const auto& plev  = GetParticles(lev);
            for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
            {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                const auto& ptile = plev.at(std::make_pair(gid, tid));
                const auto& ptd = ptile.getConstParticleTileData();
                const size_t np = ptile.numParticles();

                AMREX_FOR_1D(np, i,
                {
                    for (int j = 0; j < NSR; ++j) {
                        AMREX_ALWAYS_ASSERT(ptd.m_aos[i].rdata(j) == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < NSI; ++j) {
                        AMREX_ALWAYS_ASSERT(ptd.m_aos[i].idata(j) == ptd.m_aos[i].id());
                    }
                    if constexpr (NAR > 0) {
                        for (int j = 0; j < NAR; ++j) {
                            AMREX_ALWAYS_ASSERT(ptd.m_rdata[j][i] == ptd.m_aos[i].id());
                        }
                    }
                    if constexpr (NAI > 0) {
                        for (int j = 0; j < NAI; ++j) {
                            AMREX_ALWAYS_ASSERT(ptd.m_idata[j][i] == ptd.m_aos[i].id());
                        }
                    }
                    for (int j = 0; j < num_rr; ++j) {
                        AMREX_ALWAYS_ASSERT(ptd.m_runtime_rdata[j][i] == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < num_ii; ++j) {
                        AMREX_ALWAYS_ASSERT(ptd.m_runtime_idata[j][i] == ptd.m_aos[i].id());
                    }
                });
            }
        }
    }
};

struct TestParams
{
    IntVect size;
    int max_grid_size;
    int num_ppc;
    int is_periodic;
    IntVect move_dir;
    int do_random;
    int nsteps;
    int nlevs;
    int do_regrid;
};

void testRedistributeMultiple ();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    amrex::Print() << "Running RedistributeMultiple test\n";
    testRedistributeMultiple();

    amrex::Finalize();
}

void get_test_params (TestParams& params, const std::string& prefix)
{
    ParmParse pp(prefix);
    pp.get("size", params.size);
    pp.get("max_grid_size", params.max_grid_size);
    pp.get("num_ppc", params.num_ppc);
    pp.get("is_periodic", params.is_periodic);
    pp.get("move_dir", params.move_dir);
    pp.get("do_random", params.do_random);
    pp.get("nsteps", params.nsteps);
    pp.get("nlevs", params.nlevs);
    pp.get("do_regrid", params.do_regrid);
}

void testRedistributeMultiple ()
{
    BL_PROFILE("testRedistributeMultiple");
    TestParams params;
    get_test_params(params, "redistribute");

    int is_per[] = {AMREX_D_DECL(params.is_periodic,
                                 params.is_periodic,
                                 params.is_periodic)};

    Vector<IntVect> rr(params.nlevs-1);
    for (int lev = 1; lev < params.nlevs; lev++) {
        rr[lev-1] = IntVect(AMREX_D_DECL(2,2,2));
    }

    RealBox real_box;
    for (int n = 0; n < BL_SPACEDIM; n++) {
        real_box.setLo(n, 0.0);
        real_box.setHi(n, params.size[n]);
    }

    IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
    IntVect domain_hi(AMREX_D_DECL(params.size[0]-1, params.size[1]-1, params.size[2]-1));
    const Box base_domain(domain_lo, domain_hi);

    Vector<Geometry> geom(params.nlevs);
    geom[0].define(base_domain, &real_box, CoordSys::cartesian, is_per);
    for (int lev = 1; lev < params.nlevs; lev++) {
        geom[lev].define(amrex::refine(geom[lev-1].Domain(), rr[lev-1]),
                         &real_box, CoordSys::cartesian, is_per);
    }

    Vector<BoxArray> ba(params.nlevs);
    Vector<DistributionMapping> dm(params.nlevs);
    IntVect lo(0);
    IntVect size = params.size;
    for (int lev = 0; lev < params.nlevs; ++lev) {
        ba[lev].define(Box(lo, lo+params.size-1));
        ba[lev].maxSize(params.max_grid_size);
        dm[lev].define(ba[lev]);
        lo += size/2;
        size *= 2;
    }

    // Create multiple particle containers sharing the same geometry/ba/dm,
    // each with a different number of runtime real/int components.
    Vector<std::unique_ptr<TestParticleContainer>> pcs(NUM_CONTAINERS);
    Vector<TestParticleContainer*> pc_ptrs(NUM_CONTAINERS);
    for (int c = 0; c < NUM_CONTAINERS; ++c) {
        pcs[c] = std::make_unique<TestParticleContainer>(geom, dm, ba, rr,
                                                         /*num_runtime_real=*/c,
                                                         /*num_runtime_int=*/(NUM_CONTAINERS-1-c));
        pcs[c]->InitParticles(IntVect(params.num_ppc));
        pcs[c]->checkAnswer();
        pc_ptrs[c] = pcs[c].get();
    }

    auto np_old = pcs[0]->TotalNumberOfParticles();

    for (int step = 0; step < params.nsteps; ++step)
    {
        for (int c = 0; c < NUM_CONTAINERS; ++c) {
            pcs[c]->moveParticles(params.move_dir, params.do_random);
        }

        // local RedistributeMultiple
        RedistributeMultiple(pc_ptrs, 0, pcs[0]->finestLevel(), 0, /*local=*/1);

        for (int c = 0; c < NUM_CONTAINERS; ++c) {
            pcs[c]->checkAnswer();
        }
    }

    if (params.do_regrid)
    {
        const int NProcs = ParallelDescriptor::NProcs();

        for (int lev = 0; lev < params.nlevs; ++lev) {
            Vector<int> pmap;
            for (int i = 0; i < ba[lev].size(); ++i) { pmap.push_back(i % NProcs); }
            DistributionMapping new_dm(pmap);
            for (int c = 0; c < NUM_CONTAINERS; ++c) {
                pcs[c]->SetParticleDistributionMap(lev, new_dm);
            }
        }

        // global RedistributeMultiple after regrid
        RedistributeMultiple(pc_ptrs, 0, pcs[0]->finestLevel(), 0, /*local=*/0);

        for (int c = 0; c < NUM_CONTAINERS; ++c) {
            pcs[c]->checkAnswer();
        }

        for (int lev = 0; lev < params.nlevs; ++lev) {
            Vector<int> pmap;
            for (int i = 0; i < ba[lev].size(); ++i) { pmap.push_back((i+1) % NProcs); }
            DistributionMapping new_dm(pmap);
            for (int c = 0; c < NUM_CONTAINERS; ++c) {
                pcs[c]->SetParticleDistributionMap(lev, new_dm);
            }
        }

        RedistributeMultiple(pc_ptrs, 0, pcs[0]->finestLevel(), 0, /*local=*/0);

        for (int c = 0; c < NUM_CONTAINERS; ++c) {
            pcs[c]->checkAnswer();
        }
    }

    if (geom[0].isAllPeriodic()) {
        for (int c = 0; c < NUM_CONTAINERS; ++c) {
            AMREX_ALWAYS_ASSERT(np_old == pcs[c]->TotalNumberOfParticles());
        }
    }

    // Cross-check: starting from the same initial state, RedistributeMultiple
    // and per-container Redistribute() must produce the same particle counts
    // per grid/tile.
    Vector<std::unique_ptr<TestParticleContainer>> pcs_ref(NUM_CONTAINERS);
    Vector<TestParticleContainer*> pc_ref_ptrs(NUM_CONTAINERS);
    for (int c = 0; c < NUM_CONTAINERS; ++c) {
        pcs_ref[c] = std::make_unique<TestParticleContainer>(geom, dm, ba, rr,
                                                             /*num_runtime_real=*/c,
                                                             /*num_runtime_int=*/(NUM_CONTAINERS-1-c));
        pcs_ref[c]->InitParticles(IntVect(params.num_ppc));
        pc_ref_ptrs[c] = pcs_ref[c].get();
    }

    Vector<std::unique_ptr<TestParticleContainer>> pcs_multi(NUM_CONTAINERS);
    Vector<TestParticleContainer*> pc_multi_ptrs(NUM_CONTAINERS);
    for (int c = 0; c < NUM_CONTAINERS; ++c) {
        pcs_multi[c] = std::make_unique<TestParticleContainer>(geom, dm, ba, rr,
                                                                /*num_runtime_real=*/c,
                                                                /*num_runtime_int=*/(NUM_CONTAINERS-1-c));
        pcs_multi[c]->InitParticles(IntVect(params.num_ppc));
        pc_multi_ptrs[c] = pcs_multi[c].get();
    }

    // Use a deterministic (non-random) move here so both groups get
    // identical displacements: interleaving random moves across the ref/
    // multi groups would consume the shared RNG stream in different orders.
    for (int c = 0; c < NUM_CONTAINERS; ++c) {
        pcs_ref[c]->moveParticles(params.move_dir, /*do_random=*/0);
        pcs_multi[c]->moveParticles(params.move_dir, /*do_random=*/0);
    }

    for (int c = 0; c < NUM_CONTAINERS; ++c) {
        pcs_ref[c]->Redistribute(0, pcs_ref[c]->finestLevel(), 0, 0);
    }
    RedistributeMultiple(pc_multi_ptrs, 0, pcs_multi[0]->finestLevel(), 0, /*local=*/0);

    for (int c = 0; c < NUM_CONTAINERS; ++c) {
        pcs_ref[c]->checkAnswer();
        pcs_multi[c]->checkAnswer();
        AMREX_ALWAYS_ASSERT(pcs_ref[c]->TotalNumberOfParticles() == pcs_multi[c]->TotalNumberOfParticles());
        for (int lev = 0; lev <= pcs_ref[c]->finestLevel(); ++lev) {
            for (MFIter mfi = pcs_ref[c]->MakeMFIter(lev); mfi.isValid(); ++mfi) {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                auto index = std::make_pair(gid, tid);
                auto const& ref_map = pcs_ref[c]->GetParticles(lev);
                auto const& multi_map = pcs_multi[c]->GetParticles(lev);
                auto ref_it = ref_map.find(index);
                auto multi_it = multi_map.find(index);
                std::size_t ref_np = (ref_it == ref_map.end()) ? 0 : ref_it->second.numParticles();
                std::size_t multi_np = (multi_it == multi_map.end()) ? 0 : multi_it->second.numParticles();
                AMREX_ALWAYS_ASSERT(ref_np == multi_np);
            }
        }
    }

    amrex::Print() << "pass\n";
}
