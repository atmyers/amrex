#ifndef AMREX_PARTICLECONTAINERREDISTRIBUTEMULTIPLE_H_
#define AMREX_PARTICLECONTAINERREDISTRIBUTEMULTIPLE_H_
#include <AMReX_Config.H>

#include <AMReX_ParticleCommunication.H>
#include <AMReX_ParticleContainerBase.H>
#include <AMReX_TypeTraits.H>
#include <AMReX_Vector.H>

#include <map>
#include <utility>

namespace amrex {

/**
* \brief Redistributes several ParticleContainers of the same type at once.
*
* This is equivalent to calling `pcs[i]->Redistribute(lev_min, lev_max, nGrow,
* local, remove_negative)` on each container in turn, except that the MPI
* handshake and metadata-exchange round (which each single-container
* Redistribute() pays for individually) is merged into a single round shared
* across the whole batch. Packing, sending, and unpacking of the actual
* particle data remains per-container.
*
* All containers in `pcs` must share the same ParGDB (i.e. the same
* geometry/BoxArray/DistributionMapping at every level); they may otherwise
* differ in their runtime AddRealComp/AddIntComp counts.
*
* \tparam PC A ParticleContainer_impl (or subclass) type.
*/
template <typename PC>
requires (IsParticleContainer<PC>::value)
void RedistributeMultiple (Vector<PC*>& pcs,
                           int lev_min = 0, int lev_max = -1, int nGrow = 0,
                           int local = 0, bool remove_negative = true)
{
    if (pcs.empty()) { return; }
    if (pcs.size() == 1) {
        pcs[0]->Redistribute(lev_min, lev_max, nGrow, local, remove_negative);
        return;
    }

    BL_PROFILE("amrex::RedistributeMultiple()");

    for (auto* pc : pcs) {
        AMREX_ALWAYS_ASSERT(pc != nullptr);
        AMREX_ALWAYS_ASSERT(pc->finestLevel() == pcs[0]->finestLevel());
        for (int lev = 0; lev <= pcs[0]->finestLevel(); ++lev) {
            AMREX_ALWAYS_ASSERT(BoxArray::SameRefs(pc->ParticleBoxArray(lev),
                                                    pcs[0]->ParticleBoxArray(lev)));
            AMREX_ALWAYS_ASSERT(DistributionMapping::SameRefs(pc->ParticleDistributionMap(lev),
                                                               pcs[0]->ParticleDistributionMap(lev)));
        }
    }

    const int n = static_cast<int>(pcs.size());

    Vector<ParticleCopyOp> ops(n);
    Vector<ParticleCopyPlan> plans(n);
    Vector<Vector<std::map<std::pair<int, int>, int> > > new_sizes(n);
    Vector<int> theEffectiveFinestLevel(n);
    Vector<int> resolved_lev_max(n, lev_max);

    for (int c = 0; c < n; ++c) {
        pcs[c]->RedistributePartition(lev_min, resolved_lev_max[c], nGrow, local,
                                      remove_negative,
                                      ops[c], new_sizes[c], theEffectiveFinestLevel[c]);
    }

    Vector<ParticleCopyPlan*> plan_ptrs(n);
    for (int c = 0; c < n; ++c) {
        plans[c].buildLocal(*pcs[c], ops[c],
                            pcs[c]->h_redistribute_int_comp,
                            pcs[c]->h_redistribute_real_comp, local);
        plan_ptrs[c] = &plans[c];
    }

    const ParticleBufferMap& map = pcs[0]->BufferMap();

    ParticleCopyPlan::MultiExchange exch;
    ParticleCopyPlan::buildMPIStartMultiple(*pcs[0], map, plan_ptrs, exch);

    // by default, this uses The_Arena();
    Vector<amrex::PODVector<char, PolymorphicArenaAllocator<char> > > snd_buffers(n);
    Vector<amrex::PODVector<char, PolymorphicArenaAllocator<char> > > rcv_buffers(n);

    if (ParticleContainerBase::use_comms_arena) {
        for (int c = 0; c < n; ++c) {
            snd_buffers[c].setArena(The_Comms_Arena());
            rcv_buffers[c].setArena(The_Comms_Arena());
        }
    }

    for (int c = 0; c < n; ++c) {
        packBuffer(*pcs[c], ops[c], plans[c], snd_buffers[c]);
        pcs[c]->ClearMovedParticles(lev_min, resolved_lev_max[c],
                                    theEffectiveFinestLevel[c], new_sizes[c]);
    }

    ParticleCopyPlan::buildMPIFinishMultiple(map, plan_ptrs, exch);

    // Communicate and unpack each container's payload individually. This
    // mirrors the tail of Redistribute_impl exactly, just looped over the
    // batch; merging these per-container messages into shared payload
    // buffers/sends is a candidate follow-up if profiling shows the
    // per-container message count (rather than the handshake) still
    // dominates for a given workload.
    for (int c = 0; c < n; ++c) {
        auto& plan = plans[c];
        auto& snd_buffer = snd_buffers[c];
        auto& rcv_buffer = rcv_buffers[c];

        if (ParallelDescriptor::UseGpuAwareMpi())
        {
            communicateParticlesStart(*pcs[c], plan, snd_buffer, rcv_buffer);
            pcs[c]->ReserveForRedistribute(plan);
            unpackBuffer(*pcs[c], plan, snd_buffer, RedistributeUnpackPolicy());
            communicateParticlesFinish(plan);
            unpackRemotes(*pcs[c], plan, rcv_buffer, RedistributeUnpackPolicy());
        }
        else
        {
            Gpu::Device::streamSynchronize();
            Gpu::PinnedVector<char> pinned_snd_buffer;
            Gpu::PinnedVector<char> pinned_rcv_buffer;

            if (snd_buffer.arena()->isPinned()) {
                Gpu::Device::streamSynchronize();
                communicateParticlesStart(*pcs[c], plan, snd_buffer, pinned_rcv_buffer);
            } else {
                pinned_snd_buffer.resize(snd_buffer.size());
                Gpu::dtoh_memcpy_async(pinned_snd_buffer.dataPtr(), snd_buffer.dataPtr(), snd_buffer.size());
                Gpu::Device::streamSynchronize();
                communicateParticlesStart(*pcs[c], plan, pinned_snd_buffer, pinned_rcv_buffer);
            }

            pcs[c]->ReserveForRedistribute(plan);

            rcv_buffer.resize(pinned_rcv_buffer.size());
            unpackBuffer(*pcs[c], plan, snd_buffer, RedistributeUnpackPolicy());
            communicateParticlesFinish(plan);
            Gpu::htod_memcpy_async(rcv_buffer.dataPtr(), pinned_rcv_buffer.dataPtr(), pinned_rcv_buffer.size());
            unpackRemotes(*pcs[c], plan, rcv_buffer, RedistributeUnpackPolicy());
        }
    }

    Gpu::Device::streamSynchronize();
    for (int c = 0; c < n; ++c) {
        AMREX_ASSERT(numParticlesOutOfRange(*pcs[c], lev_min, resolved_lev_max[c], nGrow) == 0);
    }
}

} // namespace amrex

#endif // AMREX_PARTICLECONTAINERREDISTRIBUTEMULTIPLE_H_
