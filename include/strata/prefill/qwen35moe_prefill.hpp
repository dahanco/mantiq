// include/strata/prefill/qwen35moe_prefill.hpp - a chunked prompt path for the `qwen35moe` architecture.
//
// WHY THIS EXISTS SEPARATELY FROM `Prefill`.  `Prefill` (prefill.hpp) is the G2/qwen4exp prompt path: its layer
// plan is the hyper-connection residual stack (`hc` 4-wide, an `hc_lr` bottleneck and a control vector) and its
// kernels hard-code that geometry - `prefill.cpp` REFUSES any other model at its own geometry gate
// (`g.n_embd != 2560 || g.hc != 4 || g.hc_lr != 320 || ss.k != 10`), and the GDN kernels in `prefill/kernels.cu`
// are compiled for 48 value heads / 10240 conv channels.  `qwen35moe` is a different architecture: ONE residual
// stream, a lone RMSNorm per half, 32 value heads / 8192 conv channels, 16 query heads and top-8 routing.
//
// WHAT IS REUSED RATHER THAN REWRITTEN:
//   * `strata::prefill::Gemm` (gemm.hpp) - the batched cuBLAS projections, geometry-neutral.
//   * `strata::kernels`'s multi-token kernels (`verify_kernels.hpp`): `gdn_conv_l2_multi`, `gdn_ab_multi`,
//     `gdn_step_norm_multi` - the SAME kernels the verify window runs for this model today, already parametric
//     in the values this geometry needs.
//   * `to_f16`/`to_bf16`, and `core::qsa_layer` - the decode path's own full-attention layer, which the chunk
//     runs once per position (`qwen35moe_prefill.cu`, `Impl::qsa_mixer`).
//
// SCOPE OF THIS FILE'S LANDING (plan `qwen35moe-prefill-plan.md` section 9, Phases 1-3): the module's
// structure, the chunk arena, the layer plan of a plain-residual architecture, the GDN mixer half, the MoE
// half (router, shared expert, grouped routed experts, combine), and the full-attention half.
// The driver's CPU pool computes RAM-resident experts in batches; only cache hits run on the GPU.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace strata::prefill {

struct Qwen35MoePrefillStats {
    int64_t tokens = 0;     ///< prompt tokens read through this path
    int64_t chunks = 0;     ///< chunks the prompt was cut into
    double ms_total = 0;    ///< wall time of the last `run`
};

class Qwen35MoePrefill {
public:
    Qwen35MoePrefill();
    ~Qwen35MoePrefill();
    Qwen35MoePrefill(const Qwen35MoePrefill&) = delete;
    Qwen35MoePrefill& operator=(const Qwen35MoePrefill&) = delete;

    /// `chunk` is the tokens per chunk (clamped down to the prompt length).  `src`/`cache`/`host_res` follow
    /// `Prefill::init` and are kept for the expert phase; this phase allocates one arena of `bytes_needed`.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, strata::kernels::cpu::ExpertPool* pool = nullptr);

    int64_t chunk() const;

    /// Device bytes a chunk of `chunk` tokens needs (the arena `init` allocates).  A caller planning to lend a
    /// region out of the expert cache sizes it with this, exactly as `Prefill::bytes_needed` is used.
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// Positions [pos0, pos0 + n) holding `tokens`, through every layer, leaving the session state (the GDN
    /// recurrence and its conv history, and - once implemented - the attention KV pools) where the token path
    /// would have left it.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const Qwen35MoePrefillStats& stats() const { return stats_; }

    /// Called after every chunk with the chunk's residual rows (device, T x n_embd, valid until the next chunk)
    /// and the chunk's first position - the MTP draft layer's hook, as on `Prefill`.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;
    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Qwen35MoePrefillStats stats_;
};

}  // namespace strata::prefill
