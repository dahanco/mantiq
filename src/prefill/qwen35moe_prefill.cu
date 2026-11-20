// src/prefill/qwen35moe_prefill.cu - see include/strata/prefill/qwen35moe_prefill.hpp.
//
// PHASE 1 (plan `qwen35moe-prefill-plan.md` section 9): the module, the chunk arena, the plain-residual layer
// plan, and the GDN mixer half.  PHASE 2 adds the MoE half: the router, the shared expert, the routed experts
// read once per distinct expert and computed in one grouped launch, and the combine.  PHASE 3 adds the
// full-attention half: the ten full-attention layers run the decode path's own `core::qsa_layer`, one position
// at a time, so the chunk's attention is the reference arithmetic rather than a second transcription of it.
#include "strata/prefill/qwen35moe_prefill.hpp"

#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/failure.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
// The same two constants `Prefill` carves: the dequantization scratch of the largest dense weight, and cuBLAS's
// workspace.  qwen35moe's widest projection is `attn_qkv` [8192, 2048] = 16.8M elements, well inside 32M.
constexpr int64_t GEMM_SCRATCH_ELEMS = 32ll << 20;   // FP16 elements
constexpr size_t GEMM_WS_BYTES = 32u << 20;
constexpr size_t ALIGN = 256;
// The same ceiling `Prefill` uses for `--prefill auto`: a logical prompt batch of up to 8192 tokens.  A run
// starts at the requested chunk (or 8192 for auto) and halves it until the arena fits the free VRAM - as
// llama.cpp's n_batch grows to its KV/workspace budget rather than stopping at a fixed ubatch.  Measured on
// the 5311-token Qwopus prompt: 512 chunks ran 201.6 tok/s, 1024 257.3 and 2048 295.7, so stopping at 2048
// left throughput on the table.
constexpr int64_t kMaxPrefillChunk = 8192;
constexpr int64_t kMinPrefillChunk = 256;
// Room left outside the arena for what is allocated after it (cuBLAS state, the verify window, the commit
// graph).  The driver reserves its own `--vram-reserve-mib` above this; here the check is against what is
// actually free when the arena is carved, so an over-subscribed WDDM allocation cannot silently page.
constexpr uint64_t kArenaHeadroomBytes = 256ull << 20;
// The blobs a layer's routed experts are staged into when the source cannot hand the GPU a device address
// over PCIe (`ExpertSource::device_alias`) - `kExpertRingSlots` of them at once, so a layer with more distinct
// routed experts than this takes several grouped launches.  Bounded deliberately: the alternative is one
// blob per distinct expert, which for a wide chunk is most of `n_expert` and does not fit beside the cache.
constexpr int64_t kExpertRingSlots = 24;
// The widest tile `shared_expert_multi` takes in one call.  It serves a wider window by running itself once per
// 32 columns, so the q8_1 activation image it reads has to hold 32 columns and no more - the buffers below are
// sized for this, not for the chunk.
constexpr int kSharedCols = 32;
// The queries the batched full-attention half hands `qsa_decode_attn_batch` at once.  Its per-query partials
// follow the tile's OWN capacity (`chunks * n_head * (head_dim + 2)`), and the indexer-free selection's ceiling
// is 32K cells, so this bounds that scratch at `kQsaAttnBatch` times ~8.5 MB in the worst case - which is why
// the attention is tiled rather than run over the whole chunk.
constexpr int64_t kQsaAttnBatch = 8;
// The phase counters `MNTQ_PREFILL_PROF` fills.  Slots 0-5 are the layer's own halves, 6-9 break the MoE half
// down because it is the one that dominates a chunk: the router and shared expert, the host-side plan, the
// per-expert staging plus the grouped launches, and the combine.
constexpr int kNPhase = 10;

// The gate/up and down types `native_expert_grouped` has a kernel for (iq_kernels.cu:1289 and :1313).  They are
// checked BEFORE the call: that function answers an unsupported type with `std::exit(1)`, which no caller can
// recover from.  Note the two sets differ - type 20 (IQ4_NL) is a down type only.
bool grouped_gu_type_ok(int t) {
    switch (t) {
        case 1: case 16: case 17: case 18: case 21: case 22: case 23: case 29: case 19: case 42:
        case 13: case 12: case 14: case 8: case 11: case 10: return true;
        default: return false;
    }
}
bool grouped_d_type_ok(int t) {
    switch (t) {
        case 1: case 16: case 17: case 18: case 20: case 21: case 22: case 29: case 19: case 23: case 42:
        case 13: case 12: case 14: case 8: case 11: case 10: return true;
        default: return false;
    }
}

// ------------------------------------------------------------------ the residual add
// The plain-residual architecture consumes a half's output with a bare `R += bo` (verify.cpp:869 and :1034), so
// this is the whole write-back of both halves.
__global__ void add_rows_kernel(float* __restrict__ R, const float* __restrict__ bo, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) R[i] += bo[i];
}
void add_rows(float* R, const float* bo, int64_t n, cudaStream_t cs) {
    if (n <= 0) return;
    add_rows_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, cs>>>(R, bo, n);
}

__global__ void set_count_kernel(int32_t* dst, int32_t count) {
    *dst = count;
}

// ------------------------------------------------------------------ the expert-result staging buffer
// `h_parts` is filled by the CPU expert pool and then handed to the device once per layer - `T*K*N` floats, 134
// MB at the widest chunk.  Through a pageable `std::vector`, `cudaMemcpyAsync` has to bounce the data through
// an internal page-locked staging buffer, and on this link (PCIe 3.0 x16) that cost the upload about a fifth of
// its bandwidth: the copy engine only reaches full rate when it can read the source pages directly.  So the
// buffer is page-locked instead (`cudaHostAlloc`), which is what makes the upload a single DMA.
//
// It is a CAPACITY, not a size: the pool's jobs are handed raw row pointers into it, the CPU fills exactly the
// entries it owns and the resident plan fills the rest on the device, so the bytes between the two writers are
// never read and the buffer must never be zeroed or reallocated between the jobs being built and the upload.
struct PinnedFloats {
    float* p = nullptr;
    size_t cap = 0;
    bool pinned = false;
    std::vector<float> fallback;   ///< only when the host refuses to pin: the DMA then stages through the driver
    float* data() const { return p; }
    /// Grows to hold `n` floats.  Returns false only when BOTH the pinned and the pageable host allocation are
    /// refused, which the caller reports rather than running on a buffer that is too small.  A refused
    /// `cudaHostAlloc` is not fatal: the pinning is a bandwidth optimization, so the buffer falls back to
    /// pageable memory (the upload then copies through the driver's own staging, ~10.9 rather than 13.0 GB/s)
    /// and says so - the machine that refuses it has 30 GiB pinned for the expert arena already.
    bool reserve(size_t n) {
        if (n <= cap) return true;
        if (pinned && p != nullptr) cudaFreeHost(p);
        p = nullptr;
        cap = 0;
        pinned = false;
        void* q = nullptr;
        if (cudaHostAlloc(&q, n * sizeof(float), cudaHostAllocDefault) == cudaSuccess) {
            p = static_cast<float*>(q);
            cap = n;
            pinned = true;
            return true;
        }
        std::fprintf(stderr, "mntq generate: qwen35moe prompt path: could not pin the %zu MiB expert staging "
                             "buffer (%s); falling back to pageable memory\n",
                     n * sizeof(float) >> 20, cudaGetErrorString(cudaGetLastError()));
        fallback.resize(n);
        p = fallback.data();
        cap = n;
        return true;
    }
    ~PinnedFloats() {
        if (pinned && p != nullptr) cudaFreeHost(p);
    }
    PinnedFloats() = default;
    PinnedFloats(const PinnedFloats&) = delete;
    PinnedFloats& operator=(const PinnedFloats&) = delete;
};

// ------------------------------------------------------------------ the CPU experts' merge
// The CPU results arrive as a COMPACT image -row `c` is the `c`-th entry a CPU expert computed, not the `c`-th
// row of `parts`- because that is what makes their upload and the resident experts independent: the DMA writes
// no row the grouped launch touches, so the two can run at once.  This puts every row back where the combine
// reads it.  A grid dimension is capped at 65535 entries, while an 8192-token chunk with top-8 routing
// produces 65536 entries, so the y dimension strides over the rows rather than assuming one block per entry.
__global__ void scatter_rows_kernel(float* __restrict__ dst, const int32_t* __restrict__ rows,
                                    const float* __restrict__ src, int width, int n_rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = blockIdx.y; r < n_rows; r += gridDim.y)
        if (c < width) dst[(size_t) rows[r] * width + c] = src[(size_t) r * width + c];
}

// ------------------------------------------------------------------ the chunk arena
// A bump allocator over one device block, 256-byte aligned - the shape `prefill.cpp`'s `Alloc` has.  In
// `count_only` mode it walks the same list to answer `bytes_needed` without touching the device.
struct Alloc {
    uint8_t* cur = nullptr;
    size_t left = 0;
    size_t used = 0;
    bool count_only = false;
    bool ok = true;

    void* take_bytes(size_t bytes) {
        const size_t a = (bytes + ALIGN - 1) & ~(size_t) (ALIGN - 1);
        used += a;
        if (count_only) return (void*) (uintptr_t) 16;   // non-null: the walk only sums
        if (a > left) { ok = false; return nullptr; }
        void* p = cur;
        cur += a;
        left -= a;
        return p;
    }
    template <typename T>
    T* take(size_t count) {
        return reinterpret_cast<T*>(take_bytes(count * sizeof(T)));
    }
};

/// The device buffers of one chunk.  The order here IS the layout (both `carve_chunk` and `bytes_needed` walk
/// it), so a buffer added here must be added in the same place in both.
struct Buffers {
    float* emb = nullptr;           ///< (T, n_embd) the gathered embeddings
    float* R = nullptr;             ///< (T, n_embd) the single residual stream of the plain architecture
    float* mixed = nullptr;         ///< (T, n_embd) the half's normalized read (FP32)
    uint16_t* mixed_f16 = nullptr;  ///< ... its FP16 image (the `native`/`f16` GEMMs' activations)
    uint16_t* mixed_bf = nullptr;   ///< ... its BF16 image (the resident-BF16 GEMMs')
    float* bo = nullptr;            ///< (T, n_embd) the half's output, before the residual add
    int32_t* tokens = nullptr;      ///< (T) the chunk's token ids, as the embedding gather wants them
    int32_t* n_keep = nullptr;      ///< (1) the tokens the GDN commit runs (the whole chunk)
    float* qkv = nullptr;           ///< (T, conv_channels) the GDN q|k|v projection
    float* hbuf = nullptr;          ///< (T, conv_channels) the conv + L2 norm, as q|k|v
    float* gate = nullptr;          ///< (T, v_heads)
    float* beta = nullptr;          ///< (T, v_heads)
    float* z = nullptr;             ///< (T, value_dim) the GDN gate projection
    float* y = nullptr;             ///< (T, value_dim) the recurrence's output (FP32)
    uint16_t* y_f16 = nullptr;      ///< ... its FP16 image (the ssm_out GEMM's activations)
    // ---- the MoE half
    float* logits = nullptr;         ///< (T, n_expert) the router's logits
    int32_t* ids = nullptr;          ///< (T, top_k) the routed experts, in the ROUTER's order
    float* weights = nullptr;        ///< (T, top_k) the router's weights, already renormalized
    float* parts = nullptr;          ///< (T*top_k, n_embd) the routed experts, row = token*top_k + slot
    float* cpu_parts = nullptr;      ///< (T*top_k, n_embd) the CPU experts' results, row = the CPU entry's index
    int32_t* cpu_dst = nullptr;      ///< (T*top_k) a CPU entry -> the row of `parts` the merge copies it to
    float* shared = nullptr;         ///< (T, n_embd) the shared expert, ALREADY gated by its sigmoid
    float* sh_gate = nullptr;        ///< (T, n_ff) the shared expert's gate projection
    float* sh_up = nullptr;          ///< (T, n_ff) ... its up projection
    float* sh_g = nullptr;           ///< (T) ... its per-token scalar gate (pre-sigmoid)
    uint8_t* sh_scratch = nullptr;   ///< shared_expert_scratch_bytes(n_ff)
    uint8_t* sh_q8_1 = nullptr;      ///< the shared expert's q8_1 activation image, kSharedCols columns
    uint8_t* xq = nullptr;           ///< (T) the routed experts' q8_1 activation rows, n_embd/32 blocks each
    uint8_t* ex_scratch = nullptr;   ///< native_expert_scratch_bytes(T*top_k, n_ff)
    uint8_t* ring = nullptr;         ///< kExpertRingSlots staged expert blobs, expert_layout().max_blob apart
    unsigned long long* g_ptr = nullptr;  ///< (n_expert) each distinct expert's device blob address
    int32_t* g_start = nullptr;      ///< (n_expert + 1) group -> its first entry, in group order
    int32_t* g_count = nullptr;      ///< (batches) the groups each staged batch runs
    int32_t* g_dst = nullptr;        ///< (T*top_k) entry -> the row of `parts` it computes
    int32_t* g_tok = nullptr;        ///< (T*top_k) entry -> the token whose activation it reads
    uint16_t* gemm_scratch = nullptr;
    void* gemm_ws = nullptr;
    // ---- the full-attention half
    float* qsa_qf = nullptr;            ///< (T, n_head, 2 * head_dim) the raw `attn_q` projection, q | gate
    float* qsa_q = nullptr;             ///< (T, n_head, head_dim) its FIRST half, normed and rotated
    float* qsa_kc = nullptr;            ///< (T, n_head_kv, head_dim) the normed and rotated keys
    float* qsa_vc = nullptr;            ///< (T, n_head_kv, head_dim) the values, as projected
    float* qsa_attn = nullptr;          ///< (T, n_head, head_dim) the attention output
    uint16_t* qsa_attn_h = nullptr;     ///< ... its FP16 image (the o-projection's activation)
    int32_t* qsa_ids = nullptr;         ///< (kQsaAttnBatch, cap) the identity selection of one query tile
    int32_t* qsa_steps = nullptr;       ///< (T, kStepCount) the per-position step records
    float* qsa_attn_scratch = nullptr;  ///< kQsaAttnBatch times the decode kernel's per-query partials
};

/// The largest selection the session's context can produce: the indexer-free full-attention layers select every
/// cached cell, so a query's width is `min(pos + 1, kTopkMaxCells)` and the whole path is sized from the biggest
/// one.  Never zero, so the carve can size the selection and attention scratch from it.
int64_t qsa_cap_for(const core::SessionState& ss) {
    return std::max<int64_t>(1, std::min<int64_t>(ss.max_cells, strata::kernels::kTopkMaxCells));
}

void carve_chunk(int64_t T, const core::ModelGeometry& g, int64_t k, int64_t qsa_cap, Alloc& o, Buffers& b) {
    const size_t N = (size_t) g.n_embd, C = (size_t) g.ssm_conv_channels, ZV = (size_t) g.ssm_value_dim;
    const size_t HV = (size_t) g.ssm_v_heads;
    const size_t NE = (size_t) g.n_expert, FF = (size_t) g.n_ff, K = (size_t) k;
    const size_t ENT = (size_t) T * K;
    b.emb = o.take<float>((size_t) T * N);
    b.R = o.take<float>((size_t) T * N);
    b.mixed = o.take<float>((size_t) T * N);
    b.mixed_f16 = o.take<uint16_t>((size_t) T * N);
    b.mixed_bf = o.take<uint16_t>((size_t) T * N);
    b.bo = o.take<float>((size_t) T * N);
    b.tokens = o.take<int32_t>((size_t) T);
    b.n_keep = o.take<int32_t>(1);
    b.qkv = o.take<float>((size_t) T * C);
    b.hbuf = o.take<float>((size_t) T * C);
    b.gate = o.take<float>((size_t) T * HV);
    b.beta = o.take<float>((size_t) T * HV);
    b.z = o.take<float>((size_t) T * ZV);
    b.y = o.take<float>((size_t) T * ZV);
    b.y_f16 = o.take<uint16_t>((size_t) T * ZV);
    // ---- the MoE half.  The expert blobs a layer stages are `max_blob` apiece, and only a native pack has a
    // ring at all: a canonical pack answers `max_blob` with the fixed BLOB and is refused before it is used.
    const auto& lay = strata::kernels::cpu::expert_layout();
    const size_t max_blob = lay.native ? (size_t) lay.max_blob : 0;
    b.logits = o.take<float>((size_t) T * NE);
    b.ids = o.take<int32_t>(ENT);
    b.weights = o.take<float>(ENT);
    b.parts = o.take<float>(ENT * N);
    // The CPU experts' own image, in CPU-entry order rather than in `parts`'s row order: keeping it compact is
    // what lets its upload and the resident experts' launch run at once (see `routed_hybrid`), and its device
    // pair is the index list the merge puts each row back with.
    b.cpu_parts = o.take<float>(ENT * N);
    b.cpu_dst = o.take<int32_t>(ENT);
    b.shared = o.take<float>((size_t) T * N);
    b.sh_gate = o.take<float>((size_t) T * FF);
    b.sh_up = o.take<float>((size_t) T * FF);
    b.sh_g = o.take<float>((size_t) T);
    b.sh_scratch = (uint8_t*) o.take_bytes((size_t) strata::kernels::shared_expert_scratch_bytes((int64_t) FF));
    b.sh_q8_1 = (uint8_t*) o.take_bytes(strata::kernels::native_q8_1_bytes((int) N, kSharedCols));
    b.xq = (uint8_t*) o.take_bytes((size_t) T * (N / 32) * 36);
    b.ex_scratch =
        (uint8_t*) o.take_bytes(strata::kernels::native_expert_scratch_bytes((int64_t) ENT, (int64_t) FF));
    b.ring = (uint8_t*) o.take_bytes((size_t) kExpertRingSlots * max_blob);
    b.g_ptr = o.take<unsigned long long>(NE);
    b.g_start = o.take<int32_t>(NE + 1);
    b.g_count = o.take<int32_t>((size_t) (NE + (size_t) kExpertRingSlots - 1) / (size_t) kExpertRingSlots);
    b.g_dst = o.take<int32_t>(ENT);
    b.g_tok = o.take<int32_t>(ENT);
    b.gemm_scratch = o.take<uint16_t>((size_t) GEMM_SCRATCH_ELEMS);
    b.gemm_ws = o.take_bytes(GEMM_WS_BYTES);
    // ---- the full-attention half.  `qsa_cap` is the largest selection the session's context can produce
    // (`min(max_cells, kTopkMaxCells)`): the identity selection returns every cached cell, so a query's width
    // and the tile capacity `iota_rows`/`qsa_decode_attn_batch` are sized from it follow the position.
    const size_t NH = (size_t) g.n_head, NKV = (size_t) g.n_head_kv, HD = (size_t) g.head_dim;
    b.qsa_qf = o.take<float>((size_t) T * NH * 2 * HD);
    b.qsa_q = o.take<float>((size_t) T * NH * HD);
    b.qsa_kc = o.take<float>((size_t) T * NKV * HD);
    b.qsa_vc = o.take<float>((size_t) T * NKV * HD);
    b.qsa_attn = o.take<float>((size_t) T * NH * HD);
    b.qsa_attn_h = o.take<uint16_t>((size_t) T * NH * HD);
    b.qsa_ids = o.take<int32_t>((size_t) kQsaAttnBatch * (size_t) qsa_cap);
    b.qsa_steps = o.take<int32_t>((size_t) T * (size_t) strata::kernels::kStepCount);
    b.qsa_attn_scratch = o.take<float>(
        (size_t) kQsaAttnBatch * (size_t) strata::kernels::qsa_decode_attn_scratch_floats(qsa_cap,
                                                                                           core::qsa_shapes(g)));
}

/// The GDN state is ONE contiguous block per GDN layer: the recurrence (states x v_heads x states floats) then
/// the conv history (conv_channels x (d_conv - 1)).  `verify.cpp:455` builds the same number.
int64_t gdn_state_floats(const core::ModelGeometry& g) {
    return (int64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
           g.ssm_conv_channels * (g.ssm_d_conv - 1);
}

/// Y = X . W^T for a weight that is either resident (FP16 or BF16) or native GGUF blocks - the choice
/// `prefill.cpp`'s `native_proj`/`bf16_proj` make, kept in one place because this path has both kinds.
bool proj(Gemm& gm, const core::WeightRef* w, const uint16_t* x_f16, const uint16_t* x_bf, float* y, int64_t T,
          const std::string& name, std::string& err, int64_t ldy = 0) {
    if (w == nullptr) { err = "qwen35moe prefill: " + name + " is missing"; return false; }
    if (w->kind == core::WeightKind::F16InF32 && w->data != nullptr) {
        gm.f16(x_f16, (const uint16_t*) w->data, y, T, w->ne1, w->ne0, ldy);
        return true;
    }
    if (w->kind == core::WeightKind::Bf16InF32 && w->data != nullptr) {
        if (x_bf == nullptr) {
            err = "qwen35moe prefill: " + name + " is BF16 and has no BF16 activation image";
            return false;
        }
        gm.bf16(x_bf, (const uint16_t*) w->data, y, T, w->ne1, w->ne0, ldy);
        return true;
    }
    if (w->native_data == nullptr || w->native_type < 0) {
        err = "qwen35moe prefill: " + name + " has no GGUF blocks (run with --native)";
        return false;
    }
    gm.native(x_f16, w->native_type, w->native_data, y, T, w->ne1, w->ne0, ldy);
    return true;
}

}  // namespace

// ================================================================== Impl
struct Qwen35MoePrefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t chunk = 0;
    cudaStream_t cs = nullptr;
    cudaStream_t cs2 = nullptr;   ///< the CPU results' upload: its own stream so the DMA runs beside the experts
    cudaEvent_t ev_up = nullptr;  ///< marks that upload's completion on `cs2`
    Gemm gemm;
    uint8_t* arena = nullptr;
    size_t arena_bytes = 0;
    Buffers bufs;
    int64_t gdn_floats = 0;
    std::vector<int64_t> gdn_idx;   ///< per trunk layer: its index among the GDN layers, or -1
    std::vector<int64_t> qsa_idx;   ///< per trunk layer: its index among the full-attention layers, or -1
    // The MoE half's host-side plan, rebuilt per layer between the router's synchronization and the launches.
    std::vector<int32_t> h_ids;      ///< (T*top_k) the router's ids, read back from the device
    std::vector<int32_t> h_start;    ///< (n_expert + 1) group -> its first entry
    std::vector<int32_t> h_count;    ///< (n_expert) entries per expert, then the fill cursors
    std::vector<int32_t> h_ord;      ///< the distinct experts, ascending
    std::vector<int32_t> h_dst, h_tok, h_count_batch;
    std::vector<int32_t> cpu_dst;  ///< (T*top_k) the CPU entries' rows of `parts`, in the order they were built
    std::vector<unsigned long long> h_ptr;  ///< one device blob address per batch slot
    std::vector<uint8_t> h_stage;    ///< the copy `copy_blob` writes for a source whose blob does not stay put
    strata::kernels::cpu::ExpertPool* pool = nullptr;  // borrowed from the driver, never a second worker pool
    std::vector<float> h_mixed;
    PinnedFloats h_cpu;     ///< the CPU experts' results, compact, page-locked (see PinnedFloats)
    std::vector<int32_t> h_steps;  ///< (T * kStepCount) the batched full-attention half's per-position records
    std::vector<uint8_t> h_act;
    std::vector<strata::kernels::cpu::ActQ> h_act_q2;
    std::vector<strata::kernels::cpu::ExpertJobMulti> cpu_jobs;
    std::vector<std::vector<uint8_t>> cpu_blobs;

    // `MNTQ_PREFILL_PROF` only: `routed_hybrid`'s six stages, so `moe_stage`'s single number can be told apart from
    // the CPU pool's.  rb_ms[i] is the wall time between mark i and mark i+1, rb_n its call count.
    double rb_ms[6] = {};
    int64_t rb_n = 0;
    long long rb_cpu_jobs = 0, rb_res_experts = 0, rb_cpu_tokens = 0;
    // The same for `moe_route` (slot 6): the router's projection, the shared expert, the ids read-back that the
    // group plan needs on the host, and the q8_1 activation image.  They sum to slot 6.
    double rr_ms[4] = {};
    int64_t rr_n = 0;
    // The shared expert's HOST-side time alone (the call returning, with every kernel queued but not drained),
    // so `rr_ms[1]` can be split into the launch/recursion cost and the GPU tail that follows it.
    double rr_host_ms = 0;

    // PHASE 5 TUNING ONLY, and off unless `MNTQ_PREFILL_PROF` is set: the wall time of each phase of the layer,
    // measured with a stream sync on either side so the host clock attributes every microsecond of GPU work to
    // the phase that issued it.  The syncs themselves cost a little, so these numbers are a BREAKDOWN and not a
    // substitute for the end-to-end time `stats_.ms_total` reports.
    bool prof = false;
    double t_ph[kNPhase] = {};
    int64_t n_ph[kNPhase] = {};
    /// A scoped (or explicitly stopped) timer.  Both ends synchronize the stream, so the host clock charges
    /// every microsecond of queued GPU work to the phase that issued it.  `stop()` exists for the MoE half,
    /// whose stages are separated by host work rather than by a scope.
    struct Phase {
        Impl* m = nullptr;
        int i = 0;
        Clock::time_point t0;
        Phase() = default;
        Phase(Impl* mm, int ii) : m(mm->prof ? mm : nullptr), i(ii) {
            if (m) { cudaStreamSynchronize(m->cs); t0 = Clock::now(); }
        }
        void stop() {
            if (m == nullptr) return;
            cudaStreamSynchronize(m->cs);
            m->t_ph[i] += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            ++m->n_ph[i];
            m = nullptr;
        }
        ~Phase() { stop(); }
    };

    ~Impl() {
        if (arena != nullptr) cudaFree(arena);
        if (cs2 != nullptr) cudaStreamDestroy(cs2);
        if (ev_up != nullptr) cudaEventDestroy(ev_up);
    }

    /// One half's read under the plain-residual architecture: `mixed = rms_norm(R, w)` - the lone RMSNorm
    /// verify.cpp uses in place of the hyper-connection read (`residual_norm_read`), then the two activation
    /// images the GEMMs take.  The norm itself is `strata::kernels::rms_norm_weighted`, the very kernel the
    /// verify window calls, so the read is bit-identical to it.
    bool read_norm(const core::LayerView& v, const char* name, int64_t T, std::string& err) {
        const core::ModelGeometry& gm = *g;
        const core::WeightRef* w = v.get(name);
        if (w == nullptr) { err = "qwen35moe prefill: " + v.name(name) + " is missing"; return false; }
        if (w->kind != core::WeightKind::F32) {
            err = "qwen35moe prefill: " + v.name(name) + " must be F32 for the plain-residual read";
            return false;
        }
        if (cudaMemcpyAsync(bufs.mixed, bufs.R, (size_t) T * gm.n_embd * 4, cudaMemcpyDeviceToDevice, cs) !=
            cudaSuccess) {
            err = "qwen35moe prefill: the residual read's copy failed";
            return false;
        }
        strata::kernels::rms_norm_weighted(bufs.mixed, (const float*) w->data, T, gm.n_embd, EPS, cs);
        to_f16(bufs.mixed, bufs.mixed_f16, T * gm.n_embd, cs);
        to_bf16(bufs.mixed, bufs.mixed_bf, T * gm.n_embd, cs);
        return true;
    }

    /// The GDN mixer half of layer `l`: qkv -> conv + L2 -> alpha/beta -> gate -> the recurrence (which COMMITS
    /// the state) -> ssm_out.  The kernels are the verify window's own, parametric in the geometry, so nothing
    /// here depends on the 48-head/10240-channel constants `prefill/kernels.cu` is compiled with.
    bool gdn_mixer(const core::LayerView& v, int64_t l, int64_t T, std::string& err) {
        const core::ModelGeometry& gm = *g;
        const int64_t N = gm.n_embd, C = gm.ssm_conv_channels, ZV = gm.ssm_value_dim;
        const int64_t HK = gm.ssm_k_heads, HV = gm.ssm_v_heads;
        const core::WeightRef* wqkv = v.get("attn_qkv.weight");
        const core::WeightRef* wg = v.get("attn_gate.weight");
        const core::WeightRef* wout = v.get("ssm_out.weight");
        const core::WeightRef* wa = v.get("ssm_alpha.weight");
        const core::WeightRef* wb = v.get("ssm_beta.weight");
        const core::WeightRef* wc = v.get("ssm_conv1d.weight");
        const core::WeightRef* wnm = v.get("ssm_norm.weight");
        const core::WeightRef* wdt = v.get("ssm_dt.bias");
        const core::WeightRef* wsa = v.get("ssm_a");
        if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) {
            err = "qwen35moe prefill: layer " + std::to_string(l) + " is missing a GDN tensor";
            return false;
        }
        const int64_t gi = gdn_idx[(size_t) l];
        float* state = ss->gdn_state + (size_t) gi * (size_t) gdn_floats;
        float* conv = state + (uint64_t) gm.ssm_state_size * gm.ssm_v_heads * gm.ssm_state_size;

        if (!proj(gemm, wqkv, bufs.mixed_f16, bufs.mixed_bf, bufs.qkv, T, v.name("attn_qkv.weight"), err))
            return false;
        if (!proj(gemm, wg, bufs.mixed_f16, bufs.mixed_bf, bufs.z, T, v.name("attn_gate.weight"), err))
            return false;
        // Commit recurrence/conv state in bounded windows, while projections and MoE retain the wide batch.
        // Each tile reads the history written by the previous tile on this same stream.
        for (int64_t t0 = 0; t0 < T; t0 += strata::kernels::kVerifyMaxT) {
            const int nt = (int) std::min<int64_t>(strata::kernels::kVerifyMaxT, T - t0);
            set_count_kernel<<<1, 1, 0, cs>>>(bufs.n_keep, nt);
            strata::kernels::gdn_conv_l2_multi(conv, bufs.qkv + t0 * C, (const float*) wc->data,
                                               bufs.hbuf + t0 * C, (int) C, (int) (2 * HK), EPS, nt, cs, 0);
            strata::kernels::gdn_ab_multi(bufs.mixed + t0 * N, (const uint16_t*) wa->data,
                                          (const uint16_t*) wb->data, (const float*) wdt->data,
                                          (const float*) wsa->data, bufs.gate + t0 * HV, bufs.beta + t0 * HV,
                                          (int) N, (int) HV, nt, cs);
            strata::kernels::gdn_conv_commit(conv, bufs.qkv + t0 * C, (int) C, bufs.n_keep, cs);
            strata::kernels::gdn_step_norm_multi(state, bufs.hbuf + t0 * C, (int) C,
                                                 bufs.gate + t0 * HV, bufs.beta + t0 * HV, bufs.z + t0 * ZV,
                                                 (const float*) wnm->data, EPS, bufs.y + t0 * ZV,
                                                 (int) HK, (int) HV, nt, bufs.n_keep, true, cs, 0);
        }
        to_f16(bufs.y, bufs.y_f16, T * ZV, cs);
        if (!proj(gemm, wout, bufs.y_f16, nullptr, bufs.bo, T, v.name("ssm_out.weight"), err)) return false;
        return true;
    }

    /// The full-attention half of layer `l`, in the decode path's own arithmetic (the identity selection over
    /// the KV pool, the Q/K norms, the rotation, the append, the attention and the sigmoid gate) - see the two
    /// implementations below.  The batched one is the default; the per-position one is the reference and the
    /// fallback for the KV formats the batched one does not cover.
    bool qsa_mixer(int64_t l, int64_t T, int64_t pos0, std::string& err) {
        const int64_t qi = qsa_idx[(size_t) l];
        if (qi < 0 || ss->qsa_states == nullptr) {
            err = "qwen35moe prefill: layer " + std::to_string(l) +
                  " is full attention and the session has no QSA state";
            return false;
        }
        const core::QsaState& st = ss->qsa_states[qi];
        // The batched path reads the resident FP16 pool directly and drives no block movers.  Every other KV
        // format (streamed, INT8, Q4_0, K8V4) needs the ones `qsa_layer` drives, so it takes the per-position
        // path; `MNTQ_QSA_SERIAL=1` forces that path for the A/B.
        static const bool serial = std::getenv("MNTQ_QSA_SERIAL") != nullptr;
        if (serial || st.kv_mode != 0 || st.kv_int8 || st.kv_q4 || st.kv_hybrid)
            return qsa_serial(l, T, pos0, st, err);
        return qsa_batched(l, T, pos0, st, err);
    }

    /// The per-position full-attention half: the decode path's own `core::qsa_layer`, one token at a time.
    /// This is the reference arithmetic, so it cannot drift from the token path's.
    bool qsa_serial(int64_t l, int64_t T, int64_t pos0, const core::QsaState& st, std::string& err) {
        const core::ModelGeometry& gm = *g;
        const int64_t N = gm.n_embd;
        for (int64_t t = 0; t < T; ++t) {
            if (!core::qsa_layer(*wt, gm, l, pos0 + t, /*pos_base=*/0, st, ss->qsa_bufs,
                                 bufs.mixed + (size_t) t * (size_t) N, bufs.bo + (size_t) t * (size_t) N, cs, err)) {
                err = "qwen35moe prefill: layer " + std::to_string(l) + ", position " +
                      std::to_string(pos0 + t) + ": " + err;
                return false;
            }
        }
        return true;
    }

    /// The BATCHED full-attention half: the projections are GEMMs over the whole chunk, K/V and the selection
    /// go in one round of launches each, and the attention is `qsa_decode_attn_batch` in tiles.  This is the
    /// shape `prefill.cpp`'s glm half takes, with two differences that follow from the architecture: there is
    /// no DSA indexer, so the selection is the IDENTITY (`iota_rows`), and its 16-head/2-KV-head geometry is
    /// served by the 8:1 GQA instantiation of the decode attention.
    ///
    /// The numbers this trades for its speed are the batched path's own, and `prefill.cpp` accepts the same
    /// ones: the projections take FP16 activations where the token path takes Q8_K/Q8_0, so the result is
    /// FP32-level rather than bit-identical to `qsa_serial`.
    bool qsa_batched(int64_t l, int64_t T, int64_t pos0, const core::QsaState& st, std::string& err) {
        const core::ModelGeometry& gm = *g;
        const int64_t NH = gm.n_head, NKV = gm.n_head_kv, HD = gm.head_dim;
        const int64_t gqa = NKV > 0 ? NH / NKV : 0;
        // `qsa_decode_attn_batch` answers an unsupported geometry, a null pool or a missing buffer with
        // `std::exit(1)`, which no caller can recover from - so every one of its demands is checked HERE first.
        if (HD != 256 || NKV <= 0 || NH != gqa * NKV || (gqa != 8 && gqa != 12)) {
            err = "qwen35moe prefill: layer " + std::to_string(l) + "'s attention geometry (" +
                  std::to_string(NH) + " heads / " + std::to_string(NKV) + " kv heads / head_dim " +
                  std::to_string(HD) + ") has no batched decode-attention kernel";
            return false;
        }
        if (st.page_table == nullptr || st.k_pool == nullptr || st.v_pool == nullptr) {
            err = "qwen35moe prefill: layer " + std::to_string(l) + "'s FP16 KV pool is not resident";
            return false;
        }
        const strata::kernels::QsaShapes s = core::qsa_shapes(gm);

        const core::LayerView v(*wt, l);
        const core::WeightRef* wq = v.get("attn_q.weight");
        const core::WeightRef* wk = v.get("attn_k.weight");
        const core::WeightRef* wv = v.get("attn_v.weight");
        const core::WeightRef* wo = v.get("attn_output.weight");
        const core::WeightRef* wqn = v.get("attn_q_norm.weight");
        const core::WeightRef* wkn = v.get("attn_k_norm.weight");
        const char* missing = !wq ? "attn_q.weight" : !wk ? "attn_k.weight" : !wv ? "attn_v.weight"
                              : !wo ? "attn_output.weight" : !wqn ? "attn_q_norm.weight"
                              : !wkn ? "attn_k_norm.weight" : nullptr;
        if (missing != nullptr) {
            err = "qwen35moe prefill: layer " + std::to_string(l) + ": " + v.name(missing) + " is missing";
            return false;
        }

        // ---- the three projections, batched over the chunk
        if (!proj(gemm, wk, bufs.mixed_f16, bufs.mixed_bf, bufs.qsa_kc, T, v.name("attn_k.weight"), err))
            return false;
        if (!proj(gemm, wv, bufs.mixed_f16, bufs.mixed_bf, bufs.qsa_vc, T, v.name("attn_v.weight"), err))
            return false;
        if (!proj(gemm, wq, bufs.mixed_f16, bufs.mixed_bf, bufs.qsa_qf, T, v.name("attn_q.weight"), err))
            return false;

        // ---- K: norm and rotate, then the whole chunk into the pool in one launch.  V goes in as projected -
        // the same split `qsa_layer` makes (only K is rotated).
        strata::prefill::rms_rows(bufs.qsa_kc, (const float*) wkn->data, T * NKV, HD, HD, EPS, cs);
        strata::prefill::rope(bufs.qsa_kc, T, NKV, HD, NKV * HD, pos0, (float) strata::kernels::qsa_freq_base(), cs);
        strata::prefill::kv_append(bufs.qsa_kc, bufs.qsa_vc, T, pos0, st.page_table, s.page_size, st.k_pool,
                                   st.v_pool, nullptr, nullptr, nullptr, nullptr, cs, &st.host, nullptr);

        // ---- the query: the FIRST half of each head, normed and rotated.  The second half is the gate and is
        // read from `qsa_qf` by `gate_attn_heads` after the attention.
        strata::prefill::split_q_heads(bufs.qsa_qf, bufs.qsa_q, T * NH * HD, NH, HD, cs);
        strata::prefill::rms_rows(bufs.qsa_q, (const float*) wqn->data, T * NH, HD, HD, EPS, cs);
        strata::prefill::rope(bufs.qsa_q, T, NH, HD, NH * HD, pos0, (float) strata::kernels::qsa_freq_base(), cs);

        // ---- the per-position step records: `n_kv` and the width the selection returns
        h_steps.resize((size_t) T * strata::kernels::kStepCount);
        for (int64_t t = 0; t < T; ++t)
            strata::kernels::qsa_step_fill(h_steps.data() + t * strata::kernels::kStepCount, pos0 + t, s);
        if (cudaMemcpyAsync(bufs.qsa_steps, h_steps.data(), (size_t) T * strata::kernels::kStepCount * sizeof(int32_t),
                            cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            err = "qwen35moe prefill: the QSA step records did not upload";
            return false;
        }

        // ---- the identity selection and the attention, in tiles.  A tile's capacity follows its OWN last
        // position (`width = pos + 1` never exceeds it), so the partials `qsa_decode_attn_batch` writes are
        // sized by that tile and not by the session's 32K-cell ceiling.
        const strata::kernels::QsaAttnPools pools = core::qsa_attn_pools(st);
        const int64_t cap_alloc = qsa_cap_for(*ss);
        for (int64_t t0 = 0; t0 < T; t0 += kQsaAttnBatch) {
            const int64_t nb = std::min<int64_t>(kQsaAttnBatch, T - t0);
            const int64_t cap = std::min<int64_t>(pos0 + t0 + nb, cap_alloc);
            strata::prefill::iota_rows(bufs.qsa_ids, nb, cap, cs);
            strata::kernels::qsa_decode_attn_batch(bufs.qsa_q + t0 * NH * HD, pools, bufs.qsa_ids,
                                                   bufs.qsa_steps + t0 * strata::kernels::kStepCount, cap, s,
                                                   bufs.qsa_attn_scratch, bufs.qsa_attn + t0 * NH * HD, nb, cs);
        }

        // ---- the gate (each head's second half) and the output projection.  FP16 activations, as
        // `prefill.cpp`'s QSA half uses: the batched path's arithmetic, not the token path's Q8_K image.
        strata::prefill::gate_attn_heads(bufs.qsa_attn, bufs.qsa_qf, bufs.qsa_attn_h, T * NH * HD, NH, HD, cs);
        if (!proj(gemm, wo, bufs.qsa_attn_h, nullptr, bufs.bo, T, v.name("attn_output.weight"), err)) return false;
        return true;
    }

    // Like ggml's mul_mat_id, keep the weights in their memory tier and group activation rows by expert.
    // Only activations/results cross PCIe for a CPU expert, not its multi-megabyte weight blob.
    bool routed_hybrid(int64_t l, int64_t T, const strata::kernels::cpu::NativeFmt& f,
                       const strata::kernels::NativeExpertLayout& el, std::string& err) {
        using namespace strata::kernels::cpu;
        const size_t N = (size_t) g->n_embd, K = (size_t) ss->k, n_experts = (size_t) g->n_expert;
        const size_t ent = (size_t) T * K;
        Clock::time_point tp[6];
        int nmark = 0;
        const auto mark = [&](bool sync) {
            if (prof) { if (sync) cudaStreamSynchronize(cs); tp[nmark] = Clock::now(); }
            ++nmark;
        };
        h_mixed.resize((size_t) T * N);
        h_act.resize((size_t) T * kNativeActBytes);
        // `reserve`, NOT a fill: every row is written before it is read - a CPU job fills the entries it owns
        // and the resident experts fill theirs - so zeroing the buffer here was 134 MB of host stores per layer
        // per chunk, 13.9 GB over one Qwopus3.6 35B A3B prompt (`2048 + 2048 + 1215` tokens), for values
        // nothing ever reads.  As a capacity it only ever grows to the widest chunk's `T*K*N`.
        if (!h_cpu.reserve(ent * N)) {
            err = "qwen35moe prefill: the expert staging buffer could not be allocated";
            return false;
        }
        if (cudaMemcpyAsync(h_mixed.data(), bufs.mixed, h_mixed.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
            cudaStreamSynchronize(cs) != cudaSuccess) {
            err = "qwen35moe prefill: CPU activations did not download";
            return false;
        }
        mark(false);
        const bool own_q2 = f.gu_type == 42 && q2_own_kernels();
        if (own_q2) h_act_q2.resize((size_t) T);
        for (int64_t t = 0; t < T; ++t) {
            if (own_q2) act_quant_any(h_mixed.data() + (size_t) t * N, (int) N, h_act_q2[(size_t) t]);
            else native_quant_act(f, h_mixed.data() + (size_t) t * N, h_act.data() + (size_t) t * kNativeActBytes);
        }
        mark(false);

        cpu_jobs.clear();
        cpu_blobs.resize(n_experts);
        h_ptr.clear();
        cpu_dst.clear();
        size_t cpu_n = 0;   // the CPU entries seen so far: the row of `cpu_parts` the next one is written to
        std::vector<int32_t> starts{0}, dst, tok;
        for (int32_t e : h_ord) {
            const int32_t slot = cache && host_res ? host_res[(size_t) l * n_experts + (size_t) e] : -1;
            if (slot >= 0 && slot < cache->slots()) {
                h_ptr.push_back((unsigned long long) (uintptr_t) cache->device_slot(slot));
                for (int p = h_start[(size_t) e]; p < h_start[(size_t) e + 1]; ++p) {
                    dst.push_back(h_dst[(size_t) p]);
                    tok.push_back(h_tok[(size_t) p]);
                }
                starts.push_back((int32_t) dst.size());
                continue;
            }
            ExpertJobMulti job;
            const uint8_t* raw[3] = {};
            uint64_t sizes[3] = {};
            if (src->raw_parts(l, e, raw, sizes)) {
                if (sizes[0] != f.up_off || sizes[1] != f.down_off - f.up_off ||
                    sizes[2] != f.bytes - f.down_off) {
                    err = "qwen35moe prefill: CPU expert slice size mismatch";
                    return false;
                }
                job.gate = raw[0]; job.up = raw[1]; job.down = raw[2];
            } else if (src->stable(l, e)) {
                job.blob = src->blob(l, e);
            } else {
                auto& copy = cpu_blobs[(size_t) e];
                copy.resize(f.bytes);
                if (src->copy_blob(l, e, copy.data())) job.blob = copy.data();
            }
            if (!job.blob && (!job.gate || !job.up || !job.down)) {
                err = "qwen35moe prefill: CPU expert could not be read";
                return false;
            }
            for (int p = h_start[(size_t) e]; p < h_start[(size_t) e + 1]; ++p) {
                if (own_q2) job.act[job.nt] = &h_act_q2[(size_t) h_tok[(size_t) p]];
                job.nact[job.nt] = h_act.data() + (size_t) h_tok[(size_t) p] * kNativeActBytes;
                // The compact row, not `parts`'s: the entry's own row of `parts` is where the merge puts it
                // back, after the grouped launch has run, so nothing here is written by two producers.
                job.out[job.nt++] = h_cpu.data() + cpu_n * N;
                cpu_dst.push_back(h_dst[(size_t) p]);
                ++cpu_n;
                if (job.nt == MAXT) {
                    cpu_jobs.push_back(job);
                    job.nt = 0;
                }
            }
            if (job.nt) cpu_jobs.push_back(job);
        }
        mark(false);
        rb_cpu_jobs += (long long) cpu_jobs.size();
        rb_res_experts += (long long) h_ptr.size();
        for (const auto& j : cpu_jobs) rb_cpu_tokens += j.nt;
        pool->run_split_multi_native(f, cpu_jobs.data(), (int) cpu_jobs.size());
        mark(true);
        // The plan is ready and has two independent consumers: the CPU results go up the bus, the resident
        // experts run on the device.  They write DISJOINT rows of `parts` (a CPU entry's row is written by the
        // merge below, a resident expert's by the grouped launch), so nothing orders them against each other and
        // running them in sequence only wasted the launch's time behind a DMA.  The upload therefore gets its own
        // stream and the two are left to overlap; the merge then waits for both and puts the CPU rows back.
        //
        // THE LAUNCH IS ENQUEUED FIRST, ON PURPOSE.  When the staging buffer is pinned the results' copy is a
        // real asynchronous DMA and the order would not matter.  When it is pageable - `cudaHostAlloc` refused,
        // see `PinnedFloats` - the driver PACKS the staging buffer in the calling thread before it queues the
        // transfer, so the copy blocks the host for its whole duration: enqueuing it first would hold the launch
        // back by the entire upload and there would be no overlap at all, which is the whole point of the split.
        const bool has_cpu = cpu_n > 0;
        if (has_cpu && cudaMemcpyAsync(bufs.cpu_dst, cpu_dst.data(), cpu_n * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            err = "qwen35moe prefill: the CPU expert results did not upload";
            return false;
        }
        const int32_t ng = (int32_t) h_ptr.size();
        if (ng) {
            if (cudaMemcpyAsync(bufs.g_ptr, h_ptr.data(), h_ptr.size() * sizeof(h_ptr[0]),
                                cudaMemcpyHostToDevice, cs) != cudaSuccess ||
                cudaMemcpyAsync(bufs.g_start, starts.data(), starts.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice, cs) != cudaSuccess ||
                cudaMemcpyAsync(bufs.g_dst, dst.data(), dst.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice, cs) != cudaSuccess ||
                cudaMemcpyAsync(bufs.g_tok, tok.data(), tok.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice, cs) != cudaSuccess ||
                cudaMemcpyAsync(bufs.g_count, &ng, sizeof(ng), cudaMemcpyHostToDevice, cs) != cudaSuccess) {
                cudaStreamSynchronize(cs);
                err = "qwen35moe prefill: resident expert plan did not upload";
                return false;
            }
            strata::kernels::native_expert_grouped(el, bufs.g_ptr, bufs.g_start, bufs.g_count,
                                                   bufs.g_dst, bufs.g_tok, ng, (int64_t) dst.size(),
                                                   bufs.xq, bufs.ex_scratch, bufs.parts, cs);
        }
        // The results' own DMA, now that the resident experts are already running beside it.
        if (has_cpu &&
            (cudaMemcpyAsync(bufs.cpu_parts, h_cpu.data(), cpu_n * N * sizeof(float),
                             cudaMemcpyHostToDevice, cs2) != cudaSuccess ||
             cudaEventRecord(ev_up, cs2) != cudaSuccess)) {
            err = "qwen35moe prefill: the CPU expert results did not upload";
            return false;
        }
        // The host waits on the upload's event, which is also what makes the merge's read of it safe.
        if (has_cpu && cudaEventSynchronize(ev_up) != cudaSuccess) {
            err = "qwen35moe prefill: the CPU expert results did not upload";
            return false;
        }
        mark(false);
        if (has_cpu) {
            const int rows = (int) cpu_n;
            const dim3 sm((unsigned) ((N + 255) / 256), (unsigned) std::min<int>(rows, 65535));
            scatter_rows_kernel<<<sm, 256, 0, cs>>>(bufs.parts, bufs.cpu_dst, bufs.cpu_parts, (int) N, rows);
        }
        // The local compact plan and the merge must outlive their asynchronous copies and the launch above.
        if (cudaStreamSynchronize(cs) != cudaSuccess) {
            err = "qwen35moe prefill: resident experts did not complete";
            return false;
        }
        mark(false);
        if (prof) {
            ++rb_n;
            for (int i = 0; i + 1 < 6; ++i)
                rb_ms[i] += std::chrono::duration<double, std::milli>(tp[i + 1] - tp[i]).count();
        }
        return true;
    }

    /// The MoE half of layer `l`: the router, the shared expert, the routed experts - each DISTINCT expert read
    /// once and computed in one grouped launch - and the combine.  The arithmetic is the verify window's
    /// (verify.cpp:879-1016) and `moe_route`/`moe_shared`/`moe_combine_parts`' (layer.cpp:394-495): the logits
    /// are `mixed . ffn_gate_inp^T`, softmaxed over ALL experts, then a stable top-k renormalized with the
    /// 2**-14 clamp (`router_top10`, the 256-expert/general form - the window's packed lane is gated on a
    /// 512-expert/10 geometry this model does not have).  The shared expert's output is ADDED to the routed one
    /// unweighted and comes out of `shared_expert_multi` ALREADY multiplied by its own sigmoid, so nothing here
    /// applies a sigmoid a second time.
    bool moe_mixer(const core::LayerView& v, int64_t l, int64_t T, std::string& err) {
        const core::ModelGeometry& gm = *g;
        const int64_t N = gm.n_embd, FF = gm.n_ff, NE = gm.n_expert, K = ss->k;
        if (K < 1 || K > 15) {
            err = "qwen35moe prefill: the session's top-k (" + std::to_string(K) +
                  ") is outside the range the combine kernel takes";
            return false;
        }

        // The MoE half is the one that dominates a chunk, so its own stages are timed separately (slots 6-9);
        // the four are a breakdown of this half's slot and therefore sum to it.
        Clock::time_point rp[5];
        int rmark = 0;
        const auto rmark_fn = [&](bool sync) {
            if (prof) { if (sync) cudaStreamSynchronize(cs); rp[rmark] = Clock::now(); }
            ++rmark;
        };
        Impl::Phase p_route(this, 6);
        rmark_fn(false);
        const core::WeightRef* wr = v.get("ffn_gate_inp.weight");
        if (!proj(gemm, wr, bufs.mixed_f16, bufs.mixed_bf, bufs.logits, T, v.name("ffn_gate_inp.weight"), err))
            return false;
        strata::kernels::router_top10(bufs.logits, (int) T, (int) NE, (int) K, bufs.ids, bufs.weights, cs);
        rmark_fn(true);

        // ---- the shared expert
        const core::WeightRef* wsg = v.get("ffn_gate_shexp.weight");
        const core::WeightRef* wsu = v.get("ffn_up_shexp.weight");
        const core::WeightRef* wsd = v.get("ffn_down_shexp.weight");
        const core::WeightRef* wgi = v.get("ffn_gate_inp_shexp.weight");
        if (!wsg || !wsu || !wsd || !wgi) {
            err = "qwen35moe prefill: layer " + std::to_string(l) + " is missing a shared-expert tensor";
            return false;
        }
        strata::kernels::NativeSharedWeights nsw;
        nsw.gate_f16 = wsg->kind == core::WeightKind::F16InF32 && wsg->data != nullptr;
        nsw.up_f16 = wsu->kind == core::WeightKind::F16InF32 && wsu->data != nullptr;
        nsw.down_f16 = wsd->kind == core::WeightKind::F16InF32 && wsd->data != nullptr;
        nsw.gate_type = wsg->native_type;
        nsw.up_type = wsu->native_type;
        nsw.down_type = wsd->native_type;
        nsw.gate_data = wsg->native_data;
        nsw.up_data = wsu->native_data;
        nsw.down_data = wsd->native_data;
        nsw.q8_1 = bufs.sh_q8_1;
        if ((!nsw.gate_f16 && (nsw.gate_type < 0 || nsw.gate_data == nullptr)) ||
            (!nsw.up_f16 && (nsw.up_type < 0 || nsw.up_data == nullptr)) ||
            (!nsw.down_f16 && (nsw.down_type < 0 || nsw.down_data == nullptr))) {
            err = "qwen35moe prefill: layer " + std::to_string(l) +
                  ": the shared expert is neither resident FP16 nor a native GGUF tensor";
            return false;
        }
        if (wgi->kind != core::WeightKind::Bf16InF32 || wgi->data == nullptr) {
            err = "qwen35moe prefill: layer " + std::to_string(l) +
                  ": ffn_gate_inp_shexp.weight is not a resident BF16 vector";
            return false;
        }
        {
            static bool once = false;
            if (prof && !once) {
                once = true;
                std::fprintf(stderr, "  shared weights: gate_f16=%d up_f16=%d down_f16=%d types %d/%d/%d\n",
                             (int) nsw.gate_f16, (int) nsw.up_f16, (int) nsw.down_f16, nsw.gate_type,
                             nsw.up_type, nsw.down_type);
            }
        }
        try {
            const Clock::time_point sh_t0 = Clock::now();
            strata::kernels::shared_expert_multi((int) T, bufs.mixed, bufs.mixed_bf, nsw,
                                                 (const uint16_t*) wgi->data, bufs.sh_gate, bufs.sh_up,
                                                 bufs.sh_g, bufs.shared, N, FF, cs);
            if (prof) rr_host_ms += std::chrono::duration<double, std::milli>(Clock::now() - sh_t0).count();
        } catch (const std::exception& e) {
            err = std::string("qwen35moe prefill: the shared expert failed: ") + e.what();
            return false;
        }
        rmark_fn(true);

        // ---- the routed experts.  Their activation image is the q8_1 rows the CPU pool quantizes too, and the
        // group plan needs the ids on the host, which costs one synchronization per layer.
        if (src == nullptr) {
            err = "qwen35moe prefill: no expert source was handed to the module";
            return false;
        }
        const auto& lay = strata::kernels::cpu::expert_layout();
        if (!lay.native) {
            err = "qwen35moe prefill: the experts are not a native (per-layer GGUF format) pack";
            return false;
        }
        if (lay.n_expert != NE || l < 0 || l >= (int64_t) lay.fmt.size()) {
            err = "qwen35moe prefill: the expert pack does not cover layer " + std::to_string(l);
            return false;
        }
        const strata::kernels::cpu::NativeFmt& f = lay.fmt[(size_t) l];
        if (f.n_embd != N || f.n_ff != FF) {
            err = "qwen35moe prefill: layer " + std::to_string(l) + "'s experts are " + std::to_string(f.n_embd) +
                  "x" + std::to_string(f.n_ff) + ", not " + std::to_string(N) + "x" + std::to_string(FF);
            return false;
        }
        if (!grouped_gu_type_ok(f.gu_type) || !grouped_gu_type_ok(f.up_type) || !grouped_d_type_ok(f.d_type)) {
            err = "qwen35moe prefill: layer " + std::to_string(l) + "'s expert types (" +
                  std::to_string(f.gu_type) + "/" + std::to_string(f.up_type) + "/" + std::to_string(f.d_type) +
                  ") have no grouped kernel";
            return false;
        }
        const auto el = strata::kernels::native_expert_layout(f.gu_type, f.up_type, f.d_type, N, FF);
        if ((uint64_t) el.bytes != lay.blob_bytes(l) || (size_t) el.bytes > (size_t) lay.max_blob) {
            err = "qwen35moe prefill: layer " + std::to_string(l) + "'s expert blob size disagrees with the pack";
            return false;
        }

        const size_t ENT = (size_t) T * (size_t) K;
        h_ids.resize(ENT);
        if (cudaMemcpyAsync(h_ids.data(), bufs.ids, ENT * sizeof(int32_t), cudaMemcpyDeviceToHost, cs) !=
                cudaSuccess ||
            cudaStreamSynchronize(cs) != cudaSuccess) {
            err = "qwen35moe prefill: the router's ids did not come back";
            return false;
        }
        rmark_fn(false);
        strata::kernels::quantize_q8_1_rows(bufs.mixed, T, N, bufs.xq, cs);
        rmark_fn(true);
        p_route.stop();
        if (prof) {
            ++rr_n;
            for (int i = 0; i + 1 < 5; ++i)
                rr_ms[i] += std::chrono::duration<double, std::milli>(rp[i + 1] - rp[i]).count();
        }
        Impl::Phase p_plan(this, 7);
        src->begin_layer(l, h_ids.data(), (int64_t) ENT);

        // ---- the group plan: entries grouped by expert, each carrying the row of `parts` it computes and the
        // token whose activation it reads.  `router_top10` already emitted the ids in the router's own order, so
        // the row of `parts` for an entry is the entry's own index and the combine needs no remap at all.
        h_count.assign((size_t) NE, 0);
        for (size_t i = 0; i < ENT; ++i) {
            const int32_t e = h_ids[i];
            if (e < 0 || e >= NE) {
                err = "qwen35moe prefill: the router returned expert " + std::to_string(e);
                return false;
            }
            ++h_count[(size_t) e];
        }
        h_start.assign((size_t) NE + 1, 0);
        for (size_t e = 0; e < (size_t) NE; ++e) h_start[e + 1] = h_start[e] + h_count[e];
        for (size_t e = 0; e < (size_t) NE; ++e) h_count[e] = h_start[e];   // now the fill cursors
        h_ord.clear();
        for (size_t e = 0; e < (size_t) NE; ++e) {
            if (h_start[e + 1] > h_start[e]) h_ord.push_back((int32_t) e);
        }
        h_dst.assign(ENT, 0);
        h_tok.assign(ENT, 0);
        for (size_t i = 0; i < ENT; ++i) {
            const int32_t p = h_count[(size_t) h_ids[i]]++;
            h_dst[(size_t) p] = (int32_t) i;
            h_tok[(size_t) p] = (int32_t) (i / (size_t) K);
        }

        if (pool) {
            p_plan.stop();
            Impl::Phase p_stage(this, 8);
            if (!routed_hybrid(l, T, f, el, err)) return false;
            p_stage.stop();
            Impl::Phase p_combm(this, 9);
            strata::kernels::native_moe_combine_multi(bufs.parts, bufs.weights, bufs.shared, bufs.bo, N, K,
                                                      (int) T, cs);
            return true;
        }

        // Compact starts by *group*, not expert id: short chunks can leave gaps in h_ord.
        std::vector<int32_t> group_start;
        for (int32_t e : h_ord) group_start.push_back(h_start[(size_t) e]);
        group_start.push_back((int32_t) ENT);
        h_start.swap(group_start);
        const size_t nd = h_ord.size();
        const size_t nb = (nd + (size_t) kExpertRingSlots - 1) / (size_t) kExpertRingSlots;
        h_ptr.assign(nb * (size_t) kExpertRingSlots, 0);
        h_count_batch.assign(nb, 0);
        if (cudaMemcpyAsync(bufs.g_dst, h_dst.data(), ENT * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
                cudaSuccess ||
            cudaMemcpyAsync(bufs.g_tok, h_tok.data(), ENT * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
                cudaSuccess ||
            cudaMemcpyAsync(bufs.g_start, h_start.data(), (nd + 1) * sizeof(int32_t), cudaMemcpyHostToDevice,
                            cs) != cudaSuccess) {
            err = "qwen35moe prefill: the expert plan did not upload";
            return false;
        }

        // ---- the routed experts, one grouped launch per `kExpertRingSlots` distinct experts.  A blob the source
        // can hand the GPU over PCIe is read where it lies; the rest are staged into the ring, whose slots are
        // safe to reuse because every copy and launch below is on `cs` and therefore ordered against each other.
        p_plan.stop();
        Impl::Phase p_stage(this, 8);
        int64_t slot = 0;
        for (size_t b = 0; b < nb; ++b) {
            const size_t g0 = b * (size_t) kExpertRingSlots;
            const size_t gn = std::min((size_t) kExpertRingSlots, nd - g0);
            h_count_batch[b] = (int32_t) gn;
            for (size_t j = 0; j < gn; ++j) {
                const int64_t e = h_ord[g0 + j];
                if (const uint8_t* alias = src->device_alias(l, e)) {
                    h_ptr[g0 + j] = (unsigned long long) (uintptr_t) alias;
                    continue;
                }
                uint8_t* dst = bufs.ring + (size_t) (slot % kExpertRingSlots) * (size_t) lay.max_blob;
                ++slot;
                const uint8_t* hb = src->blob(l, e);
                if (hb == nullptr || !src->stable(l, e)) {
                    // blocking on purpose: `h_stage` is the next expert's too, so the copy must be past it
                    if (h_stage.size() < (size_t) el.bytes) h_stage.resize((size_t) el.bytes);
                    if (!src->copy_blob(l, e, h_stage.data()) ||
                        cudaMemcpy(dst, h_stage.data(), (size_t) el.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = "qwen35moe prefill: layer " + std::to_string(l) + " expert " + std::to_string(e) +
                              " could not be read";
                        return false;
                    }
                } else if (cudaMemcpyAsync(dst, hb, (size_t) el.bytes, cudaMemcpyHostToDevice, cs) !=
                           cudaSuccess) {
                    err = "qwen35moe prefill: an expert stage copy failed";
                    return false;
                }
                h_ptr[g0 + j] = (unsigned long long) (uintptr_t) dst;
            }
            if (cudaMemcpyAsync(bufs.g_ptr + g0, h_ptr.data() + g0, gn * sizeof(unsigned long long),
                                cudaMemcpyHostToDevice, cs) != cudaSuccess ||
                cudaMemcpyAsync(bufs.g_count + b, &h_count_batch[b], sizeof(int32_t), cudaMemcpyHostToDevice,
                                cs) != cudaSuccess) {
                err = "qwen35moe prefill: the expert addresses did not upload";
                return false;
            }
            strata::kernels::native_expert_grouped(el, bufs.g_ptr + g0, bufs.g_start + g0, bufs.g_count + b,
                                                   bufs.g_dst, bufs.g_tok, (int64_t) gn,
                                                   (int64_t) ENT, bufs.xq,
                                                   bufs.ex_scratch, bufs.parts, cs);
        }
        p_stage.stop();
        Impl::Phase p_combm(this, 9);

        try {
            strata::kernels::native_moe_combine_multi(bufs.parts, bufs.weights, bufs.shared, bufs.bo, N, K,
                                                      (int) T, cs);
        } catch (const std::exception& e) {
            err = std::string("qwen35moe prefill: the combine failed: ") + e.what();
            return false;
        }
        return true;
    }
};

// ================================================================== Qwen35MoePrefill
Qwen35MoePrefill::Qwen35MoePrefill() = default;
Qwen35MoePrefill::~Qwen35MoePrefill() = default;

int64_t Qwen35MoePrefill::chunk() const { return impl_ ? impl_->chunk : 0; }

uint64_t Qwen35MoePrefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    if (chunk < 1) return 0;
    chunk = std::min(chunk, kMaxPrefillChunk);
    Alloc o;
    o.count_only = true;
    Buffers b;
    carve_chunk(chunk, g, ss.k, qsa_cap_for(ss), o, b);
    return (uint64_t) o.used;
}

bool Qwen35MoePrefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                            core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res,
                            int64_t chunk, void* stream, std::string& err, strata::kernels::cpu::ExpertPool* pool) {
    impl_.reset();
    if (g.arch != core::ModelArch::Qwen35Moe) {
        err = "qwen35moe prefill: the model's geometry is not Qwen35Moe";
        return false;
    }
    if (g.hc != 0) {
        err = "qwen35moe prefill: the geometry carries hyper-connection streams (hc != 0); this path is the "
              "plain-residual architecture";
        return false;
    }
    if (g.n_embd > strata::kernels::cpu::H || g.n_ff > strata::kernels::cpu::FF) {
        err = "qwen35moe prefill: expert geometry exceeds the CPU pool's scratch";
        return false;
    }
    if (chunk < 1) { err = "qwen35moe prefill: chunk must be positive"; return false; }
    if (ss.k < 1) { err = "qwen35moe prefill: the session's top-k is unset"; return false; }
    const int64_t requested_chunk = chunk;
    chunk = std::min(chunk, kMaxPrefillChunk);

    auto m = std::make_unique<Impl>();
    m->wt = &wt;
    m->g = &g;
    m->ss = &ss;
    m->src = src;
    m->cache = cache;
    m->host_res = host_res;
    m->chunk = chunk;
    m->cs = (cudaStream_t) stream;
    if (cudaStreamCreateWithFlags(&m->cs2, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&m->ev_up, cudaEventDisableTiming) != cudaSuccess) {
        err = "qwen35moe prefill: the CPU experts' upload stream could not be created";
        return false;
    }
    const char* cpu_prefill = std::getenv("MNTQ_PREFILL_CPU");
    m->pool = cpu_prefill && std::atoi(cpu_prefill) == 0 ? nullptr : pool;
    m->prof = std::getenv("MNTQ_PREFILL_PROF") != nullptr;
    m->gdn_floats = gdn_state_floats(g);
    m->gdn_idx.assign((size_t) g.n_layers, -1);
    m->qsa_idx.assign((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (core::is_qsa_layer(g, l)) m->qsa_idx[(size_t) l] = qi++;
            else m->gdn_idx[(size_t) l] = gi++;
        }
    }

    // llama.cpp grows n_batch to fit its KV/workspace budget; do the same here.  Under WDDM an oversubscribed
    // cudaMalloc succeeds and then pages to system memory, so check the free VRAM as well as the allocation,
    // and halve the chunk until both the arena and the carve fit beside the headroom for later allocations.
    uint64_t bytes = 0;
    for (;;) {
        bytes = bytes_needed(g, ss, chunk);
        size_t free_b = 0, total_b = 0;
        const bool roomy = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess &&
                           (uint64_t) free_b >= bytes + kArenaHeadroomBytes;
        Alloc o;
        if (roomy && cudaMalloc(&m->arena, bytes) == cudaSuccess) {
            m->arena_bytes = (size_t) bytes;
            o.cur = m->arena;
            o.left = (size_t) bytes;
            carve_chunk(chunk, g, ss.k, qsa_cap_for(ss), o, m->bufs);
            if (o.ok) break;
            cudaFree(m->arena);
            m->arena = nullptr;
            m->arena_bytes = 0;
        } else {
            cudaGetLastError();
        }
        if (chunk <= kMinPrefillChunk) {
            err = "qwen35moe prefill: the chunk arena (" + std::to_string(bytes >> 20) +
                  " MiB) does not fit the free VRAM";
            return false;
        }
        chunk = std::max(kMinPrefillChunk, chunk / 2);
    }
    m->chunk = chunk;
    if (chunk != requested_chunk)
        std::fprintf(stderr, "mntq generate: qwen35moe prompt path: chunk %lld -> %lld tokens so its %llu MiB "
                             "arena fits\n", (long long) requested_chunk, (long long) chunk,
                     (unsigned long long) (bytes >> 20));
    if (!m->gemm.init_external(stream, m->bufs.gemm_scratch, GEMM_SCRATCH_ELEMS, m->bufs.gemm_ws, GEMM_WS_BYTES,
                               err))
        return false;
    impl_ = std::move(m);
    return true;
}

bool Qwen35MoePrefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    if (!impl_) { err = "qwen35moe prefill: not initialized"; return false; }
    Impl& m = *impl_;
    const core::ModelGeometry& g = *m.g;
    const core::WeightTable& wt = *m.wt;
    const int64_t N = g.n_embd;
    const auto t0 = Clock::now();
    std::fill(std::begin(m.t_ph), std::end(m.t_ph), 0.0);
    std::fill(std::begin(m.n_ph), std::end(m.n_ph), int64_t{0});
    std::vector<int32_t> host_tok;
    int64_t done = 0, chunks = 0;
    try {
        for (int64_t c0 = 0; c0 < n; c0 += m.chunk) {
            if (should_stop && should_stop()) { err = "cancelled"; return false; }
            const int64_t T = std::min(m.chunk, n - c0);

            host_tok.resize((size_t) T);
            for (int64_t i = 0; i < T; ++i) host_tok[(size_t) i] = (int32_t) tokens[c0 + i];
            if (cudaMemcpyAsync(m.bufs.tokens, host_tok.data(), (size_t) T * 4, cudaMemcpyHostToDevice, m.cs) !=
                cudaSuccess) {
                err = "qwen35moe prefill: the token copy failed";
                return false;
            }
            // ---- the embeddings, broadcast to the architecture's ONE residual stream.  The phase is scoped to
            // the gather ALONE: an `Impl::Phase` left alive across the trunk below would nest over every layer
            // phase and report the whole chunk a second time under "add/embed".
            {
                Impl::Phase pe(&m, 4);
                if (const core::NativeEmbed* ne = core::native_embed()) {
                    ne->gather_dev(m.bufs.tokens, T, m.bufs.emb, m.cs);
                } else {
                    const core::WeightRef* w = wt.find("token_embd.weight");
                    if (w == nullptr || w->codebook_iq4nl ||
                        (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
                        err = "qwen35moe prefill: token_embd.weight is missing or not an S2/S4/S8 tensor";
                        return false;
                    }
                    const auto* codes = (const uint8_t*) w->data;
                    const auto* scales = (const float*) (codes + w->codes_bytes);
                    const auto* offsets =
                        w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
                    const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
                    const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
                    strata::kernels::embedding_gather_dev(codes, scales, offsets, m.bufs.tokens, (int) T, w->ne0,
                                                          w->code_bits, w->code_bias, w->group_elems, row_codes,
                                                          row_groups, m.bufs.emb, m.cs);
                }
                strata::kernels::broadcast_streams(m.bufs.emb, m.bufs.R, N, 1, (int) T, m.cs);
            }

            // ---- the trunk.  Per layer: mixer half -> residual add -> second read -> MoE half -> residual add
            // (the plain path of verify.cpp: `mixed = rms_norm(R, ...)`, `R += bo`, twice).
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const core::LayerView v(wt, l);
                {
                    Impl::Phase p(&m, 0);
                    if (!m.read_norm(v, "attn_norm.weight", T, err)) return false;
                }
                if (core::is_qsa_layer(g, l)) {
                    // The full-attention half runs the decode path's own layer, one position at a time - see
                    // `Impl::qsa_mixer`.  This is what makes the chunk path's attention the reference's.
                    Impl::Phase p(&m, 1);
                    if (!m.qsa_mixer(l, T, pos0 + c0, err)) return false;
                } else {
                    Impl::Phase p(&m, 2);
                    if (!m.gdn_mixer(v, l, T, err)) return false;
                }
                {
                    Impl::Phase p(&m, 4);
                    add_rows(m.bufs.R, m.bufs.bo, T * N, m.cs);
                }
                {
                    Impl::Phase p(&m, 0);
                    if (!m.read_norm(v, "post_attention_norm.weight", T, err)) return false;
                }
                {
                    Impl::Phase p(&m, 3);
                    if (!m.moe_mixer(v, l, T, err)) return false;
                }
                {
                    Impl::Phase p(&m, 4);
                    add_rows(m.bufs.R, m.bufs.bo, T * N, m.cs);
                }
            }

            if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = "qwen35moe prefill: the chunk did not complete";
                return false;
            }
            done += T;
            ++chunks;
            if (on_chunk) {
                Impl::Phase p(&m, 5);
                if (!on_chunk(m.bufs.R, T, pos0 + c0, err)) return false;
            }
        }
    } catch (const std::exception& e) {
        err = std::string("qwen35moe prefill: ") + e.what();
        return false;
    }
    stats_.tokens += done;
    stats_.chunks += chunks;
    stats_.ms_total = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (m.prof) {
        static const char* const nm[kNPhase] = {"read_norm", "qsa_mixer", "gdn_mixer", "moe_mixer",
                                                "add/embed", "on_chunk",  "moe_route", "moe_plan",
                                                "moe_stage", "moe_combm"};
        std::fprintf(stderr, "mntq prefill prof: %.0f ms over %lld tokens in %lld chunks\n", stats_.ms_total,
                     (long long) done, (long long) chunks);
        for (int i = 0; i < kNPhase; ++i)
            std::fprintf(stderr, "  %-10s %9.0f ms  (%lld calls, %.2f ms/call)\n", nm[i], m.t_ph[i],
                         (long long) m.n_ph[i], m.n_ph[i] ? m.t_ph[i] / (double) m.n_ph[i] : 0.0);
        // The last two overlap now: "cpu upload" is the upload's own span and "merge+tail" is everything that
        // was still outstanding when it finished (the resident launch's tail plus the merge), so their SUM - not
        // either one alone - is the time the two branches cost.
        static const char* const sn[5] = {"act quantize", "job build", "cpu pool run", "cpu upload",
                                          "merge+tail"};
        std::fprintf(stderr, "  routed_hybrid over %lld calls: cpu jobs %lld (%lld tokens), resident experts %lld\n",
                     (long long) m.rb_n, m.rb_cpu_jobs, m.rb_cpu_tokens, m.rb_res_experts);
        for (int i = 0; i < 5; ++i)
            std::fprintf(stderr, "  %-16s %9.0f ms\n", sn[i], m.rb_ms[i]);
        static const char* const rn[4] = {"router proj", "shared expert", "ids read-back", "act image"};
        for (int i = 0; i < 4; ++i)
            std::fprintf(stderr, "  %-16s %9.0f ms\n", rn[i], m.rr_ms[i]);
        std::fprintf(stderr, "  %-16s %9.0f ms  (of the shared expert above; the rest is its GPU tail)\n",
                     "shared host", m.rr_host_ms);
    }
    return true;
}

}  // namespace strata::prefill
