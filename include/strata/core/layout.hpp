// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * the tensor manifest agrees on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// Which architecture a `ModelGeometry` describes.  Every dimension below is shared by name, but three of
/// the things built from them are NOT universal - the DSA indexer, the hyper-connection residual, and the
/// per-layer tensor set - so `check_one` and the layer graph have to READ this rather than infer a missing
/// feature from a zero.
enum class ModelArch : int {
    Qwen4Exp = 0,   ///< the implemented architecture: QSA (MLA + DSA indexer) on every 4th layer, gated `hc` residual
    Qwen35Moe = 1,  ///< Qwen3.5-MoE (Qwopus): plain GQA + per-head QK-RMSNorm + partial RoPE, plain residual, no indexer
};

/// The model's geometry, taken from the artifact's own metadata.  Every field here
/// is a number a kernel depends on, so a change is a change to a kernel contract and not a tuning knob.
struct ModelGeometry {
    ModelArch arch = ModelArch::Qwen4Exp;
    int64_t n_embd = 2560;
    int64_t n_layers = 48;
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... 47

    // GDN (36 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;   ///< 2*128*16 + 128*48
    int64_t ssm_value_dim = 6144;        ///< 128 * 48

    // QSA (12 layers)
    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;

    // gated residual, on every layer
    int64_t hc = 4;
    int64_t hc_lr = 320;

    // MoE, on every layer
    int64_t n_expert = 512;
    int64_t n_ff = 640;

    int64_t hc_dim() const { return hc * n_embd; }
    /// How many n_embd-wide streams the residual `R` actually holds: the hyper-connection COUNT (`hc`) for
    /// qwen4exp, and ONE for the plain-residual architecture (`hc == 0`).  This is the storage width every
    /// buffer carved for `R` has to use, and writing `hc * n_embd` directly is wrong at `hc == 0` - it sizes
    /// `R` to zero bytes and leaves the single real stream unallocated.  `hc_dim()` stays the hyper-connection
    /// module's OWN width (`hc_lr`-bottleneck input), a different number, and it is correctly zero here.
    int64_t residual_streams() const { return hc > 0 ? hc : 1; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
};

/// True for the full-attention layers (`layer % 4 == 3`).  It is given twice over -
/// `full_attention_interval = 4` and an explicit `attention.compress_ratios` array - and this is the first of
/// the two.  The predicate is the SAME for qwen35moe (`full_attention_interval = 4` in its metadata too), so
/// the name's "qsa" is the qwen4exp mixer and not the cadence: qwen35moe's full-attention layers carry plain
/// GQA, and only the per-layer TENSOR SET (`check_one`) differs.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

/// The Qwen3.5-MoE / Qwopus geometry, transcribed from the artifact's own GGUF metadata
/// (`qwen35moe.*` keys, not a model card).
///
/// WHAT IS *NOT* HERE IS THE POINT.  `hc`/`hc_lr` are 0 (a plain residual, no hyper-connection streams),
/// `idx_q_heads`/`idx_key_dim` are 0 (no DSA indexer) and there is no PLE.  A zero is not a substitute for
/// the `arch` field: `hc_dim() == 0` and "this architecture has no hyper-connections" are the same NUMBER and
/// different FACTS, and the layer graph has to act on the fact.  That is why `check_one` branches on `arch`
/// rather than testing a dimension for zero.
///
/// `n_layers` is the TRUNK count (40).  The artifact's `block_count` is 41 because it also counts the MTP
/// block at index 40; that is a separate module (`nextn.*`), not a trunk layer, and it is not served here yet.
inline ModelGeometry qwen35moe_geometry() {
    ModelGeometry g;
    g.arch = ModelArch::Qwen35Moe;
    g.n_embd = 2048;              // embedding_length
    g.n_layers = 40;              // block_count (41) minus the MTP block
    g.qsa_interval = 4;           // full_attention_interval: layers 3, 7, ... 39

    // Gated DeltaNet, 30 layers: state_size 128, 16 k-heads, 32 v-heads, conv 4
    g.ssm_state_size = 128;
    g.ssm_k_heads = 16;
    g.ssm_v_heads = 32;
    g.ssm_d_conv = 4;
    g.ssm_conv_channels = 2 * 128 * 16 + 128 * 32;   // = 8192, the `attn_qkv` width
    g.ssm_value_dim = 128 * 32;                      // = 4096, the `attn_gate` / `ssm_out` width

    // full attention, 10 layers: plain GQA, 16 q-heads over 2 kv-heads, head 256, rotary on the first 64
    g.n_head = 16;
    g.n_head_kv = 2;
    g.head_dim = 256;
    g.idx_q_heads = 0;            // no indexer
    g.idx_key_dim = 0;

    // plain residual: no hyper-connections
    g.hc = 0;
    g.hc_lr = 0;

    // MoE on every layer: 256 experts, top-8, intermediate 512
    g.n_expert = 256;
    g.n_ff = 512;
    return g;
}

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace strata::core
