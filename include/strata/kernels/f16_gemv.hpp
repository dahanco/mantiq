// include/strata/kernels/f16_gemv.hpp - the single-token GEMV for a resident F16 weight.
//
// WHY THIS EXISTS.  `bf16_gemv_fp32_mmvf` covers a BF16 matrix, and `native_mmvq` covers the quantized GGUF
// formats - but neither reads a plain FP16 matrix, and a pack may hold a tensor in exactly that form: the
// index records `kind = 5` (F16), the loader copies its two bytes per element verbatim into the arena, and
// `WeightRef::quantized()` is false.  Two tensors of one real artifact land here (the layer-0 `attn_qkv` and
// `attn_gate`), and before this file the engine refused them twice over: `native_mmvq_supported(1)` is false
// so they never entered the native `skip` set, and every projection dispatch demanded either GGUF blocks or
// an S-form.  The result was a load that reported success and then looped on `verify: ... is not served
// natively`.
//
// The contract is the BF16 MMVF's, with fp16 in place of bf16: one FP32 activation vector against a
// contiguous 2-D FP16 matrix laid out with `ne0` CONTIGUOUS (`w[o * n_in + i]`), an FP32 output.  The decode
// uses `f32_from_f16`, which is EXACT - every fp16 value is representable in f32 - so this path loses nothing
// that the pack did not already lose to fp16, unlike a conversion to bf16 (which would drop three mantissa
// bits) and unlike routing through `Gemm::native` (whose dequant does not know type 1).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// `y[o] = sum_i x[i] * f32(w[o * n_in + i])`, single token.  `x` and `y` are F32; `w` holds F16 bits.
/// n_in must be positive and even, n_out positive, both <= INT_MAX.  x 8-byte aligned, w 4-byte aligned,
/// y 4-byte aligned.  Throws on invalid geometry or pointers.
void f16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream);

/// The same over up to 8 activation rows `x[t * ldx]` -> `y[t * ldy + j]`: one launch, the weight read once,
/// and every output row bit-identical to its own single-row call.
void f16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                              int64_t n_in, int64_t n_out, int n_tok, void* stream);

}  // namespace strata::kernels
