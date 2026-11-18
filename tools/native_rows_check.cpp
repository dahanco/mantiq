// Regression checks for prompt-sized native expert groups, without a model or GPU.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    using namespace strata::kernels::cpu;
    if (!cpu_avx2_ok()) {
        std::puts("SKIP: AVX2 required");
        return 77;
    }
    ggml_cpu_init();
    const int n = 512, rows = 19;
    const size_t rb = ggml_row_size(GGML_TYPE_IQ4_NL, n);
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_0, n);
    std::vector<unsigned char> weights(rb * rows);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = (unsigned char) (i * 73 + 19);
    const ggml_fp16_t scale = ggml_fp32_to_fp16(0.01f);
    for (size_t i = 0; i < weights.size(); i += ggml_type_size(GGML_TYPE_IQ4_NL))
        std::memcpy(weights.data() + i, &scale, sizeof(scale));
    for (int nt : {1, 2, 4, 5, 8, 9, 17, 64, 255, 256}) {
        std::vector<unsigned char> acts(ab * nt);
        std::vector<float> x(n), actual((size_t) nt * rows, -123.0f), expected(actual);
        std::vector<const void*> ap(nt);
        std::vector<float*> op(nt);
        for (int t = 0; t < nt; ++t) {
            for (int i = 0; i < n; ++i) x[i] = std::sin(0.13f * i + 0.07f * t);
            ap[t] = acts.data() + ab * t;
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), acts.data() + ab * t, n);
            op[t] = actual.data() + (size_t) t * rows;
            float* ref = expected.data() + (size_t) t * rows;
            iq4nl256_down_rows(weights.data(), rb, n, &ap[t], 1, &ref, 3, rows - 2);
        }
        iq4nl256_down_rows(weights.data(), rb, n, ap.data(), nt, op.data(), 3, rows - 2);
        if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float))) {
            std::fprintf(stderr, "FAIL: IQ4_NL batch %d or row-range canaries\n", nt);
            return 1;
        }
    }
    std::puts("PASS: IQ4_NL batches 1..256, partial row ranges, bitwise single-token parity");
    return 0;
}
