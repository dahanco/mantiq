// Check recurrent/conv state continuity across prompt tiles and a one-token tail.
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

int main() {
    constexpr int T = 513, S = 128, HK = 2, HV = 4, C = (2 * HK + HV) * S, V = HV * S;
    constexpr int NS = S * HV * S, NC = C * 3;
    float *state, *conv, *qkv, *w, *h, *gate, *beta, *z, *gamma, *y;
    int32_t* keep;
    cudaStream_t stream;
    std::vector<void*> allocations;
    auto alloc = [&](float*& p, size_t n) {
        check(cudaMalloc((void**) &p, n * sizeof(float)));
        allocations.push_back(p);
    };
    try {
        check(cudaStreamCreate(&stream));
        alloc(state, NS); alloc(conv, NC); alloc(qkv, T * C); alloc(w, 4 * C);
        alloc(h, T * C); alloc(gate, T * HV); alloc(beta, T * HV);
        alloc(z, T * V); alloc(gamma, S); alloc(y, T * V);
        check(cudaMalloc((void**) &keep, sizeof(int32_t)));
        allocations.push_back(keep);
        auto init = [&](float* p, int n, float base, float amplitude) {
            std::vector<float> v(n);
            for (int i = 0; i < n; ++i) v[i] = base + amplitude * std::sin(0.19f * i);
            check(cudaMemcpy(p, v.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        };
        init(qkv, T * C, 0, 0.1f); init(w, C * 4, 0.1f, 0.02f);
        init(gate, T * HV, -0.3f, 0.01f); init(beta, T * HV, 0.5f, 0.1f);
        init(z, T * V, 0.1f, 0.2f); init(gamma, S, 1, 0.01f);
        std::vector<float> expected, got(T * V + NS + NC);
        for (int tile : {1, 16, 64, strata::kernels::kVerifyMaxT}) {
            init(state, NS, 0, 0.001f); init(conv, NC, 0, 0.01f);
            for (int t = 0; t < T; t += tile) {
                const int32_t nt = std::min(tile, T - t);
                check(cudaMemcpy(keep, &nt, sizeof(nt), cudaMemcpyHostToDevice));
                strata::kernels::gdn_conv_l2_multi(conv, qkv + t * C, w, h + t * C, C, 2 * HK,
                                                   1e-6f, nt, stream, 0);
                strata::kernels::gdn_conv_commit(conv, qkv + t * C, C, keep, stream);
                strata::kernels::gdn_step_norm_multi(state, h + t * C, C, gate + t * HV, beta + t * HV,
                                                     z + t * V, gamma, 1e-6f, y + t * V, HK, HV,
                                                     nt, keep, true, stream, 0);
                check(cudaStreamSynchronize(stream));
            }
            check(cudaMemcpy(got.data(), y, T * V * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(got.data() + T * V, state, NS * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(got.data() + T * V + NS, conv, NC * sizeof(float), cudaMemcpyDeviceToHost));
            if (expected.empty()) expected = got;
            else if (std::memcmp(got.data(), expected.data(), got.size() * sizeof(float)))
                throw std::runtime_error("tiled output/state differs from single-token GDN");
        }
        for (void* p : allocations) cudaFree(p);
        cudaStreamDestroy(stream);
        std::puts("PASS: 513-token GDN output, recurrent state and conv history are bitwise equal at tiles 1/16/64/256");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        for (void* p : allocations) cudaFree(p);
        return 1;
    }
}
