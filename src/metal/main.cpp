// src/metal/main.cpp - mntq-metal: GLM-5.3-Flash on Apple Silicon.
//
//   mntq-metal --gguf MODEL.gguf --serve [--max-context N] ...   the engine `mntq.py` / serve/server.py drive
//   mntq-metal --gguf MODEL.gguf --ids 1,2,3 [--n 64]            greedy continuation of token ids, with timings
#include "glm_metal.hpp"
#include "spec.hpp"
#include "suffix_drafter.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace mntq::metal {
int serve(GlmMetal& model, const std::vector<std::string>& args);
int mtp_draft_cap();
}

namespace {

void usage() {
#ifdef MNTQ_CPU_ONLY
    const char* pool = "  --expert-gb X        memory for the routed-expert pool (default: what the machine's free\n"
                       "                       RAM leaves after the dense weights and the cache)\n"
                       "  --threads N          threads the forward runs on (default: the machine's physical cores)\n"
                       "  --kv-f16             MLA latent cache in F16 (as the GPU and the CUDA engine; half the bytes)\n"
                       "  --kv-f32             MLA latent cache in F32 (the default here: ~1e-6 logit error)\n";
#else
    const char* pool = "  --expert-gb X        unified memory for the routed-expert pool (default: what the GPU's\n"
                       "                       working set leaves after the dense weights and the cache)\n"
                       "  --expert-ram-gb X    system RAM for the experts the pool evicts: a pool miss is then a memory\n"
                       "                       copy, not an SSD read.  Off by default: measured not to help (the io of a\n"
                       "                       streaming pool is bound by the copy into VRAM, not the SSD read)\n"
                       "  --kv-f16             MLA latent cache in F16 (half the bytes; default F32, F16 on the GPU)\n"
                       "  --kv-f32             keep the MLA latent cache F32 on the GPU too (MNTQ_METAL_KV_F32=1)\n";
#endif
    std::fprintf(stderr,
                 "usage: " MNTQ_EXE_NAME " --gguf MODEL.gguf [--serve | --ids 1,2,3 [--n N]]\n"
                 "  --max-context N      tokens the KV cache holds (default 32768)\n"
                 "  --ubatch N           prompt tokens per slice (default 2048)\n"
                 "%s"
                 "  --io-threads N       parallel SSD reads for expert misses (default 16)\n"
                 "  --expert-profile F   ranked expert profile that fills the pool at load (data/expert-profile-glm.bin)\n"
                 "  --expert-profile-save F  this machine's learned profile: read instead when it exists, rewritten\n"
                 "                       after each request\n"
                 "  --no-mtp             no MTP drafting (the NextN block is not loaded)\n"
                 "  --dflash F           draft with this DFlash2 drafter GGUF instead of the NextN block\n"
                 "  --suffix-draft N     draft from an earlier repeat of the last N+ tokens of context (prompt lookup,\n"
                 "                       default 3; 0: off) - the drafter where the file carries none (no NextN block)\n"
#ifndef MNTQ_CPU_ONLY
                 "  --cpu                run on ggml-cpu instead of the GPU (reference runs)\n"
#endif
                 , pool);
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        v.push_back((int32_t) std::strtol(p, &end, 10));
        p = *end == ',' ? end + 1 : end;
        if (end == p && *p) break;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf;
    int64_t max_ctx = 32768;
    bool serve = false;
    std::vector<int32_t> ids;
    int n_gen = 32;
    // the suffix-lookup drafter (prompt lookup): on by default, the drafter for a file with none of its own.  It runs
    // only where neither a DFlash2 drafter nor a NextN block is there to draft instead.
    int suffix_draft = 3;
    if (const char* v = std::getenv("MNTQ_GLM_SUFFIX_DRAFT"); v && *v) suffix_draft = std::max(0, std::atoi(v));
    const int suffix_cap = std::min(16, std::max(0, [] {
        const char* v = std::getenv("MNTQ_GLM_SUFFIX_DRAFT_LEN");
        return v && *v ? std::atoi(v) : 8;
    }()));
    mntq::metal::GlmOptions o;
    o.mtp = std::getenv("MNTQ_GLM_NO_MTP") == nullptr && std::getenv("MNTQ_GLM_NO_SPEC") == nullptr;
    if (const char* v = std::getenv("MNTQ_METAL_DSA_MM"); v && *v) o.dsa_mm = std::atoi(v) != 0;
    if (const char* v = std::getenv("MNTQ_METAL_DSA_KERNEL"); v && *v) o.dsa_kernel = std::atoi(v) != 0;
    if (const char* v = std::getenv("MNTQ_METAL_KV_F32"); v && *v) o.kv_f32 = std::atoi(v) != 0;
    std::vector<std::string> rest;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, MNTQ_EXE_NAME ": %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--gguf" || a == "--native") gguf = next();
        else if (a == "--max-context") max_ctx = std::atoll(next());
        else if (a == "--serve") serve = true;
        else if (a == "--ids") ids = parse_ids(next());
        else if (a == "--n") n_gen = std::atoi(next());
        else if (a == "--ubatch") o.n_ubatch = std::atoi(next());
        else if (a == "--attn-chunk") o.attn_chunk = std::atoi(next());
        else if (a == "--expert-gb") o.expert_budget = (size_t) (std::atof(next()) * 1e9);
        else if (a == "--expert-ram-gb") o.expert_ram_bytes = (size_t) (std::atof(next()) * 1e9);
        else if (a == "--pool-slots") o.pool_slots = std::atoi(next());
        else if (a == "--io-threads") o.io_threads = std::atoi(next());
        else if (a == "--expert-profile") o.expert_profile = next();
        else if (a == "--expert-profile-save") o.expert_profile_save = next();
        else if (a == "--threads") o.cpu_threads = std::atoi(next());
        else if (a == "--kv-f16") o.kv_f16 = true;
        else if (a == "--kv-f32") o.kv_f32 = true;
        else if (a == "--cpu") o.use_gpu = false;
        else if (a == "--no-mtp") o.mtp = false;
        else if (a == "--dflash") o.dflash = next();
        else if (a == "--suffix-draft") {
            // the lookup's window is sized here (o.max_window below) and drafted in serve(): both must read the same
            // value, so the flag goes on to serve() as well - a 0 that only one of them saw left a window too small
            // for the lookup's chains and every request failed with "a verify window takes 1 to 4 tokens"
            const std::string v = next();
            suffix_draft = std::max(0, std::atoi(v.c_str()));
            rest.push_back(a);
            rest.push_back(v);
        }
        else if (a == "--dump") o.dump_dir = next();
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            rest.push_back(a);   // the serve loop takes the server's other flags
        }
    }
    if (gguf.empty()) {
        usage();
        return 2;
    }
    if (!o.dflash.empty()) o.mtp = false;   // as the CUDA engine: a DFlash2 drafter replaces the NextN block
    if (o.mtp) o.max_window = std::max(o.max_window, mntq::metal::mtp_draft_cap() + 1);
    if (!o.dflash.empty()) o.max_window = std::max(o.max_window, 16);
    // a lookup window (a lookup round's drafts + the token they follow) - widened even when the NextN block was asked
    // for, since the lookup drafts the rounds the block does not (and the whole run where the file has no block)
    if (suffix_draft > 0 && o.dflash.empty()) o.max_window = std::max(o.max_window, suffix_cap + 1);
    mntq::metal::GlmMetal model;
    model.set_options(o);
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!model.load(gguf, max_ctx, err)) {
        std::fprintf(stderr, MNTQ_EXE_NAME ": %s\n", err.c_str());
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, MNTQ_EXE_NAME ": loaded in %.1f s: %s\n", load_s, model.describe().c_str());
    if (serve) return mntq::metal::serve(model, rest);

    if (ids.empty()) ids = {1};
    std::vector<float> logits;
    auto t1 = std::chrono::steady_clock::now();
    if (!model.forward(ids, logits, err)) {
        std::fprintf(stderr, MNTQ_EXE_NAME ": %s\n", err.c_str());
        return 1;
    }
    const double pp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const auto sp = model.stats();
    std::printf("prompt %zu tokens in %.2f s (%.1f tok/s); %.2f GB read, io %.2f s, gpu %.2f s, host %.2f s "
                "(%llu graphs)\n",
                ids.size(), pp, ids.size() / pp, sp.bytes_read / 1e9, sp.io_seconds, sp.gpu_seconds,
                sp.host_seconds, (unsigned long long) sp.graphs);
    std::vector<int32_t> out;
    t1 = std::chrono::steady_clock::now();
    const auto s0 = model.stats();
    const size_t V = (size_t) model.geometry().n_vocab;
    auto argmax = [&](const float* l) {
        int32_t best = 0;
        for (size_t v = 1; v < V; ++v)
            if (l[v] > l[(size_t) best]) best = (int32_t) v;
        return best;
    };
    mntq::metal::DecodeHooks hk;   // greedy, n_gen tokens
    hk.pick = argmax;
    // The lookup's history is the prompt followed by every emitted token, so at every round its last token IS the
    // carried `y` (the token the window will feed at p0): the proposal then continues exactly where the text is.
    // The lookup drafts where the file carries no NextN block of its own - and alongside the block where it does:
    // decode_mtp hands it the rounds the context repeats (spec.cpp's DraftPick), so a repeated prompt gets the
    // lookup's chains and new text gets the block's steps.
    std::unique_ptr<mntq::metal::SuffixDrafter> sfx;
    if (suffix_draft > 0 && !model.has_dflash()) {
        sfx = std::make_unique<mntq::metal::SuffixDrafter>(suffix_draft, 64, (size_t) max_ctx + 4096);
        sfx->append(ids.data(), ids.size());
    }
    hk.emit = [&](int32_t t) {
        out.push_back(t);
        if (sfx) sfx->append(t);
        return (int) out.size() < n_gen;
    };
    hk.left = [&] { return (int64_t) (n_gen - (int) out.size()); };
    int drafted = 0, accepted = 0, rounds = 0;
    auto tally = [&](int n, int acc, double) {
        drafted += n;
        accepted += acc;
        rounds += n > 0;
    };
    bool ok = true;
    const int32_t y = argmax(logits.data());
    if (hk.emit(y)) {
        if (model.has_dflash()) {
            const mntq::metal::Drafter df = [&](int32_t tok, int64_t, int n, std::vector<int32_t>& d, std::string& e) {
                return model.dflash_draft(tok, n, 0.95f, 0, 0.0f, d, e);
            };
            ok = mntq::metal::decode_spec(model, y, model.dflash_block() - 1, df, tally, hk, err);
        } else if (model.has_mtp()) {
            mntq::metal::DraftLength len(mntq::metal::mtp_draft_cap());
            ok = mntq::metal::decode_mtp(model, y, len, sfx.get(), suffix_cap, tally, hk, err);
        } else if (sfx) {
            const mntq::metal::Drafter lookup = [&](int32_t, int64_t, int n, std::vector<int32_t>& d, std::string&) {
                std::vector<int32_t> buf((size_t) n);
                const int k = sfx->propose(n, buf.data());
                d.assign(buf.begin(), buf.begin() + k);
                return true;
            };
            ok = mntq::metal::decode_spec(model, y, suffix_cap, lookup, tally, hk, err);
        } else {
            for (int32_t t = y;;) {
                if (!(ok = model.forward({t}, logits, err))) break;
                t = argmax(logits.data());
                if (!hk.emit(t)) break;
            }
        }
    }
    if (!ok) {
        std::fprintf(stderr, MNTQ_EXE_NAME ": %s\n", err.c_str());
        return 1;
    }
    const double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const auto& s = model.stats();
    std::printf("tokens:");
    for (int32_t t : out) std::printf(" %d", t);
    const uint64_t hits = s.expert_hits - s0.expert_hits, misses = s.expert_misses - s0.expert_misses;
    const uint64_t ram = s.ram_hits - s0.ram_hits;
    if (rounds > 0)
        std::printf("\ndrafts: %d rounds, %d drafted, %d accepted (%.2f tokens a pass)", rounds, drafted, accepted,
                    (double) (accepted + rounds) / rounds);
    std::printf("\ndecode %d tokens in %.2f s (%.2f tok/s); experts %.1f%% hit, %.2f GB read, io %.2f s, gpu %.2f s, "
                "host %.2f s (%llu graphs)\n",
                n_gen, tg, (n_gen - 1) / tg, hits + misses ? 100.0 * hits / (hits + misses) : 100.0,
                (s.bytes_read - s0.bytes_read) / 1e9, s.io_seconds - s0.io_seconds, s.gpu_seconds - s0.gpu_seconds,
                s.host_seconds - s0.host_seconds, (unsigned long long) (s.graphs - s0.graphs));
    if (misses > 0 && ram > 0)
        std::printf("expert cache %.1f GB: %.1f%% of the misses served from memory (no SSD read)\n",
                    s.ram_bytes / 1e9, 100.0 * ram / misses);
    return 0;
}
