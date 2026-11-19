// src/metal/expert_pool.cpp - see expert_pool.hpp.
#include "expert_pool.hpp"
#include "platform.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

namespace mntq::metal {

void* host_alloc(size_t bytes) {
    return plat::map_anon(bytes);
}

void host_free(void* p, size_t bytes) {
    plat::unmap(p, bytes);
}

bool host_pin(void* p, size_t bytes) {
    static const bool off = std::getenv("MNTQ_METAL_NO_PIN") != nullptr;
    if (off || p == nullptr || bytes == 0) return true;
    if (plat::lock(p, bytes)) return true;
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
        std::fprintf(stderr, MNTQ_EXE_NAME ": cannot pin %.1f GB in memory (%s): it stays pageable, so other programs' "
                     "demand can swap it out\n", bytes / 1e9, std::strerror(errno));
    return false;
}

static size_t page_round(size_t n) {
    const size_t pg = plat::page_size();
    return (n + pg - 1) / pg * pg;
}

// ---------------------------------------------------------------- ParallelReader

// One read() is one Batch; a worker holds a reference to the batch it drains, so a straggler from the last read() never
// touches the next one's pieces.
struct Batch {
    std::vector<ReadJob> pieces;
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    std::atomic<int> failed_errno{0};
};

struct ParallelReader::State {
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::shared_ptr<Batch> batch;
    uint64_t generation = 0;
    bool quit = false;

    void drain(Batch& b) {
        for (;;) {
            const size_t i = b.next.fetch_add(1);
            if (i >= b.pieces.size()) return;
            const ReadJob& j = b.pieces[i];
            size_t got = 0;
            while (got < j.bytes) {
                const int64_t r = plat::file_pread(j.fd, (char*) j.dst + got, j.bytes - got, j.offset + got);
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) {
                    b.failed_errno.store(r < 0 ? errno : EIO);
                    break;
                }
                got += (size_t) r;
            }
            if (b.done.fetch_add(1) + 1 == b.pieces.size()) {
                std::lock_guard<std::mutex> lk(mu);
                cv_done.notify_all();
            }
        }
    }
};

ParallelReader::ParallelReader(int threads) : s_(new State) {
    for (int i = 0; i < std::max(threads - 1, 0); ++i) {
        s_->threads.emplace_back([this] {
            uint64_t seen = 0;
            for (;;) {
                std::shared_ptr<Batch> b;
                {
                    std::unique_lock<std::mutex> lk(s_->mu);
                    s_->cv_work.wait(lk, [&] { return s_->quit || s_->generation != seen; });
                    if (s_->quit) return;
                    seen = s_->generation;
                    b = s_->batch;
                }
                s_->drain(*b);
            }
        });
    }
}

ParallelReader::~ParallelReader() {
    {
        std::lock_guard<std::mutex> lk(s_->mu);
        s_->quit = true;
    }
    s_->cv_work.notify_all();
    for (auto& t : s_->threads) t.join();
    delete s_;
}

bool ParallelReader::read(const std::vector<ReadJob>& jobs, std::string& err) {
    constexpr size_t kPiece = 1u << 20;
    auto b = std::make_shared<Batch>();
    for (const ReadJob& j : jobs)
        for (size_t at = 0; at < j.bytes; at += kPiece)
            b->pieces.push_back({j.fd, j.offset + at, (char*) j.dst + at, std::min(kPiece, j.bytes - at)});
    if (b->pieces.empty()) return true;
    {
        std::lock_guard<std::mutex> lk(s_->mu);
        s_->batch = b;
        ++s_->generation;
    }
    s_->cv_work.notify_all();
    s_->drain(*b);
    {
        std::unique_lock<std::mutex> lk(s_->mu);
        s_->cv_done.wait(lk, [&] { return b->done.load() == b->pieces.size(); });
    }
    if (const int e = b->failed_errno.load()) {
        err = std::string("expert read failed: ") + std::strerror(e);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- ExpertPool

ExpertPool::~ExpertPool() {
    delete reader_;
    if (dev_backend_) ggml_backend_free(dev_backend_);
    if (stage_buf_) ggml_backend_buffer_free(stage_buf_);
    for (auto& g : groups_)
        if (g.ram_buf) ggml_backend_buffer_free(g.ram_buf);
    for (auto* b : bufs_) ggml_backend_buffer_free(b);
    for (auto& m : mem_) host_free(m.first, m.second);
    if (ctx_) ggml_free(ctx_);
}

size_t ExpertPool::slots_total() const {
    size_t n = 0;
    for (const auto& g : groups_) n += (size_t) g.active;
    return n;
}

void ExpertPool::pin_slots(Group& g, int from, int to) {
    if (!host_ptr_) return;   // a device pool lives in its own memory: nothing here to pin
    for (int w = 0; w < 3 && to > from; ++w) {   // whole pages over the slots' bytes
        const size_t pg = plat::page_size();
        const uintptr_t a = (uintptr_t) g.t[w]->data + (size_t) from * g.ebytes[w];
        const uintptr_t b = (uintptr_t) g.t[w]->data + (size_t) to * g.ebytes[w];
        const uintptr_t a0 = a / pg * pg, b1 = (b + pg - 1) / pg * pg;
        host_pin((void*) a0, b1 - a0);
    }
}

// one expert slice: a host pool reads it straight into the tensor, a device pool reads it into staging and copies
ReadJob ExpertPool::job(Group& g, const ExpertMatrix& m, int e, int w, int s) const {
    ReadJob j;
    j.fd = m.fd;
    j.offset = m.offset + (uint64_t) e * m.expert_bytes;
    j.bytes = g.ebytes[w];
    if (host_ptr_) {
        j.dst = (char*) g.t[w]->data + (size_t) s * g.ebytes[w];
    } else {
        j.t = g.t[w];
        j.t_off = (size_t) s * g.ebytes[w];
    }
    return j;
}

bool ExpertPool::stream(std::vector<ReadJob>& jobs, std::string& err) {
    if (host_ptr_) return reader_->read(jobs, err);   // a host pool: the bytes land in the tensors themselves
    auto copy_in = [&](const std::vector<ReadJob>& v) {
        if (dev_backend_) {
            for (const ReadJob& j : v)
                ggml_backend_tensor_set_async(dev_backend_, j.t, j.src ? j.src : j.dst, j.t_off, j.bytes);
            ggml_backend_synchronize(dev_backend_);
        } else {
            for (const ReadJob& j : v) ggml_backend_tensor_set(j.t, j.src ? j.src : j.dst, j.t_off, j.bytes);
        }
    };
    // The slices that read into a host section of their own: a cache entry (already holding a hit, filled by the read
    // that follows) - the section is registered with the device, so the copy out of it is one queued command.  The
    // rest have no such section (no host cache) and go through stage_, which one batch at a time has to reuse.
    std::vector<ReadJob> mem, staged;
    for (const ReadJob& j : jobs)
        if (j.src != nullptr || j.dst != nullptr) mem.push_back(j);
        else staged.push_back(j);
    if (!mem.empty()) {
        std::vector<ReadJob> reads;
        for (const ReadJob& j : mem)
            if (j.src == nullptr) reads.push_back(j);
        if (!reads.empty() && !reader_->read(reads, err)) return false;
        copy_in(mem);
    }
    for (size_t i = 0; i < staged.size();) {
        std::vector<ReadJob> batch;   // the slices read in one pass, laid out in stage_
        size_t used = 0;
        while (i < staged.size() && (batch.empty() || used + staged[i].bytes <= stage_size_)) {
            ReadJob j = staged[i++];
            j.dst = (char*) stage_ + used;
            used += j.bytes;
            batch.push_back(j);
        }
        if (!reader_->read(batch, err)) return false;
        copy_in(batch);
    }
    return true;
}

int ExpertPool::ram_lookup(int layer, int expert) {
    Layer& L = layers_[(size_t) layer];
    Group& G = groups_[(size_t) L.group];
    if (G.ram_n == 0) return -1;
    const auto it = G.ram_at.find((int64_t) layer * n_expert_ + expert);
    if (it == G.ram_at.end()) return -1;
    G.ram_use[(size_t) it->second] = ++tick_;
    return it->second;
}

int ExpertPool::ram_reserve(int layer, int expert) {
    Layer& L = layers_[(size_t) layer];
    Group& G = groups_[(size_t) L.group];
    if (G.ram_n == 0) return -1;
    int s = -1;
    if (G.ram_used < G.ram_n) {
        s = G.ram_used++;
    } else {
        uint64_t best = UINT64_MAX;
        for (int i = 0; i < G.ram_n; ++i)
            if (G.ram_use[(size_t) i] < best) {
                best = G.ram_use[(size_t) i];
                s = i;
            }
        if (const int64_t old = G.ram_key[(size_t) s]; old >= 0) G.ram_at.erase(old);
    }
    const int64_t key = (int64_t) layer * n_expert_ + expert;
    G.ram_key[(size_t) s] = key;
    G.ram_use[(size_t) s] = ++tick_;
    G.ram_at[key] = s;
    return s;
}

size_t ExpertPool::bytes_in_use() const {
    size_t n = 0;
    for (const auto& g : groups_) n += (size_t) g.active * (g.ebytes[0] + g.ebytes[1] + g.ebytes[2]);
    return n;
}

size_t ExpertPool::bytes_cap() const {
    size_t n = 0;
    for (const auto& g : groups_) n += (size_t) g.n_slots * (g.ebytes[0] + g.ebytes[1] + g.ebytes[2]);
    return n;
}

size_t ExpertPool::grow(size_t bytes) {
    const size_t room = bytes_cap() - bytes_in_use();
    if (room == 0 || bytes == 0) return 0;
    const double f = std::min(1.0, (double) bytes / (double) room);   // every group the same share of its room
    size_t taken = 0;
    for (auto& g : groups_) {
        const int add = (int) (f * (g.n_slots - g.active));
        pin_slots(g, g.active, g.active + add);
        g.active += add;
        taken += (size_t) add * (g.ebytes[0] + g.ebytes[1] + g.ebytes[2]);
    }
    bytes_ = bytes_in_use();
    return taken;
}

ggml_tensor* ExpertPool::tensor(int layer, int which) const {
    const int g = layers_[(size_t) layer].group;
    return g < 0 ? nullptr : groups_[(size_t) g].t[which];
}

bool ExpertPool::init(const std::vector<std::vector<ExpertMatrix>>& layers, int n_expert, int min_slots,
                      size_t budget, size_t cap, int forced_slots, size_t ram_bytes, ggml_backend_device* dev,
                      int io_threads, std::string& err) {
    cap = std::max(cap, budget);
    n_expert_ = n_expert;
    dev_ = dev;
    host_ptr_ = true;
    if (dev) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        host_ptr_ = props.caps.buffer_from_host_ptr;
    }
    usage_.assign(layers.size() * (size_t) n_expert, 0);
    layers_.assign(layers.size(), Layer{});
    std::vector<int> group_layers;   // layers per group
    for (size_t l = 0; l < layers.size(); ++l) {
        if (layers[l].empty()) continue;
        Layer& L = layers_[l];
        for (int w = 0; w < 3; ++w) L.m[w] = layers[l][(size_t) w];
        L.slot_of.assign((size_t) n_expert, -1);
        int found = -1;
        for (size_t g = 0; g < groups_.size() && found < 0; ++g) {
            bool same = true;
            for (int w = 0; w < 3; ++w)
                same = same && groups_[g].type[w] == L.m[w].type && groups_[g].ne0[w] == L.m[w].ne0 &&
                       groups_[g].ne1[w] == L.m[w].ne1;
            if (same) found = (int) g;
        }
        if (found < 0) {
            Group G;
            for (int w = 0; w < 3; ++w) {
                G.type[w] = L.m[w].type;
                G.ne0[w] = L.m[w].ne0;
                G.ne1[w] = L.m[w].ne1;
                G.ebytes[w] = L.m[w].expert_bytes;
            }
            groups_.push_back(G);
            group_layers.push_back(0);
            found = (int) groups_.size() - 1;
        }
        L.group = found;
        ++group_layers[(size_t) found];
    }

    // slots: the budget's share of every format's experts, never fewer than one ubatch can touch in a layer
    double full = 0.0;
    for (size_t g = 0; g < groups_.size(); ++g)
        full += (double) group_layers[g] * n_expert *
                (double) (groups_[g].ebytes[0] + groups_[g].ebytes[1] + groups_[g].ebytes[2]);
    const double frac = cap == 0 ? 1.0 : std::min(1.0, (double) cap / full);         // the slots mapped
    const double use = budget == 0 ? 1.0 : std::min(1.0, (double) budget / full);    // the slots in use now
    // a pool tensor holds at most kMaxSlots slots: ggml-metal's matrix-matrix expert kernel (a prompt's slices) maps
    // a tensor's experts with one thread each, at most 1024 - so a format's layers are split into groups of
    // consecutive layers that each stay under it (slots are shared, least recently used first, inside a group)
    constexpr int kMaxSlots = 1024;
    std::vector<Group> formats = std::move(groups_);
    groups_.clear();
    all_resident_ = forced_slots <= 0;
    n_pairs_ = 0;
    for (const Layer& L : layers_) n_pairs_ += L.group >= 0 ? n_expert : 0;
    std::vector<std::vector<int>> by_format(formats.size());   // each format's layers, in order (before re-tagging)
    for (size_t l = 0; l < layers_.size(); ++l)
        if (layers_[l].group >= 0) by_format[(size_t) layers_[l].group].push_back((int) l);
    for (size_t f = 0; f < formats.size(); ++f) {
        const std::vector<int>& ls = by_format[f];
        const double per_layer = forced_slots > 0 ? 0.0 : frac * n_expert;   // slots a layer brings
        // as few groups as keep each under kMaxSlots, the layers spread evenly (so no group is below min_slots and
        // pushed over its share of the budget)
        const int n = (int) ls.size();
        const int hi = forced_slots > 0 ? n : std::max(1, (int) (kMaxSlots / std::max(1.0, per_layer)));
        const int k = (n + hi - 1) / hi;
        for (int c = 0, at = 0; c < k; ++c) {
            const int nl = n / k + (c < n % k ? 1 : 0);
            Group G = formats[f];
            const int all = nl * n_expert;
            int n = forced_slots > 0 ? forced_slots : (int) (per_layer * nl);
            n = std::min({all, std::max(n, min_slots), std::max(kMaxSlots, min_slots)});
            G.n_slots = n;
            G.active = forced_slots > 0 ? n : std::min(n, std::max((int) (use * n_expert * nl), min_slots));
            all_resident_ = all_resident_ && G.active == all;
            if (std::getenv("MNTQ_METAL_POOL_DEBUG"))
                std::fprintf(stderr, "pool group: format %zu, %d layers, %d of %d slots, %.2f MB a slot\n", f, nl,
                             G.active, n, (G.ebytes[0] + G.ebytes[1] + G.ebytes[2]) / 1e6);
            for (int i = 0; i < nl; ++i) layers_[(size_t) ls[(size_t) (at + i)]].group = (int) groups_.size();
            groups_.push_back(G);
            at += nl;
        }
    }

    ggml_init_params ip{ggml_tensor_overhead() * 3 * groups_.size() + 1024, nullptr, true};
    ctx_ = ggml_init(ip);
    for (auto& G : groups_) {
        for (int w = 0; w < 3; ++w) {
            G.t[w] = ggml_new_tensor_3d(ctx_, (ggml_type) G.type[w], G.ne0[w], G.ne1[w], G.n_slots);
            const size_t sz = page_round(ggml_nbytes(G.t[w]));
            void* mem = nullptr;
            ggml_backend_buffer_t buf = nullptr;
            if (dev && !host_ptr_) {   // the device gets its own buffer; the bytes are copied in later
                buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), sz);
                mem = buf ? ggml_backend_buffer_get_base(buf) : nullptr;
            } else {
                mem = host_alloc(sz);
                if (!mem) {
                    err = "cannot allocate " + std::to_string(sz >> 20) + " MiB for the expert pool";
                    return false;
                }
                mem_.push_back({mem, sz});
                buf = dev ? ggml_backend_dev_buffer_from_host_ptr(dev, mem, sz, sz)
                          : ggml_backend_cpu_buffer_from_ptr(mem, sz);
                if (buf && dev) host_pin(mem, sz);
            }
            if (!buf) {
                err = "the GPU could not map the expert pool";
                return false;
            }
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            bufs_.push_back(buf);
            if (ggml_backend_tensor_alloc(buf, G.t[w], mem) != GGML_STATUS_SUCCESS) {
                err = "cannot place an expert pool tensor";
                return false;
            }
        }
        G.owner.assign((size_t) G.n_slots, -1);
        G.last_use.assign((size_t) G.n_slots, 0);
        G.stamp.assign((size_t) G.n_slots, 0);
    }
    if (!host_ptr_) {   // a device pool streams the experts through a host section
        size_t max_slice = 0;
        for (const auto& G : groups_)
            for (int w = 0; w < 3; ++w) max_slice = std::max(max_slice, G.ebytes[w]);
        stage_size_ = std::max<size_t>(64u << 20, max_slice);
        // A section the device can see directly (pinned, registered): writing a slice into a device tensor is then a
        // queued buffer copy, and synchronize() flushes the batch's copies in one submission.  Without it every slice
        // would be its own submission and fence wait, which dominates a streaming pool.
        if (ggml_backend_buffer_type* hb = ggml_backend_dev_host_buffer_type(dev)) {
            stage_buf_ = ggml_backend_buft_alloc_buffer(hb, stage_size_);
            if (stage_buf_) {
                stage_ = ggml_backend_buffer_get_base(stage_buf_);
                dev_backend_ = ggml_backend_dev_init(dev, nullptr);
                if (!dev_backend_) {   // no backend to queue on: fall back to plain memory
                    ggml_backend_buffer_free(stage_buf_);
                    stage_buf_ = nullptr;
                    stage_ = nullptr;
                }
            }
        }
        if (!stage_) {   // plain memory: a synchronous staging copy a slice
            stage_ = host_alloc(stage_size_);
            if (!stage_) {
                err = "cannot allocate the expert staging section";
                return false;
            }
            mem_.push_back({stage_, stage_size_});
        }
    }
    // A host cache of the experts the slots evict, so the next request for one is a memory copy, not an SSD read: on a
    // card whose VRAM cannot hold the expert set, the reads are what leaves it idle (measured on an 8.6 GB card: 1.5 s
    // of compute against 2.6 s of reads).  Only a device pool needs one (a host pool IS host memory), and it is taken
    // from the device's host buffer type: the driver then registers it, and a copy out of it is one queued command.
    if (!host_ptr_ && !all_resident_ && ram_bytes > 0) {
        std::vector<size_t> layers_of(groups_.size(), 0);
        std::vector<size_t> block(groups_.size(), 0);
        std::vector<int> want(groups_.size(), 0);
        double full = 0.0;
        for (const Layer& L : layers_)
            if (L.group >= 0) ++layers_of[(size_t) L.group];
        for (size_t g = 0; g < groups_.size(); ++g) {
            block[g] = groups_[g].ebytes[0] + groups_[g].ebytes[1] + groups_[g].ebytes[2];
            full += (double) layers_of[g] * (double) n_expert * (double) block[g];
        }
        bool ok = full > 0.0;
        for (size_t g = 0; g < groups_.size(); ++g) {
            const double share = (double) layers_of[g] * (double) n_expert * (double) block[g];
            const size_t bytes = (size_t) ((double) ram_bytes * share / full);
            const size_t most = (size_t) layers_of[g] * (size_t) n_expert;
            want[g] = (int) std::min(bytes / std::max<size_t>(block[g], 1), most);
            ok = ok && want[g] >= 1;   // every group needs an entry, or a miss would have nowhere to land
        }
        ggml_backend_buffer_type* hb = ok ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        for (size_t g = 0; hb && g < groups_.size(); ++g) {
            Group& G = groups_[g];
            const size_t bytes = (size_t) want[g] * block[g];
            ggml_backend_buffer* buf = ggml_backend_buft_alloc_buffer(hb, bytes);
            // the host buffer type falls back to plain memory when it cannot pin: without a registration the device
            // would copy through its own staging buffer and fence a slice, which is worse than the SSD
            if (!buf || ggml_backend_buffer_get_type(buf) != hb) {
                if (buf) ggml_backend_buffer_free(buf);
                ok = false;
                break;
            }
            G.ram = (char*) ggml_backend_buffer_get_base(buf);
            if (!G.ram) {
                ggml_backend_buffer_free(buf);
                ok = false;
                break;
            }
            G.ram_buf = buf;
            G.ram_block = block[g];
            G.ram_off[0] = 0;
            G.ram_off[1] = G.ebytes[0];
            G.ram_off[2] = G.ebytes[0] + G.ebytes[1];
            G.ram_n = want[g];
            G.ram_key.assign((size_t) G.ram_n, -1);
            G.ram_use.assign((size_t) G.ram_n, 0);
            ram_bytes_ += bytes;
        }
        if (!ok) {
            for (Group& G : groups_) {
                if (G.ram_buf) ggml_backend_buffer_free(G.ram_buf);
                G.ram_buf = nullptr;
                G.ram = nullptr;
                G.ram_n = G.ram_used = 0;
            }
            ram_bytes_ = 0;
            std::fprintf(stderr, MNTQ_EXE_NAME ": no host cache for the experts (the device cannot hold one): every "
                         "miss reads the SSD\n");
        } else {
            ram_on_ = true;
            std::fprintf(stderr, MNTQ_EXE_NAME ": expert host cache %.2f GB: a pool miss is served from memory\n",
                         ram_bytes_ / 1e9);
        }
    }
    reader_ = new ParallelReader(io_threads);
    bytes_ = bytes_in_use();
    for (auto& G : groups_) pin_slots(G, 0, G.active);

    if (all_resident_) {   // every expert has a fixed slot: read them all now
        std::vector<int> next(groups_.size(), 0);
        std::vector<ReadJob> jobs;
        for (size_t l = 0; l < layers_.size(); ++l) {
            Layer& L = layers_[l];
            if (L.group < 0) continue;
            Group& G = groups_[(size_t) L.group];
            L.base = next[(size_t) L.group];
            for (int e = 0; e < n_expert; ++e) {
                const int s = next[(size_t) L.group]++;
                L.slot_of[(size_t) e] = s;
                G.owner[(size_t) s] = (int) l * n_expert + e;
                for (int w = 0; w < 3; ++w) jobs.push_back(job(G, L.m[w], e, w, s));
            }
        }
        if (!stream(jobs, err)) return false;
    }
    return true;
}

void ExpertPool::count(int layer, const int32_t* experts, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (experts[i] >= 0 && experts[i] < n_expert_) ++usage_[(size_t) layer * n_expert_ + (size_t) experts[i]];
}

bool ExpertPool::warm(const std::vector<std::pair<int, int>>& ranked, std::string& err) {
    if (all_resident_) return true;
    std::vector<int> next(groups_.size(), 0);
    std::vector<std::pair<Layer*, int>> picked;   // (layer, expert) per group slot, in rank order
    std::vector<ReadJob> jobs;
    for (const auto& [l, e] : ranked) {
        if (l < 0 || l >= (int) layers_.size() || e < 0 || e >= n_expert_) continue;
        Layer& L = layers_[(size_t) l];
        if (L.group < 0 || L.slot_of[(size_t) e] >= 0) continue;
        Group& G = groups_[(size_t) L.group];
        int& s = next[(size_t) L.group];
        while (s < G.active && G.owner[(size_t) s] >= 0) ++s;
        if (s >= G.active) continue;
        G.owner[(size_t) s] = l * n_expert_ + e;
        L.slot_of[(size_t) e] = s;
        for (int w = 0; w < 3; ++w) jobs.push_back(job(G, L.m[w], e, w, s));
        picked.push_back({&L, s});
        ++s;
    }
    // rank -> recency: the first-ranked gets the newest tick
    tick_ += picked.size();
    for (size_t i = 0; i < picked.size(); ++i)
        groups_[(size_t) picked[i].first->group].last_use[(size_t) picked[i].second] = tick_ - i;
    return stream(jobs, err);
}

bool ExpertPool::ensure(int layer, const int32_t* experts, size_t n, int32_t* slots, std::string& err) {
    Layer& L = layers_[(size_t) layer];
    Group& G = groups_[(size_t) L.group];
    for (size_t i = 0; i < n; ++i)   // checked before any slot changes hands
        if (experts[i] < 0 || experts[i] >= n_expert_) {
            err = "router selected expert " + std::to_string(experts[i]) + " of " + std::to_string(n_expert_);
            return false;
        }
    ++call_;
    std::vector<ReadJob> jobs;
    std::vector<int> filled;   // slots this call reads into: emptied again if the read fails
    std::vector<std::pair<Group*, int>> reserved;   // host cache entries this call filled (dropped if the read fails)
    for (size_t i = 0; i < n; ++i) {
        const int e = experts[i];
        int s = L.slot_of[(size_t) e];
        if (s >= 0) {
            ++hits;
        } else {
            ++misses;
            uint64_t best = UINT64_MAX;
            for (int c = 0; c < G.active; ++c)
                if (G.stamp[(size_t) c] != call_ && G.last_use[(size_t) c] < best) {
                    best = G.last_use[(size_t) c];
                    s = c;
                }
            if (s < 0) {
                err = "the expert pool has fewer slots than one layer needs";
                return false;
            }
            if (const int o = G.owner[(size_t) s]; o >= 0)
                layers_[(size_t) (o / n_expert_)].slot_of[(size_t) (o % n_expert_)] = -1;
            G.owner[(size_t) s] = layer * n_expert_ + e;
            L.slot_of[(size_t) e] = s;
            filled.push_back(s);
            // the host cache: what it holds is copied straight into the slot, what it does not is read into an entry
            // (and copied from there), so only a cache miss touches the SSD
            const int cached = ram_lookup(layer, e);
            if (cached >= 0) {
                ++ram_hits;
                for (int w = 0; w < 3; ++w) {
                    ReadJob j = job(G, L.m[w], e, w, s);
                    j.src = G.ram + (size_t) cached * G.ram_block + G.ram_off[w];
                    jobs.push_back(j);
                }
            } else {
                const int fresh = ram_reserve(layer, e);
                if (fresh >= 0) ++ram_misses;
                for (int w = 0; w < 3; ++w) {
                    ReadJob j = job(G, L.m[w], e, w, s);
                    if (fresh >= 0) j.dst = G.ram + (size_t) fresh * G.ram_block + G.ram_off[w];
                    jobs.push_back(j);
                    bytes_read += G.ebytes[w];
                }
                if (fresh >= 0) reserved.push_back({&G, fresh});
            }
        }
        ++usage_[(size_t) layer * n_expert_ + (size_t) e];
        G.last_use[(size_t) s] = ++tick_;
        G.stamp[(size_t) s] = call_;
        slots[i] = s;
    }
    if (jobs.empty()) return true;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = stream(jobs, err);
    io_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        for (const int s : filled) {
            const int o = G.owner[(size_t) s];
            layers_[(size_t) (o / n_expert_)].slot_of[(size_t) (o % n_expert_)] = -1;
            G.owner[(size_t) s] = -1;
        }
        for (const auto& [g, entry] : reserved)   // what they hold was never read: let the LRU take them again
            if (g->ram_key[(size_t) entry] >= 0) {
                g->ram_at.erase(g->ram_key[(size_t) entry]);
                g->ram_key[(size_t) entry] = -1;
                g->ram_use[(size_t) entry] = 0;
            }
    }
    return ok;
}

}  // namespace mntq::metal
