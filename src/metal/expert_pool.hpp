// src/metal/expert_pool.hpp - the routed experts of a GLM GGUF in a resident slot pool, filled from the SSD.
//
// Experts are grouped by format (gate/up/down types and shapes): each group owns three ggml tensors
// [ne0, ne1, n_slots] in memory the GPU can read, and a slot holds one (layer, expert) - the expert's three
// matrices at the same index, so a MoE graph runs mul_mat_id on the pool tensors with slot ids.  Slots are shared
// by every layer of a group and evicted least-recently-used; the slots of the layer being made resident are never
// victims, and the previous layers are done (the graph is cut at every router), so a slot can be rewritten safely.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_device;

namespace mntq::metal {

struct ReadJob {
    int fd = -1;
    uint64_t offset = 0;
    void* dst = nullptr;           // where the bytes land (a host pool: inside a tensor)
    size_t bytes = 0;
    ggml_tensor* t = nullptr;      // a device pool: the tensor the bytes are copied into
    size_t t_off = 0;              // ... at this byte offset
    const void* src = nullptr;     // set: the bytes are copied from here instead of read from the file
};

// A fixed set of threads doing pread; read() splits the jobs into <= 1 MiB pieces and blocks until all are done.
class ParallelReader {
public:
    explicit ParallelReader(int threads);
    ~ParallelReader();
    ParallelReader(const ParallelReader&) = delete;
    ParallelReader& operator=(const ParallelReader&) = delete;
    bool read(const std::vector<ReadJob>& jobs, std::string& err);

private:
    struct State;
    State* s_;
};

struct ExpertMatrix {          // one of gate/up/down of one layer, as stored in the GGUF
    int fd = -1;
    uint64_t offset = 0;       // file offset of expert 0
    size_t expert_bytes = 0;   // nb[2]: one expert's slice
    int type = 0;              // ggml_type
    int64_t ne0 = 0, ne1 = 0;
};

class ExpertPool {
public:
    ExpertPool() = default;
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    // layers: per layer, gate/up/down (empty entries for dense layers).  budget: bytes for the slots in use now;
    // cap: bytes the pool may grow to (its address space is mapped up front, a slot takes memory once written).
    // forced_slots > 0 overrides both (per group).  dev == nullptr: plain host memory (the CPU backend).
    // ram_bytes > 0 and a streaming device pool: a host cache of the experts the pool evicts, so a miss is served
    // from memory instead of the SSD (ram_lookup / ram_reserve).  Sized from the model's expert bytes, 0: none.
    bool init(const std::vector<std::vector<ExpertMatrix>>& layers, int n_expert, int min_slots, size_t budget,
              size_t cap, int forced_slots, size_t ram_bytes, ggml_backend_device* dev, int io_threads,
              std::string& err);

    // take up to `bytes` more of the mapped slots into use (each group by its share); returns the bytes taken
    size_t grow(size_t bytes);
    size_t bytes_in_use() const;
    size_t bytes_cap() const;

    // the pool tensors of a layer's group (gate, up, down)
    ggml_tensor* tensor(int layer, int which) const;

    // fill the free slots with experts in this order (a ranked expert profile: most used first); the earlier an
    // expert's rank, the later LRU evicts it.  Pairs of layers without routed experts are skipped.
    bool warm(const std::vector<std::pair<int, int>>& ranked, std::string& err);

    // make experts[i] resident for `layer`; slots[i] receives its slot.  Returns false on a read error.
    bool ensure(int layer, const int32_t* experts, size_t n, int32_t* slots, std::string& err);

    // how often each (layer, expert) was picked (layer * n_expert + expert), for the learned profile
    const std::vector<uint32_t>& usage() const { return usage_; }
    void count(int layer, const int32_t* experts, size_t n);

    bool all_resident() const { return all_resident_; }
    // every routed expert has a slot in use (all resident, or a streaming pool grown that far: after each expert's
    // first use, nothing misses)
    bool holds_all() const { return all_resident_ || slots_total() >= (size_t) n_pairs_; }
    // all resident: expert e of `layer` sits in slot slot_base(layer) + e of its group
    int slot_base(int layer) const { return layers_[(size_t) layer].base; }
    size_t slots_total() const;
    size_t bytes() const { return bytes_; }
    size_t ram_bytes() const { return ram_bytes_; }   // the host cache's bytes (0: none)
    uint64_t hits = 0, misses = 0, bytes_read = 0;
    uint64_t ram_hits = 0, ram_misses = 0;            // pool misses the host cache served / had to read
    double io_seconds = 0.0;

private:
    struct Group {
        int type[3] = {0, 0, 0};
        int64_t ne0[3] = {0, 0, 0}, ne1[3] = {0, 0, 0};
        size_t ebytes[3] = {0, 0, 0};
        int n_slots = 0;                 // the slots mapped (the tensors' ne[2])
        int active = 0;                  // the slots in use: eviction and placement stay below it
        ggml_tensor* t[3] = {nullptr, nullptr, nullptr};
        std::vector<int> owner;          // slot -> layer * n_expert + expert, -1 free
        std::vector<uint64_t> last_use;  // slot -> tick
        std::vector<uint32_t> stamp;     // slot -> ensure() call that pinned it
        // the host cache of the experts the slots above evict (a streaming device pool, ram_bytes > 0): one entry
        // holds one expert's three matrices, at ram_off[w] inside it.  It comes from the device's host buffer type
        // where there is one, so the device reads an entry directly (see ExpertPool::stream).
        ggml_backend_buffer* ram_buf = nullptr;
        char* ram = nullptr;
        size_t ram_block = 0;            // bytes one cached expert takes
        size_t ram_off[3] = {0, 0, 0};   // where gate/up/down start inside an entry
        int ram_n = 0;                   // entries
        int ram_used = 0;                // entries filled
        std::unordered_map<int64_t, int> ram_at;   // layer * n_expert + expert -> entry, -1: not cached
        std::vector<int64_t> ram_key;              // entry -> key, -1: free
        std::vector<uint64_t> ram_use;             // entry -> tick
    };
    struct Layer {
        int group = -1;
        ExpertMatrix m[3];
        std::vector<int> slot_of;        // expert -> slot, -1 not resident
        int base = 0;                    // all resident: the layer's first slot
    };
    void pin_slots(Group& g, int from, int to);
    ReadJob job(Group& g, const ExpertMatrix& m, int e, int w, int s) const;
    bool stream(std::vector<ReadJob>& jobs, std::string& err);   // read the jobs (through staging on a device)
    int ram_lookup(int layer, int expert);    // the host cache entry holding this expert (recency refreshed), -1: none
    int ram_reserve(int layer, int expert);   // an entry for it: a free one, else the least recently used
    std::vector<Group> groups_;
    std::vector<Layer> layers_;
    ggml_context* ctx_ = nullptr;
    std::vector<ggml_backend_buffer*> bufs_;
    std::vector<std::pair<void*, size_t>> mem_;
    ggml_backend_device* dev_ = nullptr;
    // the device can wrap host memory with the tensor's own address (Metal); a device that cannot (Vulkan) gets real
    // buffers and the bytes are copied in through stage_.  A stage pinned and registered with the device turns the
    // writes into queued buffer copies flushed once a batch; plain memory costs a fence wait a slice.
    bool host_ptr_ = true;
    void* stage_ = nullptr;
    size_t stage_size_ = 0;
    ggml_backend_buffer* stage_buf_ = nullptr;
    bool ram_on_ = false;                // the groups carry a host cache: a miss is served from memory
    size_t ram_bytes_ = 0;               // all of them together
    ggml_backend* dev_backend_ = nullptr;
    ParallelReader* reader_ = nullptr;
    int n_expert_ = 0;
    int n_pairs_ = 0;                    // (layer, expert) pairs with routed experts
    std::vector<uint32_t> usage_;
    uint64_t tick_ = 0;
    uint32_t call_ = 0;
    size_t bytes_ = 0;
    bool all_resident_ = false;
};

void* host_alloc(size_t bytes);           // page-aligned, zeroed; the GPU can wrap it without a copy
void host_free(void* p, size_t bytes);
// keep [p, p + bytes) in memory: unified memory the GPU reads must not be compressed or swapped out by another
// program's demand (a graph would then wait seconds on page-ins).  MNTQ_METAL_NO_PIN=1 skips it.  False: refused
// (the wire limit, vm.user_wire_limit); the memory then stays pageable, said once.
bool host_pin(void* p, size_t bytes);

}  // namespace mntq::metal
