// src/metal/suffix_drafter.hpp - the suffix-lookup drafter (prompt lookup) for mntq-metal / mntq-cpu: no weights, no
// GPU, microseconds a proposal.
//
// It proposes the tokens that followed the longest earlier occurrence of the sequence's current suffix: where the
// output repeats earlier text (a prompt that quotes itself, edits, refactors, repeated code) the continuation is
// usually exact, and one verify window then accepts several tokens for about the price of one pass over the weights.
// It is the drafter for the models that carry none of their own - no NextN block, no DFlash2 - which on a MoE model is
// the common case, and it needs nothing beyond the context already in the cache.
//
// Index: every trigram of the history (prompt + accepted output) maps to its WAYS most recent end positions in a
// fixed-size open-addressing table, so memory is bounded (~20 bytes a history token) and an append is O(1).  A
// proposal checks those candidates, extends each match backwards up to max_match, and takes the longest (the most
// recent on ties).  A match shorter than min_match proposes nothing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mntq::metal {

class SuffixDrafter {
public:
    static constexpr int WAYS = 4;

    explicit SuffixDrafter(int min_match = 3, int max_match = 32, size_t capacity_tokens = 1u << 19);

    void reset();
    // Add tokens to the history (the prompt, then every accepted token).
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }

    // Write up to max_k proposed next tokens to out; returns how many (0: no match of at least min_match).
    int propose(int max_k, int32_t* out);

    // The length of the match behind the last proposal (0 if none).
    int last_match() const { return last_match_; }
    size_t size() const { return hist_.size(); }

private:
    struct Slot {
        uint64_t key = 0;             // trigram hash, never 0 (0 = empty)
        uint32_t pos[WAYS] = {};      // end positions, most recent first
        uint8_t n = 0;
    };
    Slot* find_slot(uint64_t key, bool insert);
    uint64_t key_at(size_t end) const;

    int min_match_, max_match_;
    std::vector<int32_t> hist_;
    std::vector<Slot> table_;
    size_t mask_ = 0;
    int last_match_ = 0;
};

}  // namespace mntq::metal
