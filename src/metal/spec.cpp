// src/metal/spec.cpp - see spec.hpp.
#include "spec.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace mntq::metal {

namespace {

int env_int(const char* k, int def) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoi(v) : def;
}

// A round verifies one window, so of the drafters a file allows only one drafts a given round: both propose the
// tokens after the carried one, and a window holds one chain.  The prompt lookup repeats the context and the NextN
// block guesses it: a chain of k tokens costs k + 1 rows and, where the text is written again, keeps all k, while
// the block keeps about three of four rows (29 of 30 drafts over a 48-token run, MTP cap 3).  A chain of four or
// more is then the better drafter (k / (k + 1) > 3 / 4), but only where it is a copy of the text - a short match is
// found by chance as often as not, and a chain of the wrong tokens costs a wider window for nothing (a 48-token run
// on a file with a block lost a quarter of its speed that way).  So the block drafts unless the lookup's match is a
// long one, and the lookup's own worth on this run corrects that: once its chains' rows have landed below the
// block's, the block keeps the rounds and the guess fades back, so a run that starts repeating itself is taken
// again.  With no block in the file the lookup is the only drafter and drafts every chain it has.  Neither drafter
// can change the tokens a round keeps - the window checks them, row by row.
class DraftPick {
public:
    // the lookup proposes k tokens from a match of `match` tokens of the context; has_block: the file carries a NextN
    // block that may draft this round too.  True when the lookup drafts it.
    bool take_lookup(int k, int match, bool has_block) {
        if (k < 4) return false;    // three drafts over four rows never beat the block's rates
        if (!has_block) return true;
        if (look_n_ >= 8 && look_ < block_) {   // this run measured the lookup below the block's worth
            look_ += (1.0 - look_) / 32.0;
            return false;
        }
        return match >= min_copy_;
    }

    void record(bool lookup, int drafted, int accepted) {
        if (drafted <= 0) return;   // a plain step: no drafts to score
        const double y = (double) (1 + accepted) / (double) (1 + drafted);   // tokens a row the window verified
        if (lookup) {
            look_ += (y - look_) / 8.0;
            if (look_n_ < 1024) ++look_n_;
        } else {
            block_ += (y - block_) / 8.0;
        }
    }

private:
    const int min_copy_ = env_int("MNTQ_GLM_SUFFIX_MIN_COPY", 8);   // the matched tokens that make a match a copy
    double look_ = 1.0;    // unproven: a chain of four or more is exact until this run says otherwise
    double block_ = 0.75;  // the block's measured ~3 tokens a row
    int look_n_ = 0;       // the rounds the lookup drafted
};

}  // namespace

DraftLength::DraftLength(int n_cap)
    : cap_(std::max(0, n_cap)),
      start_(std::min(std::max(0, env_int("MNTQ_GLM_MTP_START", 2)), std::max(0, n_cap))),
      fixed_(env_int("MNTQ_GLM_MTP_FIXED", -1)),
      adapt_(env_int("MNTQ_GLM_MTP_ADAPT", 1) != 0),
      hit_((size_t) cap_ + 1, 0.0),
      reach_((size_t) cap_ + 1, 0.0),
      ms_((size_t) cap_ + 1, 0.0),
      ms_n_((size_t) cap_ + 1, 0) {
    best_ = start_;
}

int DraftLength::choose() {
    if (fixed_ >= 0) return std::min(fixed_, cap_);
    if (!adapt_) return cap_;
    if (rounds_ < 6) return start_;   // the first rounds at the start length, measured
    if (probing_ > 0) return probe_n_;
    // expected tokens of a round of n drafts: 1 + sum over depths of the product of their acceptance (prior 0.75)
    auto tokens = [&](int n) {
        double e = 1.0, prod = 1.0, prev = 0.75;
        for (int i = 1; i <= n; ++i) {
            const double a = (hit_[(size_t) i] + 4.0 * prev) / (reach_[(size_t) i] + 4.0);
            prod *= a;
            e += prod;
            prev = a;
        }
        return e;
    };
    // a length's time per round: measured, or the nearest measured one scaled by the window's rows
    auto ms = [&](int n) {
        if (ms_n_[(size_t) n] > 0) return ms_[(size_t) n];
        for (int d = 1; d <= cap_; ++d)
            for (const int k : {n - d, n + d})
                if (k >= 0 && k <= cap_ && ms_n_[(size_t) k] > 0) return ms_[(size_t) k] * (1.0 + n) / (1.0 + k);
        return 1.0 + n;
    };
    int best = 0;
    double best_rate = 0.0;
    for (int n = 0; n <= cap_; ++n) {
        const double r = tokens(n) / ms(n);
        if (r > best_rate) {
            best_rate = r;
            best = n;
        }
    }
    best_ = best;
    if (rounds_ >= probe_at_) {   // now and then two rounds at a neighbour, to keep its timing current
        probe_n_ = best_ + ((rounds_ / probe_gap_) % 2 ? 1 : -1);
        if (probe_n_ >= 0 && probe_n_ <= cap_) probing_ = 2;
        probe_gap_ = std::min(probe_gap_ * 2, 1024);
        probe_at_ = rounds_ + probe_gap_;
        if (probing_ > 0) return probe_n_;
    }
    return best_;
}

void DraftLength::record(int n, int accepted, double ms) {
    ++rounds_;
    if (probing_ > 0) --probing_;
    constexpr double kDecay = 1.0 - 1.0 / 64.0;
    for (int i = 1; i <= cap_; ++i) {
        hit_[(size_t) i] *= kDecay;
        reach_[(size_t) i] *= kDecay;
    }
    for (int i = 1; i <= n; ++i) {   // depth i was reached if every earlier draft was accepted
        if (i > accepted + 1) break;
        reach_[(size_t) i] += 1.0;
        if (i <= accepted) hit_[(size_t) i] += 1.0;
    }
    run_ = n == last_n_ ? run_ + 1 : 1;
    last_n_ = n;
    if (run_ >= 3) {   // a length's time only from its third round in a row (the first ones pay for the switch)
        ms_[(size_t) n] = ms_n_[(size_t) n] == 0 ? ms : 0.8 * ms_[(size_t) n] + 0.2 * ms;
        ++ms_n_[(size_t) n];
    }
}

bool decode_spec(GlmMetal& model, int32_t y, int n_max, const Drafter& draft,
                 const std::function<void(int, int, double)>& done_round, const DecodeHooks& hooks, std::string& err) {
    const size_t V = (size_t) model.geometry().n_vocab;
    std::vector<float> logits;
    std::vector<int32_t> drafts;
    for (;;) {
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t p0 = model.position();   // y goes here
        int64_t room = std::min<int64_t>(n_max, hooks.left() - 1);
        room = std::min<int64_t>(room, model.max_context() - 1 - p0);
        drafts.clear();
        if (room > 0 && !draft(y, p0, (int) room, drafts, err)) return false;
        if (drafts.size() > (size_t) std::max<int64_t>(room, 0)) drafts.resize((size_t) std::max<int64_t>(room, 0));
        if (drafts.empty()) {   // a plain step
            if (!model.forward({y}, logits, err)) return false;
            const int32_t t = hooks.pick(logits.data());
            if (done_round) done_round(0, 0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            if (!hooks.emit(t)) return true;
            y = t;
            continue;
        }
        std::vector<int32_t> win = {y};
        win.insert(win.end(), drafts.begin(), drafts.end());
        if (!model.forward_window(win, logits, err)) return false;
        const int T = (int) win.size();
        int keep = 0;
        bool done = false;
        for (int t = 0; t < T; ++t) {
            const int32_t tok = hooks.pick(logits.data() + V * (size_t) t);
            keep = t + 1;
            if (!hooks.emit(tok)) {
                done = true;
                break;
            }
            y = tok;
            if (t + 1 < T && tok != win[(size_t) t + 1]) break;   // the first draft the model would not write
        }
        if (!model.accept(keep, err)) return false;
        if (done_round)
            done_round(T - 1, keep - 1,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (done) return true;
    }
}

bool decode_mtp(GlmMetal& model, int32_t y, DraftLength& len, SuffixDrafter* lookup, int lookup_cap,
                const std::function<void(int, int, double)>& done, const DecodeHooks& hooks, std::string& err) {
    const bool mixed = lookup != nullptr && lookup_cap > 0;
    std::vector<int32_t> lbuf(mixed ? (size_t) lookup_cap : 0);
    DraftPick pick;
    int chosen = 0;
    bool on_lookup = false;
    const Drafter mtp = [&](int32_t tok, int64_t p0, int n_max, std::vector<int32_t>& drafts, std::string& e) {
        if (mixed) {   // the lookup's chain first, where the context holds a long enough copy to beat the block's rows
            const int k = lookup->propose(std::min(lookup_cap, n_max), lbuf.data());
            if (pick.take_lookup(k, lookup->last_match(), model.has_mtp())) {
                on_lookup = true;
                drafts.assign(lbuf.begin(), lbuf.begin() + k);
                return true;
            }
        }
        on_lookup = false;
        chosen = std::min(len.choose(), n_max);
        if (model.mtp_hidden_pos() != p0 - 1) chosen = 0;   // the NextN block reads the trunk's state of p0 - 1
        for (int i = 0; i < chosen; ++i) {   // chained greedy steps: step i at p0 - 1 + i with the token after it
            int32_t d = 0;
            if (!model.mtp_step(i == 0 ? tok : drafts.back(), p0 - 1 + i, i == 0, d, e)) return false;
            drafts.push_back(d);
        }
        return true;
    };
    return decode_spec(model, y, 15, mtp, [&](int n, int acc, double ms) {
        pick.record(on_lookup, n, acc);
        if (!on_lookup) len.record(chosen, std::min(acc, n), ms);   // a lookup round says nothing of the block's lengths
        if (done) done(n, acc, ms);
    }, hooks, err);
}

}  // namespace mntq::metal
