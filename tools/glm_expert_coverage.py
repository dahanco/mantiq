"""tools/glm_expert_coverage.py - what a routing record says about how many experts a layer really needs.

    python tools/glm_expert_coverage.py [SOURCE ...] [--glm] [--slots N] [--l2-slots M] [--target P] [--out FILE]

A SOURCE is any of the engine's routing records (the readers are tools/make_profile.py's): a `--dump-routing`
binary trace, a `STRATA_GLM_USAGE` file ("layer e:count ..."), a pack's expert_counts.txt ("layer c0 c1 ...") or
a `STRATA_GLM_TRACE` text ("layer expert").  The tool reports, per layer, the share of the routed mass the
hottest N experts carry (the coverage curve), the N that reaches a coverage target, and the routing's effective
width (exp of the Shannon entropy - how many experts a token's routing is really spread over, not how many the
layer has).  It then fills a VRAM slot budget (and an optional RAM-tier budget) with the hottest (layer, expert)
pairs, greedily, and reports the coverage that placement reaches and the number to pin per layer, layer by layer.

This changes no weight and no route: it only says where a fixed slot budget buys the most routed mass, the number
to hand `--expert-profile` (or a pack's expert_prior.txt) so a start is warm.  Route-weighted figures are over
the sources' routes; the mean is by routed mass, so a long session cannot be outvoted by a short one.

    python tools/glm_expert_coverage.py usage.txt trace.bin --glm --slots 114 --l2-slots 222 --target 90

`--capture CONFIG` records the trace first: it starts the engine the config (mntq-<model>.json) describes with
`--dump-routing` and `STRATA_GLM_USAGE` on a fresh copy, answers a short built-in corpus (calibrate_glm.py's -
short prompts, longer answers, greedy, so the profile is decode-heavy), then analyses the trace it wrote.  The
routing is the model's own; the tiers the config placed only decide what is read, not what is routed.

    python tools/glm_expert_coverage.py --capture mntq-mntq-s24.json --glm --slots 114 --target 90
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_profile import is_text, read_text, read_trace   # noqa: E402  (the engine records' readers, reused)

GLM_LAYER, GLM_EXPERT = 46, 288      # GLM-5.3-Flash: 45 decoder layers + the NextN block; 288 routed experts
N_LAYER, N_EXPERT = 48, 512          # the default geometry (as make_profile.py)
CURVE = (1, 2, 4, 8, 16, 32, 64, 128)   # the top-N cut points the coverage curve is printed at
TARGETS = (50, 80, 90, 95, 99)          # coverage targets a per-layer N is reported for


def load(sources, n_layer, n_expert):
    """(layer, expert) -> routed count, summed over every source (any of the engine records)."""
    freq = defaultdict(float)
    for s in sources:
        f = read_text(s, n_expert, n_layer) if is_text(s) else read_trace(s, n_expert, n_layer)
        for p, c in f.items():
            freq[p] += c
    return freq


def layer_counts(freq, n_layer, n_expert):
    """Per layer, the experts' counts, descending (expert id is not needed past the ranking)."""
    return [sorted((freq.get((il, e), 0.0) for e in range(n_expert)), reverse=True) for il in range(n_layer)]


def coverage(counts, n=None):
    """The share of the routed mass the hottest `n` experts carry (all of them: 1.0).  `counts` is descending."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    return sum(counts[:n]) / total


def n_for_target(counts, pct):
    """The fewest hottest experts whose coverage reaches `pct` percent (0: the layer routed nothing)."""
    total = sum(counts)
    if total <= 0:
        return 0
    want = pct / 100.0 * total
    acc = 0.0
    for i, c in enumerate(counts, 1):
        acc += c
        if acc >= want:
            return i
    return sum(1 for c in counts if c > 0)


def effective_n(counts):
    """exp of the Shannon entropy of the routing: the effective number of experts a token is spread over.
    Near 1: one expert takes the layer.  Near the experts used: the routing is flat."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return math.exp(h)


def allocate(freq, slots):
    """The `slots` hottest (layer, expert) pairs, greedy by count (each pick the most-routed pair left).  A
    global ranking, so a hot pair in a narrow layer is not outbid by a slot a wide layer does not need."""
    pairs = sorted(((c, l, e) for (l, e), c in freq.items() if c > 0), key=lambda t: (-t[0], t[1], t[2]))
    return pairs[:slots]


def report(freq, n_layer, n_expert, slots, l2_slots, target, out=print):
    counts = layer_counts(freq, n_layer, n_expert)
    live = [il for il in range(n_layer) if sum(counts[il]) > 0]
    total = sum(sum(counts[il]) for il in live)
    if not live or total <= 0:
        out("no routes in the sources (an empty or unreadable record?)")
        return

    out(f"{len(live)}/{n_layer} layers routed, {total:.0f} routes, {n_expert} experts a layer")
    out("")
    head = "layer  routes   used  eff-N | " + " ".join(f"top{n:<5}" for n in CURVE)
    out(head)
    out("-" * len(head))
    for il in live:
        c = counts[il]
        used = sum(1 for x in c if x > 0)
        cov = " ".join(f"{100 * coverage(c, n):5.1f}%" for n in CURVE)
        out(f"{il:5d} {sum(c):8.0f} {used:6d} {effective_n(c):6.1f} | {cov}")

    # a uniform K experts a layer: the number to pin when the budget is spread evenly (a floor the allocation's
    # per-layer counts are read against - a layer the allocation starves is one a uniform pin would over-serve)
    out("")
    out("uniform hot set, route-weighted:")
    for k in CURVE:
        rw = sum(sum(counts[il][:k]) for il in live) / total
        out(f"  top {k:3d} a layer: {100 * rw:5.1f}% of the routed mass")

    out("")
    out("experts to pin a layer for a coverage target (sum a layer: the whole model's hot set):")
    for p in TARGETS:
        ns = [n_for_target(counts[il], p) for il in live]
        out(f"  {p:2d}%: mean {sum(ns) / len(ns):5.1f} a layer, {sum(ns)} pairs over {len(live)} layers "
            f"(min {min(ns)}, max {max(ns)})")

    # the greedy allocation: fill VRAM, then the RAM tier with what is left, and report the mass each reaches
    out("")
    budget = slots + l2_slots
    chosen = allocate(freq, budget)
    for tier, cap in (("VRAM", slots), ("RAM tier", budget)):
        got = chosen[:cap]
        cov = sum(t[0] for t in got) / total
        per = defaultdict(int)
        for _, l, _e in got:
            per[l] += 1
        counts_l = [per[l] for l in live]
        out(f"  {tier:<9} {cap:5d} slots: {100 * cov:5.1f}% of the routed mass, "
            f"{sum(counts_l) / len(counts_l):4.1f} experts a layer (min {min(counts_l)}, max {max(counts_l)})")
    n_cov = sum(1 for _, _l, _e in chosen if True)
    out("")
    out(f"  {slots} VRAM + {l2_slots} RAM = {budget} pinned pairs of {len(freq)} routed pairs "
        f"({n_cov} chosen)")


def capture(config, n_max, prompts, trace, usage):
    """Start the engine `config` describes with a routing trace and a fresh usage file, answer the built-in
    corpus, and stop it - the record report() then reads.  calibrate_glm.py's config handling (the env's MNTQ_/
    STRATA_ names) and corpus are reused, so the run matches a calibration's."""
    import json
    import threading
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from serve.server import StrataEngine, child_env, engine_args
    import mntq_names
    import calibrate_glm as CAL
    cfg = mntq_names.env_in(json.loads(Path(config).read_text(encoding="utf-8-sig")))
    ids_list = CAL.prompt_ids(cfg)
    if prompts:
        ids_list = ids_list[:prompts]
    args = [*engine_args(cfg), "--dump-routing", trace]
    env = {**child_env(cfg), "STRATA_GLM_USAGE": usage}
    print(f"starting the engine for {len(ids_list)} prompts (the load reads the expert set first; minutes) ...")
    eng = StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=env)
    try:
        for i, ids in enumerate(ids_list, 1):
            n = sum(1 for t in eng.generate(ids, n_max, {"temperature": 0}, threading.Event()) if t is not None)
            print(f"  prompt {i}/{len(ids_list)}: {n} tokens")
    finally:
        CAL.close(eng)
    print(f"recorded the routing trace {trace} and the usage file {usage}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sources", nargs="*", help="routing records: --dump-routing trace, STRATA_GLM_USAGE, "
                                              "expert_counts.txt or STRATA_GLM_TRACE")
    ap.add_argument("--capture", metavar="CONFIG", default=None,
                    help="start the engine the config (mntq-<model>.json) describes and record the routing first")
    ap.add_argument("--max-new", type=int, default=256, help="tokens a --capture answer decodes (default 256)")
    ap.add_argument("--prompts", type=int, default=0, help="answer only the first N corpus prompts (0: all)")
    ap.add_argument("--trace", default=None, help="where --capture writes the trace (default ./glm-routing.bin)")
    ap.add_argument("--usage", default=None, help="where --capture writes the usage file (default ./glm-usage.txt)")
    ap.add_argument("--glm", action="store_true",
                    help=f"GLM-5.3-Flash geometry ({GLM_LAYER} layers of {GLM_EXPERT} experts)")
    ap.add_argument("--n-layer", type=int, default=None, help=f"layers (default {N_LAYER}; --glm {GLM_LAYER})")
    ap.add_argument("--n-expert", type=int, default=None, help=f"experts a layer (default {N_EXPERT}; --glm {GLM_EXPERT})")
    ap.add_argument("--slots", type=int, default=114, help="VRAM expert slots to fill (default 114)")
    ap.add_argument("--l2-slots", type=int, default=0, help="further slots on the RAM tier (default 0)")
    ap.add_argument("--target", type=int, default=90, help="coverage target the headline reads (default 90%%)")
    ap.add_argument("--out", default=None, help="also write the report here")
    a = ap.parse_args()

    nl = a.n_layer or (GLM_LAYER if a.glm else N_LAYER)
    ne = a.n_expert or (GLM_EXPERT if a.glm else N_EXPERT)
    sources = list(a.sources)
    if a.capture:
        trace = a.trace or str(Path.cwd() / "glm-routing.bin")
        usage = a.usage or str(Path.cwd() / "glm-usage.txt")
        capture(a.capture, a.max_new, a.prompts, trace, usage)
        sources.append(trace)
    if not sources:
        ap.error("no routing records: pass one or more sources, or --capture CONFIG")
    freq = load(sources, nl, ne)
    lines: list[str] = []
    sink = lines.append if a.out else print
    report(freq, nl, ne, a.slots, a.l2_slots, a.target, out=sink)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
