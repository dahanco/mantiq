"""Tune the GLM engine's CPU lane for this PC (./mntq.sh --calibrate) - Strata's tools/calibrate.py, for GLM-5.3-Flash.

On one or two GPUs most of a token's experts are not in VRAM; the CPU LANE computes the ones in the RAM tier while the
GPU runs the rest.  Two of its settings depend on the PC more than on the model:
  STRATA_GLM_PCIE_SHARE  the share of a route's RAM-tier experts copied over PCIe and run on the GPU instead of on the
                         CPU (0: the CPU takes every one).  A fast PCIe link and a slow CPU want more; an x8 link beside
                         a many-core CPU wants none.  The engine starts with a guess from timing one expert each way.
  STRATA_GLM_CPU_LANE    the CPU threads that compute them.  Every core is not always fastest: the experts are read
                         from RAM, and past the memory's bandwidth more threads only wait (or, on a hybrid CPU, the
                         efficiency cores make the rest wait for them).  Nor is the engine's own count always enough:
                         it takes two thirds of CPUs without SMT siblings, which a VM's vCPUs also are - on one GPU
                         and such CPUs the engine runs with STRATA_GLM_CPU_LANE_MAX set to this PC's CPUs, so counts
                         above its own, up to the pool it reports (INFO cpu_threads_max), are measured too.
  STRATA_GLM_READ_CHUNKS the pieces (x3) each expert the decode reads from the SSD comes in, when part of the model
                         lives there.  The best size depends on the drive, its driver and the CPU: 8 (the engine's) on
                         a Linux NVMe, 4 on a Windows laptop's Intel RST RAID (#67).
All are measured through ONE running engine - per-request `strata_tune` keys (pcie_frac, cpu_threads, read_chunks) -
because an engine start reads the whole expert set from disk (minutes).  Decode speed only: a prompt balances its CPU
and PCIe shares itself, layer by layer.

Every measurement answers prompts it has not answered before (PROMPTS), the way a chat's text is new to the expert
tiers, and the engine works on a copy of this PC's expert usage file (usage_copy) and saves no learned expert profile
(without_writes) - its own answers must not lead the next start's warm-up.

The PCIe share is picked the way llama.cpp's scripts/moe-cache-tune.py picks its cache size and split, instead of a
raw sweep of a few noisy points: the engine is run at two expert-cache sizes (`--expert-cache`, a restart each) x a
few shares, and a linear model of a token is fitted to those runs -
    ms/token = rest + up*uploads + cpu*cpu-work + lane + overlap
where the PCIe share sends `share` of the RAM-tier experts over the link (uploads) and the rest to the CPU lane
(cpu-work) - the engine's own RAM-tier counter (STAT ram_fetch) says how many a token reads, so a bigger VRAM cache
(a smaller upload count) is what makes the two paths tell apart.  The model then predicts the shares that were NOT
run, at every measured size, and the fastest is chosen - the split comes from a model of this PC, not from a few
points.  Changing the cache size needs a restart per value, so only the engine's own size and CACHE_FRACS of it are
measured.  The share stored is the model's best AT THE CONFIG'S OWN cache size (the config owns `--expert-cache`);
the size the model would pick is reported only.  A machine or build that does not report what the model needs (its
cache size, its RAM-tier counter) falls back to the plain sweep.

A setting is kept only when it beats the engine's own by more than MIN_GAIN in an interleaved re-measurement - the
expert tiers follow the text and the OS adds noise, so single measurements differ by a few percent.

    python tools/calibrate_glm.py mntq-<model>.json      # measure and print; ./mntq.sh --calibrate also saves it
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

MIN_GAIN = 0.03                          # a setting must beat the engine's own by this much to be kept
PCIE_SHARES = (0.0, 0.1, 0.25, 0.5, 0.75)   # the plain sweep, when the time model cannot be measured on this build
# the time model's calibration set: at every cache size, the two ends of the split (0: the CPU takes every RAM-tier
# expert, 1: the lane is off and everything is copied over PCIe) and the middle - the ends are what tell the PCIe and
# CPU costs apart (see the module docstring)
CALIB_SHARES = (0.0, 0.5, 1.0)
SHARES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)   # the shares the fitted model predicts and picks from
CACHE_FRACS = (0.6,)                     # other expert-cache sizes to measure, as fractions of the engine's own slots
# a sweep stops once a setting falls this far behind its best so far: the next ones go further the same way, and a
# share far past the best runs at a crawl (one 3090 on PCIe 3.0 x8: 20.9 tok/s at none, 8.1 at 0.10, 2.1 at 0.75 -
# its last three shares took ~8 of the tuning's 19 minutes)
STOP_BELOW = 0.7
MAX_NEW = 128
# A fresh group of three - code, an explanation, a list - for every measurement.  The expert tiers follow the text they
# serve: three prompts answered over and over left them holding just those answers' experts (Mantiq-L on 9 GPUs: 99.2%
# VRAM hits and 2.6 experts a token on the CPU, 31.8 tok/s, where a chat on that engine had 95-96%, 13-16 on the CPU
# and 24-25 tok/s), so the CPU lane's settings were measured with next to nothing on it.  A run takes ~20 groups.
CODE = ("merges two sorted lists into one sorted list", "checks whether a string is a palindrome",
        "counts the words in a text file", "returns the n-th Fibonacci number with memoization",
        "parses a date like 2024-03-15 into day, month and year", "removes duplicates from a list and keeps the order",
        "finds the longest common prefix of a list of strings", "converts a Roman numeral to an integer",
        "rotates a matrix by 90 degrees", "validates an email address with a regular expression",
        "groups a list of words by their first letter", "computes the median of a list of numbers",
        "flattens a nested list of any depth", "checks whether two strings are anagrams",
        "returns the prime numbers below n", "reverses the words in a sentence",
        "reads a CSV file and sums one of its columns", "converts Celsius to Fahrenheit for a list of values",
        "finds the second largest number in a list", "splits a list into chunks of a given size")
EXPLAIN = ("a refrigerator moves heat from inside to outside", "vaccines train the immune system",
           "a bill becomes a law in a parliament", "GPS finds a position on Earth", "bread dough rises",
           "a bank decides on a loan", "tides follow the Moon", "a compiler turns source code into a program",
           "plants turn sunlight into sugar", "an electric motor turns current into motion",
           "inflation changes what money buys", "a search engine ranks web pages", "volcanoes form",
           "noise-cancelling headphones work", "the heart moves blood through the body",
           "a hash table finds a key quickly", "rainbows form", "a jet engine makes thrust",
           "public key encryption keeps a message secret", "coffee is decaffeinated")
LISTS = ("European capitals", "chemical elements", "famous painters", "programming languages", "African countries",
         "musical instruments", "planets and moons of the solar system", "Olympic sports", "kitchen herbs",
         "inventions of the 20th century", "dog breeds", "Shakespeare plays", "rivers of Asia", "board games",
         "computer scientists", "types of clouds", "Greek gods", "cheeses of France", "bridges around the world",
         "mathematicians")
PROMPTS = tuple(p for c, e, l in zip(CODE, EXPLAIN, LISTS) for p in (
    f"Write a Python function that {c}, with a docstring and two tests.",
    f"Explain in two paragraphs how {e}.",
    f"List twelve {l} with one sentence about each."))
GROUP = 3                                # prompts a measurement (the median of their rates)
SHARE_ENV, THREADS_ENV, CHUNKS_ENV = "STRATA_GLM_PCIE_SHARE", "STRATA_GLM_CPU_LANE", "STRATA_GLM_READ_CHUNKS"
KEYS = (SHARE_ENV, THREADS_ENV, CHUNKS_ENV)   # the environment settings a calibration owns (apply() sets or clears them)
# the calibration's own engine only: a CPU pool this large, the engine's own count active, so a request can try more
MAX_ENV = "STRATA_GLM_CPU_LANE_MAX"
READ_CHUNKS = (4, 2)                     # tried against the engine's own (8) when part of the model is read from the SSD


def prompt_ids(cfg: dict) -> list[list[int]]:
    """PROMPTS as the server sends them: the model's chat template, thinking off (decode speed, not reasoning)."""
    import strata_tokenizer as ST
    from serve.frontend import ChatTemplate, effort_kwargs
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    tpl_path = tpath / "chat_template.jinja"
    tpl = ChatTemplate(tpl_path if tpl_path.exists() else ROOT / "serve" / "chat_template.jinja")
    return [tok.encode(tpl.render([{"role": "user", "content": p}], **effort_kwargs("none")), parse_special=True)
            for p in PROMPTS]


def thread_candidates(default: int, extra=()) -> list[int]:
    """The engine's own count, and fewer: three quarters, two thirds, a half and a quarter (at least 2), plus `extra`
    (the performance cores less one on a hybrid CPU, one socket's cores less one on a 2-socket PC), no repeats."""
    c = [default]
    for w in (round(default * 3 / 4), round(default * 2 / 3), round(default / 2), round(default / 4), *extra):
        if 2 <= w < default and w not in c:
            c.append(w)
    return c


def more_thread_candidates(default: int, most: int) -> list[int]:
    """Counts above the engine's own, up to a pool of `most` threads: the CPUs less two (the engine's main and service
    threads keep theirs - all 16 of a 16-vCPU guest ran at 11.4 tok/s against 19.4 with 14) and halfway there,
    ascending; [] when the pool has no room above the engine's own."""
    top = most - 2
    return sorted({w for w in ((default + top + 1) // 2, top) if default < w})


def lane_max() -> int:
    """The CPUs this process may run on (its affinity where the OS has one), at most the engine's 64."""
    import os
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 0
    return min(64, n)


def smt_visible(sys_cpu: Path = Path("/sys/devices/system/cpu")) -> bool:
    """Whether any CPU has an SMT sibling (its topology's thread_siblings_list names more than itself).  There the
    engine's own count is already a thread per physical core, and more only adds siblings or another socket."""
    try:
        return any(len(cpu_list(f.read_text())) > 1 for f in sys_cpu.glob("cpu[0-9]*/topology/thread_siblings_list"))
    except (OSError, ValueError):
        return False


def lane_max_env(cfg: dict, sys_cpu: Path = Path("/sys/devices/system/cpu")) -> dict:
    """The calibration engine's MAX_ENV: a pool of this PC's CPUs - on one GPU without SMT siblings only (a split's
    GPUs have a pool each or share one, and on SMT the counts above are siblings); {} elsewhere."""
    if len(cfg.get("gpu") or [0]) != 1 or smt_visible(sys_cpu):
        return {}
    n = lane_max()
    return {MAX_ENV: str(n)} if n > 0 else {}


def cpu_list(text: str) -> list[int]:
    """A sysfs CPU list ("0-3,8,10-11") -> its CPU numbers."""
    out = []
    for part in text.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def host_thread_extras(sys_cpu: Path = Path("/sys/devices/system/cpu"),
                       hybrid_cores: Path = Path("/sys/devices/cpu_core/cpus")) -> list[int]:
    """From the CPU topology: one socket's CPUs less one (several sockets), the performance cores' CPUs less one (a
    hybrid CPU).  [] when it cannot be read."""
    try:
        online = cpu_list((sys_cpu / "online").read_text())
        pkg = {c: (sys_cpu / f"cpu{c}" / "topology" / "physical_package_id").read_text().strip() for c in online}
    except (OSError, ValueError):
        return []
    out = []
    sockets = sorted(set(pkg.values()))
    if len(sockets) >= 2:
        out.append(min(sum(1 for p in pkg.values() if p == s) for s in sockets) - 1)
    try:
        p_cpus = [c for c in cpu_list(hybrid_cores.read_text()) if c in pkg]
        if 0 < len(p_cpus) < len(online):
            out.append(len(p_cpus) - 1)
    except (OSError, ValueError):
        pass
    return [w for w in out if w >= 2]


def pick(measured: dict, default_key, min_gain: float = MIN_GAIN):
    """The key with the best median tok/s, or `default_key` unless the best beats it by more than min_gain."""
    med = {k: statistics.median(v) for k, v in measured.items() if v}
    if not med or default_key not in med:
        return default_key
    best = max(med, key=med.get)
    return best if med[best] > med[default_key] * (1.0 + min_gain) else default_key


def tune_keys(share, threads, chunks=None) -> dict:
    """The request's `strata_tune`: None = what the engine started with."""
    t = {}
    if share is not None:
        t["pcie_frac"] = float(share)
    if threads is not None:
        t["cpu_threads"] = int(threads)
    if chunks is not None:
        t["read_chunks"] = int(chunks)
    return t


def with_arg(args: list, flag: str, value: str | None) -> list:
    """`args` with `flag value` set (replaced if present), or removed when value is None - a restart at another cache
    size (tools/calibrate.py does the same for --pool-workers)."""
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, value]
    return out


def ram_fetch(engine) -> float:
    """The RAM-tier experts a token read, from the engine's last STAT (0 when it reports none): the size of the
    uploads-plus-CPU work the time model splits - a build that does not report it leaves the model unmeasurable."""
    try:
        return float((getattr(engine, "stat", None) or {}).get("ram_fetch") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def reads_disk(info: dict) -> bool:
    """Whether part of the model lives on the SSD: more experts than the VRAM and RAM tiers have slots."""
    try:
        experts = int(info.get("experts") or 0)
        held = int(info.get("vram_slots") or 0) + int(info.get("ram_slots") or 0)
    except (TypeError, ValueError):
        return False
    return experts > held > 0


class Session:
    """One running engine: decode tok/s for a setting (the median of the rates of the next GROUP prompts - each
    measurement its own, in turn through `ids_list`)."""

    def __init__(self, engine, ids_list, group: int = GROUP):
        self.engine = engine
        self.ids_list = ids_list
        self.group = max(1, min(group, len(ids_list)))
        self.at = 0

    def next_group(self) -> list:
        n = len(self.ids_list)
        out = [self.ids_list[(self.at + i) % n] for i in range(self.group)]
        self.at = (self.at + self.group) % n
        return out

    def rate(self, tune: dict | None = None) -> float:
        rates = []
        for ids in self.next_group():
            sampling = {"temperature": 0}
            if tune:
                sampling["strata_tune"] = tune
            n = sum(1 for t in self.engine.generate(ids, MAX_NEW, sampling, threading.Event()) if t is not None)
            ms = (self.engine.last or {}).get("decode_ms") or 0.0
            if n > 8 and ms > 0:
                rates.append(n / (ms / 1000.0))
        return statistics.median(rates) if rates else 0.0

    def rate_ram(self, tune: dict | None = None) -> tuple[float, float]:
        """(decode tok/s, RAM-tier experts a token read): the medians over the next group - the second read from the
        engine's own expert-tier counter after each answer, so the time model has this PC's routing, not a guess."""
        rates, fetched = [], []
        for ids in self.next_group():
            sampling = {"temperature": 0}
            if tune:
                sampling["strata_tune"] = tune
            n = sum(1 for t in self.engine.generate(ids, MAX_NEW, sampling, threading.Event()) if t is not None)
            ms = (self.engine.last or {}).get("decode_ms") or 0.0
            if n > 8 and ms > 0:
                rates.append(n / (ms / 1000.0))
                fetched.append(ram_fetch(self.engine))
        return (statistics.median(rates) if rates else 0.0,
                statistics.median(fetched) if fetched else 0.0)

    def warm_up(self, rounds: int = 2):
        for _ in range(rounds):
            self.rate()


# --- the time model (llama.cpp's scripts/moe-cache-tune.py, for the shares this engine predicts) -------------------
# A token's work the model splits is the RAM-tier experts it reads: `share` of them the PCIe share sends over the link
# and the rest the CPU lane computes.  ms/token = rest + up*uploads + cpu*cpu_work + lane*(the lane runs) +
# overlap*min(uploads, cpu_work).  The two ends of CALIB_SHARES (all CPU / all PCIe) and a second cache size are what
# let up and cpu be told apart: with one cache size uploads + cpu_work is the same count for every share, so the two
# columns are one another's mirror and the fit is undetermined.

def model_ms(m: dict, up: float, cpu: float) -> float:
    """The model's ms a token: the rest, the experts the link moved, the experts the CPU lane computed, the lane's own
    cost while it runs, and what the two paths overlap."""
    return m["base"] + m["up"] * up + m["cpu"] * cpu + (m["split"] if cpu > 0 else 0.0) + m["overlap"] * min(up, cpu)


def solve(A: list, b: list) -> list:
    """Least squares by the normal equations (Gauss-Jordan with partial pivoting); ValueError when the runs leave the
    model's terms dependent (a machine the calibration set cannot separate)."""
    n = len(A[0])
    M = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] +
         [sum(A[k][i] * b[k] for k in range(len(A)))] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        if abs(M[i][i]) < 1e-12:
            raise ValueError("the runs do not determine the time model")
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [x - f * y for x, y in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit(runs: list) -> tuple:
    """The time model from measured (cache size, share) runs, the relative error per run, and whether it is physical.
    A run carries the decode ms a token and the RAM-tier experts a token read; `share` of them went over PCIe."""
    y = [r["ms"] for r in runs]
    rows = [[1.0, r["share"] * r["ram"], (1.0 - r["share"]) * r["ram"], float(r["share"] < 1.0),
             min(r["share"], 1.0 - r["share"]) * r["ram"]] for r in runs]

    def solve_model(cols: dict):
        coef = dict(zip(cols, solve([[row[i] for i in cols.values()] for row in rows], y)))
        if coef.get("split", 0.0) < 0.0:                 # noise: the lane's own cost cannot be negative
            del cols["split"]
            return solve_model(cols)
        return {k: coef.get(k, 0.0) for k in ("base", "up", "cpu", "split", "overlap")}

    def physical(m: dict) -> bool:
        # overlap hides at most the cheaper path, and the rest of a token is a good part of the fastest run
        return (m["up"] > 0 and m["cpu"] > 0 and m["base"] >= 0.2 * min(y)
                and -m["overlap"] <= min(m["up"], m["cpu"]))

    try:
        m = solve_model({"base": 0, "up": 1, "cpu": 2, "split": 3, "overlap": 4})
    except ValueError:
        return None, [], False
    if not physical(m):
        try:                                             # the overlap term can fit noise: without it, if that helps
            m = solve_model({"base": 0, "up": 1, "cpu": 2, "split": 3})
        except ValueError:
            return None, [], False
    errs = [(model_ms(m, r["share"] * r["ram"], (1.0 - r["share"]) * r["ram"]) - yy) / yy
            for r, yy in zip(runs, y)]
    return m, errs, physical(m)


def predict_model(m: dict, ram: float) -> tuple:
    """(share, t/s) at every SHARES from the model, and its pick: the lowest share within 5% of the fastest - a share
    past the optimum only adds PCIe traffic, and single measurements differ by a few percent anyway."""
    ms = [model_ms(m, s * ram, (1.0 - s) * ram) for s in SHARES]
    picks = list(zip(SHARES, [1000.0 / x for x in ms]))
    k = min(i for i in range(len(ms)) if ms[i] <= 1.05 * min(ms))
    return picks, (SHARES[k], 1000.0 / ms[k])


def share_points(cfg: dict, own_session: "Session", ids_list, own: int, start_engine, say=print) -> tuple:
    """Measure the calibration set: the engine's own cache size and CACHE_FRACS of it (a restart each - `--expert-cache
    N` is an engine argument, not a request's), at every CALIB_SHARES.  Returns (runs, sizes), or ([], [own]) when no
    run reported a RAM-tier count (an older build: measure() sweeps the shares plainly instead)."""
    sizes = list(dict.fromkeys([own] + [max(1, round(own * f)) for f in CACHE_FRACS]))
    runs = []
    for size in sizes:
        if size == own:
            engine, s = own_session.engine, own_session
        else:
            say(f"    restarting the engine with --expert-cache {size} ...")
            engine = start_engine({**cfg, "args": with_arg(list(cfg.get("args") or []), "--expert-cache", str(size))})
            s = Session(engine, ids_list)
            s.warm_up(1)
        got = False                                      # did any run report a RAM-tier count?
        try:
            for share in CALIB_SHARES:
                rate, ram = s.rate_ram(tune_keys(share, None))
                say(f"    cache {size} experts, PCIe share {share:.2f}: {rate:.1f} tok/s, {ram:.1f} RAM-tier experts a token")
                if rate > 0 and ram > 0:
                    runs.append({"size": size, "share": share, "ms": 1000.0 / rate, "ram": ram})
                    got = True
        finally:
            if size != own:
                close(engine)
        if not got:                                      # a build without the counter: no need to restart the rest
            return [], [own]
    return runs, sizes


def usage_copy(cfg: dict, tmp: Path) -> dict:
    """The env that has the calibration's engine read and write a copy of this PC's expert usage file in `tmp`: the
    engine saves the usage after every answer and the next start's warm-up loads the experts in that order, so ~60
    answers to the tuning's prompts would lead it with their experts.  The file is the config's STRATA_GLM_USAGE
    (opt-in); without one (or "0") the engine keeps none, as the config's own starts do."""
    own = str((cfg.get("env") or {}).get("STRATA_GLM_USAGE", ""))
    if own in ("", "0"):
        return {"STRATA_GLM_USAGE": "0"}
    dst = tmp / "expert_usage.txt"
    if Path(own).is_file():
        shutil.copyfile(own, dst)
    return {"STRATA_GLM_USAGE": str(dst)}


# engine arguments that write the user's files: the learned expert profile (--expert-profile-save) and the routing trace
WRITES = ("--expert-profile-save", "--expert-profile-save-every", "--dump-routing")


def without_writes(args: list) -> list:
    """`args` less WRITES and their values: the tuning's prompts must not end up in this PC's learned profile (the
    next start would fill VRAM with their experts) or replace a routing trace."""
    out, i = [], 0
    while i < len(args):
        if args[i] in WRITES:
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Measure on the engine `cfg` describes; returns {"settings": {env: value}, "report": {...}}.  The engine runs
    with the config's environment less the settings a calibration owns, so the start's own choice is the baseline,
    on a copy of the expert usage file (usage_copy) and without the arguments that write the user's files
    (without_writes).
    `start_engine(cfg)` returns a started engine (serve.server.StrataEngine, or a stand-in in tests)."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env, engine_args

        def start_engine(c):
            return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=child_env(c))
    with tempfile.TemporaryDirectory(prefix="mntq-calibrate-") as tmp:
        base = {**cfg, "args": without_writes(list(cfg.get("args") or [])),
                "env": {**apply(cfg.get("env") or {}, {}), **usage_copy(cfg, Path(tmp)), **lane_max_env(cfg)}}
        kept = {k: v for k, v in (cfg.get("env") or {}).items() if k in KEYS}
        return measure(base, prompt_ids(cfg), start_engine, say, host_thread_extras(), kept)


def measure(cfg: dict, ids_list, start_engine, say=print, extra_threads=(), kept=None) -> dict:
    """`kept`: the config's own values of KEYS (an earlier calibration's, or set by hand) - the incumbent: the sweeps'
    choice replaces them only when it beats them, all together, by more than MIN_GAIN interleaved (#277)."""
    t0 = time.time()
    report: dict = {}
    settings: dict = {}
    say("  Loading the model for the measurements (it reads the experts into RAM and VRAM first) ...")
    eng = start_engine(cfg)
    try:
        info = dict(getattr(eng, "info", {}) or {})
        d_threads = int(info.get("cpu_threads") or 0)
        d_share = float(info.get("pcie_share", 1.0))
        report["default"] = {"pcie_share": round(d_share, 2), "cpu_threads": d_threads}
        if d_threads <= 0:
            say("    no CPU lane on this PC (every expert the GPU lacks is copied over PCIe): nothing to tune")
            report["seconds"] = round(time.time() - t0)
            return {"settings": {}, "report": report}
        s = Session(eng, ids_list)
        s.warm_up()
        # 1. the PCIe share.  With the engine's own cache size and RAM-tier counter the time model is measured (a
        #    restart at CACHE_FRACS of that size) and its best share at the config's own size is kept; without them
        #    (an older build, or a cache that fits VRAM) the shares are swept plainly.  `None` is the engine's own.
        own = int(info.get("expert_slots") or 0)
        best_share = None
        by_share = None                      # the fallback sweep's points, when the time model cannot be measured
        if own > 0:
            runs, sizes = share_points(cfg, s, ids_list, own, start_engine, say)
            m, errs, ok = fit(runs) if len(runs) >= 3 else (None, [], False)
            if m is not None and ok and errs and max(abs(e) for e in errs) <= 0.10:
                ram = {sz: statistics.median([r["ram"] for r in runs if r["size"] == sz]) for sz in sizes}
                table = {sz: predict_model(m, ram[sz]) for sz in ram}
                best_size = max(table, key=lambda sz: table[sz][1][1])
                best_share = table[own][1][0]
                report["share_model"] = {
                    "points": [{"size": r["size"], "share": r["share"], "tok_s": round(1000.0 / r["ms"], 1),
                                "ram_per_token": round(r["ram"], 1)} for r in runs],
                    "machine": {"uploads_ms_per_expert": round(m["up"], 3), "cpu_ms_per_expert": round(m["cpu"], 3),
                                "lane_ms": round(m["split"], 3), "overlap_ms_per_expert": round(m["overlap"], 3),
                                "rest_ms": round(m["base"], 3), "fit_within_pct": round(100 * max(abs(e) for e in errs), 1)},
                    "predicted": {str(sz): [round(tps, 1) for _, tps in table[sz][0]] for sz in sorted(table)},
                    "shares": list(SHARES), "best_size": best_size}
                say("    time model: uploads %.2f ms an expert, CPU lane %.2f ms an expert, the lane costs %.1f ms, "
                    "overlap %.2f, rest %.1f ms a token (fit within %.1f%%)" %
                    (m["up"], m["cpu"], m["split"], m["overlap"], m["base"], 100 * max(abs(e) for e in errs)))
                say(f"    the model's best share at the config's {own} experts is {best_share:.2f}; "
                    f"it would size the cache at {best_size} experts "
                    f"({table[best_size][1][1]:.1f} tok/s predicted)")
            else:
                say("    the runs do not fit a time model (was the machine busy?): sweeping the shares instead")
        if best_share is None:
            by_share = {None: [s.rate()]}
            say(f"    the engine's own split (PCIe share {d_share:.2f}): {by_share[None][0]:.1f} tok/s")
            for f in PCIE_SHARES:
                by_share[f] = [s.rate(tune_keys(f, None))]
                say(f"    PCIe share {f:.2f}: {by_share[f][0]:.1f} tok/s")
                if by_share[f][0] < STOP_BELOW * max(v[0] for v in by_share.values()):
                    break
            best_share = max(by_share, key=lambda k: by_share[k][0])
            report["share_sweep"] = {("own" if k is None else f"{k:.2f}"): v for k, v in by_share.items()}
        # 2. the CPU threads, at that share: the engine's own count and fewer, then more - up to the pool the engine
        #    reports (cpu_threads_max: MAX_ENV's pool, or the engine's own count where MAX_ENV does not apply - a pool
        #    a split shares, DFlash2's two lanes, an engine without it), as a request's cpu_threads is capped at the
        #    pool and a higher count would only measure the engine's own again (#178) - upward only while each count
        #    beats the best so far
        by_threads = {}
        for w in sorted(thread_candidates(d_threads, extra_threads), reverse=True):
            by_threads[w] = [s.rate(tune_keys(best_share, None if w == d_threads else w))]
            say(f"    {w} CPU threads: {by_threads[w][0]:.1f} tok/s")
            if by_threads[w][0] < STOP_BELOW * max(v[0] for v in by_threads.values()):
                break
        try:
            most = int(info.get("cpu_threads_max") or 0)
        except (TypeError, ValueError):
            most = 0
        for w in more_thread_candidates(d_threads, most):
            best_so_far = max(v[0] for v in by_threads.values())
            by_threads[w] = [s.rate(tune_keys(best_share, w))]
            say(f"    {w} CPU threads: {by_threads[w][0]:.1f} tok/s")
            if by_threads[w][0] <= best_so_far:
                break
        best_threads = max(by_threads, key=lambda k: by_threads[k][0])
        # 3. the winner against the engine's own, interleaved, three times each
        dflt, cand = (None, d_threads), (best_share, best_threads)
        confirm = {dflt: [], cand: []}
        if cand != dflt:
            say("    confirming against the engine's own settings ...")
            for _ in range(3):
                for k in (dflt, cand):
                    confirm[k].append(s.rate(tune_keys(k[0], None if k[1] == d_threads else k[1])))
        chosen = pick(confirm, dflt) if cand != dflt else dflt
        # the shares a fallback swept are in report["share_sweep"] already; a model run keeps report["share_model"]
        report.update(thread_sweep={str(k): v for k, v in by_threads.items()},
                      confirm={f"{'own' if k[0] is None else f'{k[0]:.2f}'}/{k[1]}": v for k, v in confirm.items()})
        rates = confirm.get(chosen) or (by_share.get(None) if (chosen == dflt and by_share) else None)
        # 4. the disk reads' size, at those settings - only when part of the model is read from the SSD while it
        #    answers; a size is kept as the others are, by beating the engine's own in an interleaved re-measurement
        chunks = None
        if reads_disk(info):
            at = (chosen[0], None if chosen[1] == d_threads else chosen[1])
            by_chunks = {None: [s.rate(tune_keys(*at))]}
            say(f"    the engine's own disk reads (8 pieces an expert part): {by_chunks[None][0]:.1f} tok/s")
            for c in READ_CHUNKS:
                by_chunks[c] = [s.rate(tune_keys(*at, c))]
                say(f"    {c} pieces: {by_chunks[c][0]:.1f} tok/s")
                if by_chunks[c][0] <= by_chunks[None][0]:   # fewer only get slower from there (Mercury: 8 > 4 > 2)
                    break
            best_c = max(by_chunks, key=lambda k: by_chunks[k][0])
            confirm_c = {None: [], best_c: []}
            if best_c is not None:
                say("    confirming the disk reads' size ...")
                for _ in range(3):
                    for k in (None, best_c):
                        confirm_c[k].append(s.rate(tune_keys(*at, k)))
                chunks = pick(confirm_c, None)
            report.update(chunk_sweep={("own" if k is None else str(k)): v for k, v in by_chunks.items()},
                          chunk_confirm={("own" if k is None else str(k)): v for k, v in confirm_c.items()})
            if chunks is not None:
                rates = confirm_c[chunks]
        final = (chosen[0], None if chosen[1] == d_threads else chosen[1], chunks)
        # 5. the config's own values (the incumbent), as one combination against the sweeps' choice, interleaved
        inc = incumbent(kept)
        if inc is not None and inc != final:
            say("    confirming against the config's own settings ...")
            confirm_k = {inc: [], final: []}
            for _ in range(3):
                for k in (inc, final):
                    confirm_k[k].append(s.rate(tune_keys(*k)))
            report["kept_confirm"] = {"/".join("own" if x is None else str(x) for x in k): v for k, v in confirm_k.items()}
            final = pick(confirm_k, inc)
            rates = confirm_k[final]
    finally:
        close(eng)
    if final[0] is not None:
        settings[SHARE_ENV] = f"{final[0]:.2f}"
    if final[1] is not None:
        settings[THREADS_ENV] = str(final[1])
    if final[2] is not None:
        settings[CHUNKS_ENV] = str(final[2])
    report["tok_s"] = round(statistics.median(rates), 1) if rates else None
    report["seconds"] = round(time.time() - t0)
    return {"settings": settings, "report": report}


def engine_error(log: str | None, since: int = 0) -> str | None:
    """The engine's own reason for a failed start: the last line of its log written after byte `since` that is its own
    ("glm ...", "mntq ..." or "ERR ..."), else the last line there; None without a log or a new line."""
    if not log:
        return None
    try:
        with open(log, "rb") as f:
            f.seek(since)
            lines = [x.strip() for x in f.read()[-16384:].decode("utf-8", "replace").splitlines() if x.strip()]
    except OSError:
        return None
    return next((x for x in reversed(lines) if x.startswith(("glm", "mntq", "ERR"))), lines[-1] if lines else None)


def close(eng):
    proc = getattr(eng, "proc", None)
    if proc is None:
        return
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(120)
    except Exception:
        proc.kill()


def incumbent(kept) -> tuple | None:
    """The config's own (share, threads, chunks) as tune_keys takes them (None: the engine's own); None without any."""
    try:
        if not kept:
            return None
        share = float(kept[SHARE_ENV]) if kept.get(SHARE_ENV) not in (None, "") else None
        threads = int(kept[THREADS_ENV]) if kept.get(THREADS_ENV) not in (None, "") else None
        chunks = int(kept[CHUNKS_ENV]) if kept.get(CHUNKS_ENV) not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return None if (share, threads, chunks) == (None, None, None) else (share, threads, chunks)


def apply(env: dict, settings: dict) -> dict:
    """`env` (a config's "env" block) with the calibrated settings; one the calibration did not set is removed, so the
    engine chooses it again and an older calibration's value never lingers."""
    out = {k: v for k, v in (env or {}).items() if k not in KEYS}
    for k in KEYS:
        if settings.get(k) is not None:
            out[k] = str(settings[k])
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate_glm.py <mntq-*.json>")
    import mntq_names
    res = run(mntq_names.env_in(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))))
    print(json.dumps(res, indent=1))
