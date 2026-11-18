"""Benchmark cold-state prompt processing through the engine's GEN protocol.

Example: python tools/prefill_ab.py --config models_cfg/model.json --chunks 0 256
Logs and JSON go to a temporary directory unless --output-dir is supplied.
No model config or expert profile is modified. Every measured turn must report zero reused tokens.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mntq
import serve.server as server
import strata_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--chunks", type=int, nargs="+", default=[0, 256])
    ap.add_argument("--tokens", type=int, default=5312)
    ap.add_argument("--new", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--expert-cache")
    args = ap.parse_args()
    if min(args.tokens, args.new, args.repeats, args.timeout) < 1 or min(args.chunks) < 0:
        ap.error("counts must be positive; chunks must be nonnegative")
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="mntq-prefill-"))
    output.mkdir(parents=True, exist_ok=True)
    cfg = mntq.read_json(args.config.resolve())
    tokenizer = strata_tokenizer.Tokenizer.from_dir(Path(cfg["tokenizer"]))
    prompt = tokenizer.encode((ROOT / "README.md").read_text(encoding="utf-8"))
    prompt = (prompt * (args.tokens // len(prompt) + 1))[:args.tokens]
    results = []
    for chunk in args.chunks:
        flags = list(server.engine_args(cfg))
        while "--expert-profile-save" in flags:
            i = flags.index("--expert-profile-save")
            del flags[i:i + 2]
        overrides = {"--prefill": chunk}
        if args.expert_cache is not None:
            overrides["--expert-cache"] = args.expert_cache
        for flag, value in overrides.items():
            if flag in flags:
                flags[flags.index(flag) + 1] = str(value)
            else:
                flags.extend([flag, str(value)])
        env = server.child_env(cfg)
        env["STRATA_DECODE_TIMING"] = "1"
        if args.profile:
            env["MNTQ_PREFILL_PROF"] = "1"
        log_path = output / ("chunk-%d.log" % chunk)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([cfg["exe"], "--serve"] + flags, cwd=cfg.get("cwd") or ROOT,
                                    env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=log, text=True, bufsize=1)
            timer = threading.Timer(args.timeout, proc.kill)
            timer.start()
            try:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        raise RuntimeError("engine stopped during load: %s" % log_path)
                    if line.startswith("READY"):
                        break
                for turn in range(args.repeats + 1):
                    proc.stdin.write("GEN %d temperature=0 %s\n" %
                                     (args.new, ",".join(map(str, prompt))))
                    proc.stdin.flush()
                    tokens = []
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            raise RuntimeError("engine stopped during GEN: %s" % log_path)
                        fields = line.split()
                        if fields and fields[0] == "T":
                            tokens.append(int(fields[1]))
                        if fields and fields[0] == "ERR":
                            raise RuntimeError(line.strip())
                        if fields and fields[0] == "DONE":
                            if len(fields) < 9 or int(fields[8]) != 0:
                                raise RuntimeError("expected a full cold-state prefill: " + line.strip())
                            if int(fields[2]) != len(prompt):
                                raise RuntimeError("prompt length mismatch: " + line.strip())
                            row = {"chunk": chunk, "warmup": turn == 0, "turn": turn,
                                   "prompt_ms": float(fields[3]), "decode_ms": float(fields[4]),
                                   "prompt_tps": len(prompt) * 1000 / float(fields[3]),
                                   "output_tokens": tokens, "done": line.strip()}
                            results.append(row)
                            print(json.dumps(row), flush=True)
                            break
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.wait(timeout=60)
                if proc.returncode:
                    raise RuntimeError("engine exit %d: %s" % (proc.returncode, log_path))
            finally:
                timer.cancel()
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
        (output / "results.json").write_text(
            json.dumps({"config": str(args.config), "tokens": args.tokens, "runs": results}, indent=2),
            encoding="utf-8")
    print("Results: %s" % output, flush=True)


if __name__ == "__main__":
    main()
