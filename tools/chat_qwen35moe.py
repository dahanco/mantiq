"""tools/chat_qwen35moe.py - an interactive chat over mntq's `--repl` mode.

    python tools/chat_qwen35moe.py --prompt "hello"        # first turn on the command line
    python tools/chat_qwen35moe.py                          # ask for the first turn on stdin

WHY THIS IS A SEPARATE PROCESS FROM THE ENGINE.  The engine carries no tokenizer and, for a qwen35moe pack, no
HTTP front end either: `--serve` is gated on MTP and the batched prompt path, and this pack has neither (no
`glm_prefill.cu`, and `mtp.cpp`'s hyper-connection graph is not this model's plain-residual one).  `--repl` is
the residency without them - the engine answers the `--tokens` prompt, prints `T <id>` per token and
`DONE <n> <stop|length>`, then takes one `GEN <max_new> <id,id,...>` line per turn on stdin and answers it on
the state the turn before left.  This script is the other half: it renders the prompt with the pack's own chat
template, keeps the per-turn bookkeeping the wire contract needs, and shows the answer as it arrives.

THE ONE PIECE OF STATE THAT MATTERS IS THE PENDING TOKEN.  A turn ends with the engine's `x` - the last token
it emitted - still at the current position, uncommitted (that is what lets a turn stop at the EOS or at the cap
without stranding the state on a draft row).  `GEN`'s token list therefore starts with whatever follows that
token in the conversation, not with the token itself: the engine reads the pending one back first.
"""
from __future__ import annotations

import argparse
import codecs
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_PACK = r"c:\tmp\pack-qwen35moe"
DEFAULT_GGUF = (r"C:\gguf\Qwopus\Qwopus3.6-35B-A3B-Coder-MTP-GGUF"
                r"\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf")
DEFAULT_PROFILE = r"c:\tmp\qwen35moe-profile.bin"

SPECIAL = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "eot": "<|endoftext|>",
    "think": "<think>",
    "unthink": "</think>",
}


class Printer:
    """Token ids in, text out - streamed, with the control tokens the answer carries kept out of the display.

    The byte-level BPE can split one character across tokens, so the bytes go through an incremental UTF-8
    decoder rather than `decode([id])` per token (which would print a replacement character per fragment).
    """

    def __init__(self, tk: T.Tokenizer, special: dict[str, int], dim: bool):
        self.tk = tk
        self.special = special
        self.dim = dim
        self.dec = codecs.getincrementaldecoder("utf-8")("replace")
        self.hidden = {special["im_end"], special["eot"]}
        self.n_text = 0

    def begin_turn(self) -> None:
        self.dec = codecs.getincrementaldecoder("utf-8")("replace")
        self.n_text = 0
        if self.dim:
            sys.stdout.write("\x1b[2m")

    def end_turn(self) -> None:
        self.dec.decode(b"", final=True)
        if self.dim:
            sys.stdout.write("\x1b[0m")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def feed(self, i: int) -> None:
        if i in self.hidden:
            return
        text = self.dec.decode(self.tk.token_bytes(i))
        if i == self.special["unthink"] and self.dim:
            sys.stdout.write("\x1b[0m")   # the reasoning is over: back to normal weight
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
            self.n_text += len(text)


def render_turn(msg: str, thinking: bool) -> str:
    """One user message and the assistant header the template would emit for the next generation."""
    out = "<|im_start|>user\n" + msg + "<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    out += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return out


def render_prompt(system: str, msg: str, thinking: bool) -> str:
    out = ""
    if system:
        out += "<|im_start|>system\n" + system + "<|im_end|>\n"
    return out + render_turn(msg, thinking)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="chat with a qwen35moe native pack through mntq --repl")
    ap.add_argument("--pack", default=DEFAULT_PACK)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--engine", default=os.path.join(ROOT, "build", "mntq.exe"))
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--system", default="", help="system message; empty means none")
    ap.add_argument("--prompt", default=None, help="the first user message; asked on stdin when omitted")
    ap.add_argument("--max-new", type=int, default=1024,
                    help="token cap per turn; the model reasons at length before it answers, so leave room")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--spec", type=int, default=4)
    ap.add_argument("--no-thinking", action="store_true", help="open the turn with an empty reasoning block")
    ap.add_argument("--no-dim", action="store_true", help="do not dim the reasoning block")
    ap.add_argument("--quiet", action="store_true", help="no per-turn speed line")
    a = ap.parse_args()

    if not os.path.exists(a.engine):
        raise SystemExit("no engine at %s (build it first)" % a.engine)
    thinking = not a.no_thinking

    tk = T.Tokenizer.from_dir(os.path.join(a.pack, "tokenizer"))
    special = {}
    for k, lit in SPECIAL.items():
        ids = tk.encode(lit, parse_special=True)
        if len(ids) != 1:
            raise SystemExit("tokenizer: %r encodes to %d tokens, expected 1" % (lit, len(ids)))
        special[k] = ids[0]

    first = a.prompt
    if first is None:
        sys.stdout.write("you> ")
        sys.stdout.flush()
        first = sys.stdin.readline()
        if not first:
            return 0
        first = first.rstrip("\n")

    prompt_ids = tk.encode(render_prompt(a.system, first, thinking), parse_special=True)
    if len(prompt_ids) + a.max_new > a.max_context:
        print("warning: %d prompt tokens + %d cap is past --max-context %d"
              % (len(prompt_ids), a.max_new, a.max_context), file=sys.stderr)

    # `--tokens-file`, not `--tokens`: a chat prompt is easily long enough to hit the command line's limit.
    tf = tempfile.NamedTemporaryFile("w", suffix=".tokens", delete=False)
    with tf:
        tf.write(",".join(map(str, prompt_ids)))
    tokens_file = tf.name
    cmd = [a.engine, "--pack", a.pack, "--native", a.gguf, "--expert-cache", "auto",
           "--expert-profile", a.profile, "--tokens-file", tokens_file,
           "--max-new", str(a.max_new), "--max-context", str(a.max_context),
           "--spec", str(a.spec), "--prefill", "0", "--repl", "--stop-eos"]

    print("loading %s (%d prompt tokens; the first turn is ~30 s cold)..."
          % (os.path.basename(a.pack), len(prompt_ids)), file=sys.stderr, flush=True)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1, text=True,
                         encoding="utf-8", errors="replace")

    printer = Printer(tk, special, dim=not a.no_dim)

    def read_answer() -> tuple[int, str, float]:
        """Print the tokens of one answer as they arrive; return (n, finish, ms)."""
        t0 = time.time()
        n, finish = 0, "?"
        printer.begin_turn()
        while True:
            line = p.stdout.readline()
            if not line:
                printer.end_turn()
                raise SystemExit("the engine closed the pipe (see its stderr above)")
            if line.startswith("T "):
                printer.feed(int(line[2:]))
            elif line.startswith("DONE"):
                f = line.split()
                n = int(f[1]) if len(f) > 1 else 0
                finish = f[2] if len(f) > 2 else "?"
                break
            elif line.startswith("ERR"):
                printer.end_turn()
                print("[engine] " + line.strip(), file=sys.stderr)
                return 0, "error", (time.time() - t0) * 1000.0
            else:
                sys.stderr.write("[engine] " + line)
        printer.end_turn()
        return n, finish, (time.time() - t0) * 1000.0

    def report(n: int, finish: str, ms: float) -> None:
        if finish == "length":
            print("[the turn ran into --max-new %d (or the context): raise the cap, or the answer is cut]"
                  % a.max_new, file=sys.stderr, flush=True)
        if a.quiet:
            return
        rate = 1000.0 * n / ms if ms > 0 else 0.0
        print("[%d tokens, %s, %.1f ms, %.1f tok/s]" % (n, finish, ms, rate), file=sys.stderr, flush=True)

    try:
        n, finish, ms = read_answer()
        report(n, finish, ms)
        while True:
            sys.stdout.write("you> ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                break
            msg = line.rstrip("\n")
            if msg.strip() in ("/exit", "/quit", "exit", "quit"):
                break
            if not msg.strip():
                continue
            # The engine's `x` is the last token it emitted.  After a `stop` that token is the assistant's
            # `<|im_end|>` and the template's next character is "\n"; after a `length` the message was cut
            # mid-answer, so it has to be closed before the next user block goes in.
            head = "\n" if finish != "length" else "<|im_end|>\n"
            delta = tk.encode(head + render_turn(msg, thinking), parse_special=True)
            p.stdin.write("GEN %d %s\n" % (a.max_new, ",".join(map(str, delta))))
            p.stdin.flush()
            n, finish, ms = read_answer()
            report(n, finish, ms)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
    finally:
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
            p.stdin.close()
        except Exception:
            pass
        try:
            p.wait(timeout=60)
        except Exception:
            p.kill()
        try:
            os.unlink(tokens_file)   # only now: the engine reads it while it parses its arguments
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
