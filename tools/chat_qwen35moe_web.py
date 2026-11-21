"""tools/chat_qwen35moe_web.py - a browser chat over mntq's `--repl` mode (Qwopus / qwen35moe).

    python tools/chat_qwen35moe_web.py                 # serve http://127.0.0.1:8760 and open the browser
    python tools/chat_qwen35moe_web.py --no-open       # serve only, do not open the browser
    python tools/chat_qwen35moe_web.py --prompt "hi"   # preload: load the engine and answer this first

WHY THIS IS NOT `serve/server.py`.  It was written while Mantiq's own chat (the launcher, the web UI, the OpenAI
endpoint) could not drive this pack: it goes through `--serve`, whose gate wanted `--spec >= 2` with a non-empty MTP
and the batched prompt path (`prefill.cpp`), and a qwen35moe pack had neither - no `glm_prefill.cu`, and `mtp.cpp`'s
graph was hyper-connection-shaped.  The MTP drafter landed since (its own plain-residual graph, `record_forward_plain`,
built from the file's NextN block by the setup) and `--serve` serves this pack now; this file stays as the older,
`--repl`-based front end.  Mantiq's
server also sends the WHOLE conversation with every request and leans on the engine's conversation-cache prefix
matching, which the native path does not implement.

This file is the small front end the pack CAN take: it keeps the engine resident in `--repl` and holds the
conversation on the python side, so each turn sends only what the turn adds - exactly the wire contract
`tools/chat_qwen35moe.py` uses, with an HTTP/SSE face instead of a terminal one.  Nothing in the engine is
touched.

THE ONE PIECE OF STATE THAT MATTERS IS THE PENDING TOKEN.  A turn ends with the engine's `x` - the last token
it emitted - still at the current position, uncommitted.  The next `GEN`'s token list therefore starts with
whatever follows that token in the conversation, not with the token itself: after a `stop` that token is the
assistant's `<|im_end|>` and the next character is "\\n"; after a `length` the answer was cut mid-message and
must be closed with `<|im_end|>\\n` first.  Getting this wrong shifts the whole context by one turn.
"""
from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_PACK = r"C:\tmp3\pack-qwen35moe"
DEFAULT_GGUF = (r"C:\gguf\Qwopus\Qwopus3.6-35B-A3B-Coder-MTP-GGUF"
                r"\Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf")
DEFAULT_PROFILE = r"C:\tmp3\qwen35moe-profile.bin"
# the draft layer `tools/mtp_from_gguf.py --out <pack>/mtp` writes; absent means the engine drafts nothing
DEFAULT_MTP = r"C:\tmp3\pack-qwen35moe\mtp"

SPECIAL = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "eot": "<|endoftext|>",
    "think": "<think>",
    "unthink": "</think>",
}

# --------------------------------------------------------------------------------- the chat template
# The pack ships its own jinja template, but a two-turn ChatML prompt is all this architecture needs, and
# hard-coding it is what the terminal client does - the two must agree on the prompt or the engine sees a
# different conversation than the one the browser shows.


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


class TokenPiece:
    """Token ids in, display pieces out.

    The byte-level BPE can split one character across tokens, so the bytes go through an incremental UTF-8
    decoder instead of `decode([id])` per token (which would print one replacement char per fragment).  The
    two reasoning markers are swallowed and reported as a flag instead, so the page can dim the block.
    """

    def __init__(self, tk: T.Tokenizer, special: dict[str, int], thinking: bool):
        self.tk = tk
        self.special = special
        self.hidden = {special["im_end"], special["eot"]}
        self.dec = codecs.getincrementaldecoder("utf-8")("replace")
        self.reasoning = thinking

    def feed(self, i: int) -> tuple[str, bool]:
        if i == self.special["think"]:
            self.reasoning = True
            return "", self.reasoning
        if i == self.special["unthink"]:
            self.reasoning = False
            return "", self.reasoning
        if i in self.hidden:
            return "", self.reasoning
        text = self.dec.decode(self.tk.token_bytes(i))
        return text, self.reasoning

    def flush(self) -> str:
        return self.dec.decode(b"", final=True)


# --------------------------------------------------------------------------------- the engine
class EngineGone(RuntimeError):
    pass


def mtp_info(rounds: int, ok: int, drafted: int, ms_total: float | None, tpr: float | None = None) -> dict:
    """One turn's drafting numbers, as the header shows them.

    `rounds` is how many verify windows used a draft, `ok`/`drafted` the drafts accepted and offered, and
    `tpr` the tokens a round produced (the window's own row plus every draft it accepted).  `ms_total` is the
    drafting time the turn spent, or None when the caller already has the per-round figure.
    """
    return {
        "rounds": rounds,
        "ok": ok,
        "drafted": drafted,
        "ms_round": round(ms_total / rounds, 3) if ms_total is not None and rounds else 0.0,
        "tpr": round(tpr, 3) if tpr is not None else (round((ok + rounds) / rounds, 3) if rounds else 0.0),
    }


class Engine:
    """The resident `mntq --repl` process and the per-turn bookkeeping it needs.

    Single conversation: every call is serialised on `lock`.  `finish` is the last turn's DONE word - it is
    what decides the head the next `GEN` carries (the pending-token rule above).
    """

    def __init__(self, a):
        self.exe = a.engine
        self.pack = a.pack
        self.gguf = a.gguf
        self.profile = a.profile
        self.mtp = a.mtp
        self.spec = a.spec
        self.max_context = a.max_context
        self.max_new = a.max_new
        self.system = a.system
        self.thinking = not a.no_thinking
        self.proc: subprocess.Popen | None = None
        self.tokens_file: str | None = None
        self.finish = "stop"
        self.started = False
        self.fresh = False          # the last turn was a New chat: the next message is a whole prompt
        self.lock = threading.Lock()
        self.t0 = 0.0

    # ---- lifecycle
    def _cmd(self, tokens_file: str, first_cap: int) -> list[str]:
        cmd = [self.exe, "--pack", self.pack, "--native", self.gguf, "--expert-cache", "auto"]
        if self.profile and os.path.exists(self.profile):
            cmd += ["--expert-profile", self.profile]
        # the MTP draft layer: only when the extraction has run, so an absent dir keeps the engine non-drafting
        if self.mtp and os.path.isdir(self.mtp):
            cmd += ["--mtp", self.mtp]
        cmd += ["--tokens-file", tokens_file, "--max-new", str(first_cap),
                "--max-context", str(self.max_context), "--spec", str(self.spec),
                "--prefill", "0", "--repl", "--stop-eos"]
        return cmd

    def _spawn(self, prompt_ids: list[int], first_cap: int) -> None:
        tf = tempfile.NamedTemporaryFile("w", suffix=".tokens", delete=False)
        with tf:
            tf.write(",".join(map(str, prompt_ids)))
        self.tokens_file = tf.name
        self.t0 = time.time()
        self.proc = subprocess.Popen(self._cmd(self.tokens_file, first_cap), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, bufsize=1, text=True,
                                     encoding="utf-8", errors="replace")

    def start(self, first_msg: str, first_cap: int) -> None:
        """Load the engine with `first_msg` as its prompt (the `--tokens` the REPL answers first)."""
        self.close()
        ids = encode(render_prompt(self.system, first_msg, self.thinking))
        self._spawn(ids, first_cap)
        self.started = True
        self.fresh = False

    def reset_context(self) -> None:
        """New chat on the SAME engine: the engine's `RESET` clears the sequence state - the K/V, the
        recurrence, the draft layer's own K/V - and the model, the VRAM expert tier and the RAM pool stay
        resident (nothing is loaded again).  The next message opens a conversation, so it goes as a whole
        prompt rather than a delta (`fresh`)."""
        with self.lock:
            self.finish = "stop"
            if self.proc is None or not self.started:
                self.fresh = True
                return
            assert self.proc.stdin is not None and self.proc.stdout is not None
            try:
                self.proc.stdin.write("RESET\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()      # the engine's `RESET ok`
                if not line or line.startswith("ERR"):
                    raise RuntimeError(line.strip() or "the engine closed its stdout")
            except Exception:                            # noqa: BLE001 - a dead engine: fall back to a reload
                self.close()
                self.started = False
                self.fresh = True
                return
            self.fresh = True

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        self.proc = None
        if self.tokens_file:
            try:
                os.unlink(self.tokens_file)   # only now: the engine reads it while it parses its arguments
            except OSError:
                pass
            self.tokens_file = None

    # ---- one turn
    def _gen_line(self, delta_ids: list[int], cap: int, prompt_only: bool = False) -> None:
        # A turn that follows another opens with the token the template puts after the engine's pending one.
        # A turn that opens a conversation (the first after a New chat) IS the prompt: the engine has no
        # pending token to read back, so nothing precedes it.
        ids = list(delta_ids) if prompt_only else \
            encode("\n" if self.finish != "length" else "<|im_end|>\n") + delta_ids
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write("GEN %d %s\n" % (cap, ",".join(map(str, ids))))
        self.proc.stdin.flush()

    def stream(self, cap: int):
        """Yield ('tok', id) per generated token, then ('done', n, finish, ms, mtp).  Raises EngineGone.

        `mtp` is the turn's drafting report (`mtp_info`) or an empty dict when the engine drafted nothing.
        """
        assert self.proc is not None and self.proc.stdout is not None
        t0 = time.time()
        mtp: dict = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise EngineGone("the engine closed its stdout (see its stderr in this terminal)")
            if line.startswith("T "):
                yield ("tok", int(line[2:]))
            elif line.startswith("MTP "):
                # the REPL's per-turn line: MTP <rounds> <accepted> <drafted> <ms>
                f = line.split()
                if len(f) >= 5:
                    mtp = mtp_info(int(f[1]), int(f[2]), int(f[3]), float(f[4]))
            elif line.startswith("speculation"):
                # The prompt's own answer carries the pre-REPL summary table instead, since it is generated
                # before the loop and never sees an `MTP` line.  It is the first turn's only source.
                m = re.search(r"(\d+) rounds of \d+, drafts accepted (\d+) of (\d+) "
                              r"\([\d.]+\), ([\d.]+) tokens per round", line)
                if m:
                    mtp = mtp_info(int(m.group(1)), int(m.group(2)), int(m.group(3)), None, float(m.group(4)))
            elif line.startswith("mtp "):
                m = re.search(r"([\d.]+) ms/round drafting", line)
                if m:
                    mtp["ms_round"] = float(m.group(1))
            elif line.startswith("DONE"):
                f = line.split()
                n = int(f[1]) if len(f) > 1 else 0
                finish = f[2] if len(f) > 2 else "?"
                self.finish = finish
                yield ("done", n, finish, (time.time() - t0) * 1000.0, mtp)
                return
            elif line.startswith("ERR"):
                raise RuntimeError(line.strip())

    def turn(self, message: str, cap: int, first: bool, emit):
        """Run one turn and push display events through `emit(dict)`.

        `first` says the engine was loaded with this message as its prompt (no GEN line: its answer is already
        on stdout); otherwise the message goes over as a delta on the state the last turn left - unless a New
        chat reset that state, in which case it goes as a whole prompt.
        """
        with self.lock:
            if first:
                emit({"type": "status", "text": "loading the model (~30 s cold) ..."})
                self.start(message, cap)
            else:
                if not self.started or self.proc is None:
                    raise RuntimeError("no engine")
                if self.fresh:
                    emit({"type": "status", "text": "starting a new conversation ..."})
                    self._gen_line(encode(render_prompt(self.system, message, self.thinking)), cap,
                                   prompt_only=True)
                    self.fresh = False
                else:
                    emit({"type": "status", "text": "reading your message ..."})
                    self._gen_line(encode(render_turn(message, self.thinking)), cap)
            piece = TokenPiece(TOK, SPECIAL, self.thinking)
            n, ms, finish = 0, 0.0, "stop"
            mtp: dict = {}
            for ev in self.stream(cap):
                if ev[0] == "tok":
                    text, reasoning = piece.feed(ev[1])
                    n += 1
                    if text:
                        emit({"type": "text", "text": text, "r": 1 if reasoning else 0, "n": n})
                else:
                    _, n, finish, ms, mtp = ev
                    break
            tail = piece.flush()
            if tail:
                emit({"type": "text", "text": tail, "r": 1 if piece.reasoning else 0, "n": n})
            emit({"type": "done", "n": n, "finish": finish, "ms": ms,
                  "tps": (1000.0 * n / ms) if ms > 0 else 0.0, "mtp": mtp or None})


# --------------------------------------------------------------------------------- tokenizer glue
TOK: T.Tokenizer | None = None
SPECIAL_IDS: dict[str, int] = {}


def encode(text: str) -> list[int]:
    assert TOK is not None
    return TOK.encode(text, parse_special=True)


def load_tokenizer(pack: str) -> None:
    global TOK, SPECIAL_IDS
    TOK = T.Tokenizer.from_dir(os.path.join(pack, "tokenizer"))
    SPECIAL_IDS = {}
    for k, lit in SPECIAL.items():
        ids = TOK.encode(lit, parse_special=True)
        if len(ids) != 1:
            raise SystemExit("tokenizer: %r encodes to %d tokens, expected 1" % (lit, len(ids)))
        SPECIAL_IDS[k] = ids[0]


# --------------------------------------------------------------------------------- the page
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwopus - Mantiq</title>
<style>
  :root { --bg:#101216; --panel:#171a20; --line:#262b34; --fg:#e7eaf0; --dim:#8b93a3; --accent:#5b8cff; }
  * { box-sizing:border-box; }
  html,body { height:100%; margin:0; }
  body { background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         display:flex; flex-direction:column; }
  header { display:flex; align-items:center; justify-content:space-between; gap:12px;
           padding:10px 16px; border-bottom:1px solid var(--line); background:var(--panel); }
  .brand { display:flex; align-items:baseline; gap:8px; min-width:0; }
  .brand b { font-size:15px; }
  .brand .sub { color:var(--dim); font-size:12.5px; }
  .dot { width:8px; height:8px; border-radius:50%; background:#4b5563; display:inline-block; align-self:center; }
  .dot.on { background:#2fbf71; } .dot.busy { background:#e0a33e; } .dot.err { background:#e0574c; }
  .stats { display:flex; align-items:center; gap:12px; color:var(--dim); font-size:12.5px; }
  .stats .speed { color:var(--fg); font-variant-numeric:tabular-nums; }
  .stats .mtp { color:var(--accent); font-variant-numeric:tabular-nums; }
  button { background:#232833; color:var(--fg); border:1px solid var(--line); border-radius:8px;
           padding:7px 12px; font-size:13px; cursor:pointer; }
  button:hover { background:#2a3140; }
  button:disabled { opacity:.45; cursor:default; }
  main { flex:1; overflow-y:auto; padding:22px 0 8px; }
  .wrap { max-width:820px; margin:0 auto; padding:0 16px; }
  .msg { margin:14px 0; }
  .who { font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--dim); margin-bottom:5px; }
  .bubble { white-space:pre-wrap; word-wrap:break-word; }
  .user .bubble { background:#1b2130; border:1px solid var(--line); border-radius:12px; padding:10px 13px; }
  .reason { color:var(--dim); font-style:italic; border-left:2px solid var(--line);
            margin:0 0 10px; padding:2px 0 2px 12px; white-space:pre-wrap; }
  .meta { color:var(--dim); font-size:12px; margin-top:8px; font-variant-numeric:tabular-nums; }
  .err { color:#ff8b80; }
  footer { border-top:1px solid var(--line); background:var(--panel); padding:12px 16px; }
  .compose { max-width:820px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:#12151b; color:var(--fg); border:1px solid var(--line);
             border-radius:10px; padding:11px 13px; font:inherit; max-height:200px; outline:none; }
  textarea:focus { border-color:#37405a; }
  .hint { max-width:820px; margin:7px auto 0; color:var(--dim); font-size:11.5px; }
</style>
</head>
<body>
<header>
  <div class="brand"><span class="dot" id="dot"></span><b>Qwopus3.6-35B-A3B</b>
    <span class="sub">Mantiq &middot; qwen35moe</span></div>
  <div class="stats"><span class="mtp" id="mtp"></span><span id="state">idle</span><span class="speed" id="speed"></span>
    <button id="reset">New chat</button></div>
</header>
<main><div class="wrap" id="chat"></div></main>
<footer>
  <div class="compose">
    <textarea id="box" rows="1" placeholder="Message...  (Enter to send, Shift+Enter for a new line)"></textarea>
    <button id="send">Send</button>
  </div>
  <div class="hint">First message loads the model (~30 s). <span id="ctx"></span></div>
</footer>
<script>
const chat=document.getElementById('chat'), box=document.getElementById('box'),
      main=document.querySelector('main'),
      sendBtn=document.getElementById('send'), dot=document.getElementById('dot'),
      stateEl=document.getElementById('state'), speedEl=document.getElementById('speed'),
      mtpEl=document.getElementById('mtp'),
      resetBtn=document.getElementById('reset'), ctxEl=document.getElementById('ctx');
let busy=false, autoScroll=true;

main.addEventListener('scroll',()=>{
  autoScroll=(main.scrollHeight-main.scrollTop-main.clientHeight)<40; });

function scroll(){ if(autoScroll) main.scrollTop=main.scrollHeight; }
function state(s,cls){ stateEl.textContent=s; dot.className='dot '+(cls||''); }

function addUser(text){
  const d=document.createElement('div'); d.className='msg user';
  d.innerHTML='<div class="who">You</div>'; const b=document.createElement('div');
  b.className='bubble'; b.textContent=text; d.appendChild(b);
  chat.appendChild(d); scroll();
}
function addAssistant(){
  const d=document.createElement('div'); d.className='msg bot';
  d.innerHTML='<div class="who">Qwopus</div>';
  const reason=document.createElement('div'); reason.className='reason'; reason.style.display='none';
  const bubble=document.createElement('div'); bubble.className='bubble';
  const meta=document.createElement('div'); meta.className='meta';
  d.append(reason,bubble,meta); chat.appendChild(d); scroll();
  return {reason,bubble,meta};
}
function clearChat(){ chat.innerHTML=''; }

async function send(){
  const text=box.value.trim(); if(!text||busy) return;
  box.value=''; box.style.height='auto'; addUser(text);
  const a=addAssistant();
  busy=true; sendBtn.disabled=true; state('starting','busy');
  let t0=performance.now(), last=0;
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
                                       body:JSON.stringify({message:text})});
    if(!res.ok){ a.meta.innerHTML='<span class="err">'+(await res.text())+'</span>';
                 state('error','err'); return; }
    const reader=res.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {value,done}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let i;
      while((i=buf.indexOf('\n\n'))>=0){
        const raw=buf.slice(0,i); buf=buf.slice(i+2);
        for(const line of raw.split('\n')){
          if(!line.startsWith('data:')) continue;
          let e; try{ e=JSON.parse(line.slice(5).trim()); }catch(_){ continue; }
          onEvent(e,a,t0,()=>last);
          if(e.type==='text') last=e.n;
        }
      }
    }
  }catch(err){ a.meta.innerHTML='<span class="err">'+err.message+'</span>'; state('error','err'); }
  finally{ busy=false; sendBtn.disabled=false; box.focus(); }
}

function onEvent(e,a,t0,lastFn){
  if(e.type==='status'){ state(e.text,'busy'); }
  else if(e.type==='text'){
    const box_=e.r? a.reason : a.bubble;
    if(e.r) a.reason.style.display='block';
    box_.textContent+=e.text;
    const el=(performance.now()-t0)/1000;
    if(lastFn()>8 || el>1.5) speedEl.textContent=lastFn()+' tok · '+(lastFn()/el).toFixed(1)+' tok/s';
    scroll();
  }
  else if(e.type==='done'){
    const bits=[e.n+' tokens', e.finish];
    if(e.ms>0) bits.push((e.ms/1000).toFixed(2)+' s', e.tps.toFixed(1)+' tok/s');
    if(e.finish==='length') bits.push('(hit the cap - the answer is cut)');
    a.meta.textContent=bits.join(' · ');
    speedEl.textContent=e.tps.toFixed(1)+' tok/s';
    mtpEl.textContent = !e.mtp ? 'MTP off'
      : 'MTP '+e.mtp.ok+'/'+e.mtp.drafted+' accepted · '+e.mtp.tpr.toFixed(2)+' tok/round · '
        +e.mtp.ms_round.toFixed(2)+' ms/round';
    state('ready','on'); scroll();
  }
  else if(e.type==='error'){
    a.meta.innerHTML='<span class="err">'+e.message+'</span>'; state('error','err'); scroll();
  }
}

resetBtn.onclick=async()=>{
  if(busy) return;
  resetBtn.disabled=true; state('resetting','busy');
  try{ await fetch('/api/reset',{method:'POST'}); }catch(_){}
  clearChat(); speedEl.textContent=''; mtpEl.textContent=''; resetBtn.disabled=false; state('idle','');
  box.focus();
};

box.addEventListener('input',()=>{ box.style.height='auto'; box.style.height=Math.min(box.scrollHeight,200)+'px'; });
box.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); send(); } });
sendBtn.onclick=send;

(async()=>{ try{ const s=await (await fetch('/api/status')).json();
  if(s.ctx) ctxEl.textContent='context '+s.ctx+' tokens'; }catch(_){} box.focus(); })();
</script>
</body>
</html>
"""

# --------------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "mntq-qwen35moe/1.0"
    engine: Engine

    def log_message(self, fmt, *args):        # keep the console for the engine's own output
        pass

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            s = self.engine
            body = json.dumps({"state": "ready" if (s.started and s.proc) else "idle",
                               "ctx": s.max_context, "model": "Qwopus3.6-35B-A3B"}).encode()
            self._send(200, "application/json", body)
        elif self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        if self.path == "/api/reset":
            self.engine.reset_context()
            self._send(200, "application/json", b'{"ok":true}')
            return
        if self.path != "/api/chat":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            message = (req.get("message") or "").strip()
        except (ValueError, json.JSONDecodeError):
            self._send(400, "text/plain; charset=utf-8", b"bad request")
            return
        if not message:
            self._send(400, "text/plain; charset=utf-8", b"empty message")
            return
        cap = int(req.get("max_new") or self.engine.max_new)
        first = not self.engine.started

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(ev: dict):
            self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()

        try:
            self.engine.turn(message, cap, first, emit)
        except Exception as e:                                  # noqa: BLE001 - one bad turn must not kill the server
            try:
                emit({"type": "error", "message": str(e)})
            except Exception:
                pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    ap = argparse.ArgumentParser(description="a browser chat over mntq --repl (qwen35moe / Qwopus)")
    ap.add_argument("--pack", default=DEFAULT_PACK)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--engine", default=os.path.join(ROOT, "build", "mntq.exe"))
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--mtp", default=DEFAULT_MTP,
                    help="MTP draft layer dir (from tools/mtp_from_gguf.py); absent disables drafting")
    ap.add_argument("--system", default="", help="system message; empty means none")
    ap.add_argument("--prompt", default=None, help="preload: load the engine and answer this first")
    ap.add_argument("--max-new", type=int, default=1024, help="token cap per turn (default 1024)")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--spec", type=int, default=4)
    ap.add_argument("--no-thinking", action="store_true", help="open each turn with an empty reasoning block")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    a = ap.parse_args()

    if not os.path.exists(a.engine):
        raise SystemExit("no engine at %s (build it first)" % a.engine)
    if not os.path.isdir(os.path.join(a.pack, "tokenizer")):
        raise SystemExit("no tokenizer under %s" % a.pack)

    load_tokenizer(a.pack)
    eng = Engine(a)
    Handler.engine = eng

    srv = Server((a.host, a.port), Handler)
    url = "http://%s:%d/" % (a.host, a.port)
    print("mntq chat (qwen35moe): %s  -  pack %s" % (url, a.pack), flush=True)
    print("Ctrl+C ends the server and the engine with it.", flush=True)

    if a.prompt:                    # preload so the browser opens on a loaded engine
        def preload():
            def show(ev):
                if ev.get("type") == "text":
                    sys.stdout.write(ev["text"])
                    sys.stdout.flush()
                elif ev.get("type") == "done":
                    sys.stdout.write("\n[preloaded: %d tokens, %s, %.1f tok/s]\n"
                                     % (ev["n"], ev["finish"], ev["tps"]))
                    sys.stdout.flush()
            try:
                print("preloading: the first answer streams below ...", flush=True)
                eng.turn(a.prompt, a.max_new, True, show)
                print("the engine is ready.", flush=True)
            except Exception as e:                          # noqa: BLE001
                print("preload failed: %s" % e, flush=True)
        threading.Thread(target=preload, daemon=True).start()

    if not a.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping ...", flush=True)
    finally:
        eng.close()
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
