"""Adding a model of the user's own from the Models page (/api/models/add-local).  The file is chosen in the server's
own native dialog (tkinter: the desktop session this server runs in), then the setup's own flow (mntq.py
--add-local-model <path>: the pack, the MTP drafter, the config and the start script) runs as a child process whose
output is read.  One add at a time."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# The dialog run as the main thread of a child process: on macOS Tk must be built on the process's main thread, and
# building it in one of the server's threads hangs the whole server (the page's Add Local Model never gets an answer).
_TK_PICK = r'''
import sys
try:
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:  # noqa: BLE001 - not every window manager knows it
        pass
    sys.stdout.write(filedialog.askopenfilename(
        title="Choose a model file (GGUF)",
        filetypes=[("GGUF model", "*.gguf"), ("All files", "*.*")]) or "")
    root.update()
    root.destroy()
except BaseException as e:  # noqa: BLE001 - no tkinter, or no desktop: the page is told
    sys.stderr.write(str(e))
    sys.exit(1)
'''


def pick_gguf(timeout: float = 900.0) -> str | None:
    """The server's own file dialog to choose one .gguf file; the full path is returned, or None when it is cancelled.
    Tk is created, used and closed in one thread (the one that makes it is the one that closes it); on macOS that
    thread has to be the process's main one, so there the dialog runs in a child process of its own; a machine with no
    desktop to show it on raises ValueError, which the page reports."""
    if sys.platform == "darwin":
        try:
            r = subprocess.run([sys.executable, "-c", _TK_PICK], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ValueError("the file dialog was left open too long")
        if r.returncode != 0:
            raise ValueError(f"the file dialog could not be shown: {r.stderr.strip() or 'it could not be started'}")
        return r.stdout.strip() or None
    out: dict = {}

    def pick():
        try:
            import tkinter
            from tkinter import filedialog
            root = tkinter.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:  # noqa: BLE001 - not every window manager knows it
                pass
            out["path"] = filedialog.askopenfilename(
                title="Choose a model file (GGUF)",
                filetypes=[("GGUF model", "*.gguf"), ("All files", "*.*")]) or None
            root.update()
            root.destroy()
        except Exception as e:  # noqa: BLE001 - no tkinter, or no desktop: the page is told
            out["error"] = str(e)

    t = threading.Thread(target=pick, daemon=True)
    t.start()
    t.join(timeout)
    if "error" in out:
        raise ValueError(f"the file dialog could not be shown: {out['error']}")
    return out.get("path")


def backend() -> str:
    """The engine backend this install runs: MANTIQ_BACKEND, which mntq.py puts in its server's environment (the CPU
    engine and Metal have their own, not build/mntq), else CUDA - the setup's own default when nothing is set up yet."""
    b = (os.environ.get("MANTIQ_BACKEND") or "").strip().lower()
    return b if b in ("cuda", "hip", "metal", "cpu") else "cuda"


def add_command(path: Path) -> list:
    """The setup run that adds the model file: its pack, its MTP drafter and its config (mntq.py --add-local-model)."""
    cmd = [sys.executable, str(ROOT / "mntq.py"), "--add-local-model", str(path), "--yes", "--no-start"]
    if backend() != "cuda":                            # (its engine is not build/mntq: add it to this install's own)
        cmd += ["--backend", backend()]
    return cmd


def setup_needed() -> bool:
    """Whether this PC has not been set up yet: llama.cpp's source (the pack builder's gguf-py, tools/_paths.py) and
    the engine mntq.py compiles (build/mntq, build-hip/mntq, build-metal/mntq-metal, build-cpu/mntq-cpu).  A fresh
    download has neither, and this page's "Add Local Model" is how its first model gets in: mntq.py's add_model_setup
    runs the setup's own steps 1, 3 and 4 inside that add, so the page shows them in a window of their own - the card
    below is about the model's pack."""
    if not (ROOT / "third_party" / "llama.cpp" / "gguf-py" / "gguf").is_dir():
        return True
    b = backend()
    if b == "cpu":
        return not any((ROOT / "build-cpu" / n).exists() for n in ("mntq-cpu.exe", "mntq-cpu"))
    if b == "metal":
        return not (ROOT / "build-metal" / ("mntq-metal.exe" if os.name == "nt" else "mntq-metal")).exists()
    names = ("mntq.exe", "mntq") if os.name == "nt" else ("mntq",)
    if any((ROOT / d / n).exists() for d in ("build", "build-hip") for n in names):
        return False
    return not (ROOT / "build-metal" / ("mntq-metal.exe" if os.name == "nt" else "mntq-metal")).exists()


class Add:
    """One model being added: its child process, its state and the last lines of the setup's output."""

    def __init__(self, path: Path, command: list, cwd: Path = ROOT):
        self.path = str(path)
        self.name = path.name
        self.started = time.time()
        self.setup = setup_needed()                    # this PC is not set up: the add runs the setup's own steps
        self.state = "starting"                        # starting, working, success, error, cancelled
        self.error = None
        self.log: list = []
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._read, daemon=True).start()

    def _line(self, text: str):
        text = text.strip()
        if not text:
            return
        with self._lock:
            self.log = (self.log + [text])[-60:]
            if self.state == "starting":
                self.state = "working"

    def _read(self):
        buf = b""
        while True:
            b = self.proc.stdout.read(1)               # (a progress line rewrites itself with \r)
            if not b:
                break
            if b in (b"\r", b"\n"):
                self._line(buf.decode("utf-8", "replace"))
                buf = b""
            else:
                buf += b
        self._line(buf.decode("utf-8", "replace"))
        rc = self.proc.wait()
        with self._lock:
            if self.state == "cancelled":
                return
            if rc == 0:
                self.state = "success"
            else:
                self.state = "error"
                self.error = next((ln for ln in reversed(self.log) if ln.strip()), f"the setup stopped (code {rc})")

    def cancel(self):
        with self._lock:
            if self.state in ("success", "error", "cancelled"):
                return
            self.state = "cancelled"
        self.proc.terminate()

    def snapshot(self) -> dict:
        with self._lock:
            return {"file": self.name, "path": self.path, "state": self.state, "setup": self.setup, "error": self.error,
                    "started": self.started, "message": self.log[-1] if self.log else "",
                    "log": list(self.log[-12:])}

    def active(self) -> bool:
        return self.state in ("starting", "working")


class Adder:
    """The server's one model-add slot."""

    def __init__(self, root: Path, command=None):
        self.root = Path(root)
        self.command = command                         # tests: a function (path) -> argv
        self.current: Add | None = None
        self._lock = threading.Lock()

    def start(self, path) -> Add:
        p = Path(str(path or "")).expanduser()
        if not p.is_file():
            raise ValueError(f"{p} is not a file")
        if p.suffix.lower() != ".gguf":
            raise ValueError(f"{p.name} is not a .gguf file")
        with self._lock:
            if self.current is not None and self.current.active():
                raise ValueError(f"{self.current.name} is being added; wait for it")
            self.current = Add(p, (self.command or add_command)(p))
            return self.current

    def snapshot(self) -> dict | None:
        return self.current.snapshot() if self.current is not None else None

    def busy(self) -> bool:
        return self.current is not None and self.current.active()

    def cancel(self) -> bool:
        if self.busy():
            self.current.cancel()
            return True
        return False
