"""How the setup's screen (setup_tui.py) looks: binsider's layout (github.com/orhun/binsider) in the dashboard's
colours (serve/web/tokens.css, dark) - the palette, the logo, the pieces of text the screen draws, its layout, the
Ctrl+C question and the output box.  Nothing here runs a command or waits for the setup.
"""
from __future__ import annotations

import re
import time

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import OptionList, RichLog, Static

STEP_LINE = re.compile(r"=== Step (\d+): (.*) ===$")   # (Bridge.step's line)
# the dashboard's dark theme (serve/web/tokens.css); EDGE: the outer boxes, a step brighter than its card lines
BG, SURFACE3, LINE, EDGE = "#090a14", "#1e2240", "#252a4a", "#3d4373"
INK, INK_SOFT, MUTED, FAINT = "#eceefe", "#c3c7e6", "#8a8fb4", "#555b88"
ACCENT, ACCENT2, ACCENT_TEXT, INFO_TEXT = "#8b7cff", "#38bdf8", "#b4aaff", "#7dd3fc"
OK, OK_TEXT, WARN, DANGER, TINT = "#34d399", "#6ee7b7", "#fbbf24", "#fb7185", "#1d1b37"   # TINT: --st-accent-tint
THEME = Theme(name="mntq", primary=ACCENT, secondary=ACCENT2, accent=ACCENT, foreground=INK, background=BG,
              surface="#11132a", panel="#171a34", success=OK, warning=WARN, error=DANGER, dark=True)
# "MANTIQ" in font8x8 (public domain: the IBM PC BIOS font), 2 x 2 of its pixels to a character cell, as binsider's
FONT = {"M": (0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00), "A": (0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00),
        "N": (0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00), "T": (0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00),
        "I": (0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00), "Q": (0x3C, 0x66, 0x66, 0x66, 0x6E, 0x3C, 0x0E, 0x00)}
QUADRANTS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"


def big_rows(word: str) -> list:
    """`word` in FONT: its rows, each a (quadrant character, the letter's index) a cell - the blank ones left out, all
    as wide as the widest."""
    px = [[FONT[c][r] >> b & 1 for c in word for b in range(8)] for r in range(8)]
    rows = [[(QUADRANTS[px[r][x] | px[r][x + 1] << 1 | px[r + 1][x] << 2 | px[r + 1][x + 1] << 3], x // 8)
             for x in range(0, len(px[0]), 2)] for r in range(0, 8, 2)]
    width = max(len("".join(ch for ch, _ in row).rstrip()) for row in rows)
    return [row[:width] for row in rows if "".join(ch for ch, _ in row[:width]).strip()]


def mix(a: str, b: str, t: float) -> str:
    """The colour `t` of the way from `a` to `b`."""
    pa, pb = ([int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))


def gradient(s: str, style: str = "") -> Text:
    """`s` in the dashboard's brand gradient (violet to sky, as its VRAM tier bar), column by column."""
    return Text.assemble(*((ch, f"{style} {mix(ACCENT, ACCENT2, i / max(1, len(s) - 1))}") for i, ch in enumerate(s)))


def wordmark(word: str, hot: str, hot_style: str) -> list:
    """`word` in big, in the brand gradient - the letters in `hot` (a letter each) in `hot_style` instead."""
    return [Text.assemble(*((ch, f"bold {hot_style}" if word[i] in hot else
                             f"bold {mix(ACCENT, ACCENT2, x / max(1, len(row) - 1))}")
                            for x, (ch, i) in enumerate(row))) for row in big_rows(word)]


def bar(frac: float, width: int = 24) -> Text:
    done = round(min(max(frac, 0.0), 1.0) * width)
    return Text.assemble(gradient("━" * width)[:done], ("━" * (width - done), SURFACE3))


def sweep(width: int = 24, offset: int = 0, span: int = 12) -> Text:
    """An indeterminate bar: a lit block of `span` cells travels the track and wraps around - for a start whose
    engine log says nothing the screen can count (the load is real, its total is not known)."""
    track = max(width, span + 1)
    head = offset % (track + span) - span               # (the block comes in on the left and leaves on the right)
    t = Text()
    for x in range(track):
        k = x - head
        t.append("━", f"bold {mix(ACCENT, ACCENT2, k / max(1, span - 1))}" if 0 <= k < span else SURFACE3)
    return t


def title(text: str, style: str = f"bold {INK}") -> Text:
    """A box title the way binsider sets it into the border: │text│."""
    return Text.assemble(("│", EDGE), (text, style), ("│", EDGE))


def hints(*keys) -> Text:
    """The keys for the bottom border: [Key→ what it does] ..."""
    t = Text()
    for i, (key, what) in enumerate(keys):
        t.append_text(Text.assemble((" [" if i else "[", FAINT), (key, INFO_TEXT), ("→ ", FAINT), (what, INK),
                                    ("]", FAINT)))
    return t


def splash() -> Text:
    tagline = Text.assemble(("Run Qwen, DeepSeek and GLM models ", INK),
                            ("small or large, at full speed.", f"italic {ACCENT_TEXT}"))
    lines = wordmark("MANTIQ", "AI", OK) + [
        tagline, Text("─" * tagline.cell_len, LINE), Text("https://github.com/dahanco/mantiq", f"italic {INK_SOFT}")]
    return Text("\n", justify="center").join(lines)


def clock(s: float) -> str:
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s" if s < 3600 else f"{s // 3600}h {s % 3600 // 60:02d}m"


def styled(kind: str, text: str) -> Text:
    """A line of the output box from a Bridge entry (the text the plain setup prints), or the server's ("live")."""
    if kind == "step":
        name = STEP_LINE.match(text).group(2)
        return Text(f"── {name[:1].upper() + name[1:]}", f"bold {ACCENT_TEXT}")
    line = text[2:] if text.startswith("  ") else text
    if kind == "ok":                                    # "[ok] ..."
        return Text.assemble(("✓ ", OK), (line[5:], INK))
    if kind == "warn":                                  # "[!]  ..."
        return Text.assemble(("! ", f"bold {WARN}"), (line[5:], WARN))
    if kind == "answer":
        return Text.assemble(("› ", ACCENT_TEXT), (line, ACCENT_TEXT))
    if kind == "out":
        return Text.from_ansi(text, style=MUTED)
    if kind == "live":                                  # Mantiq's server: "ready:", warnings, errors stand out
        return Text.from_ansi(text, style=f"bold {OK}" if text.startswith("ready:") else WARN if "WARNING" in text
                              else DANGER if re.search(r"\berror\b|Traceback", text, re.I) else INK_SOFT)
    if line.startswith("> "):                           # a command as it runs
        return Text(line, f"italic {FAINT}")
    return Text.from_ansi(line, style=DANGER if line.lstrip().startswith("[X]") else INK_SOFT)


def card(rows: list) -> Text:
    """The card's lines: Label: value (binsider's info box)."""
    return Text("\n").join(Text.assemble((f"{k}: ", ACCENT_TEXT), v) for k, v in rows)


class Card(Vertical):
    """The card on the screen: a line each, drawn again only when it changes - a terminal underlines a URL it finds,
    and the dashboard's and the API's, written anew with the spinner's line at every tick, flashed."""
    DEFAULT_CSS = """
    Card { height: auto; }
    Card > Static { height: auto; text-wrap: nowrap; text-overflow: ellipsis; }
    """
    LINES = 6

    def compose(self) -> ComposeResult:
        self.shown: list = [None] * self.LINES
        for _ in range(self.LINES):
            row = Static()
            row.display = False                         # (until it has a line)
            row.auto_links = False                      # (no links: else the mouse over a line draws it again)
            yield row

    def show(self, text: Text) -> None:
        lines = (list(text.split("\n")) + [None] * self.LINES)[:self.LINES]
        for k, (row, line) in enumerate(zip(self.query(Static), lines)):
            if line != self.shown[k]:
                came_or_went = (line is None) != (self.shown[k] is None)
                self.shown[k] = line
                row.display = line is not None
                # (a line replaced by another is one row either way: no new layout - at the spinner's and the clock's
                # pace that was most of an idle screen's CPU)
                row.update(line or "", layout=came_or_went)

    @property
    def plain(self) -> str:
        return "\n".join(line.plain for line in self.shown if line is not None)


def setup_card(app, spin: str) -> Text:
    """The card while the setup runs (setup_tui.SetupApp): its step, what runs now, the progress - or, for a
    start without a setup, what runs before Mantiq does (a compile after an update)."""
    n = app.cur
    name = app.titles.get(n, "Starting" if app.steps else "Starting Mantiq")
    if app.pending is not None:
        stopped = "Mantiq stopped" if app.serving is not None else "The setup stopped"
        now = Text(f"{stopped} (below)", f"bold {DANGER}") if app.pending[2] == "fail" else \
            Text("Waiting for your answer (below)", f"bold {WARN}")
    else:
        now = Text.assemble((spin + " ", ACCENT), (app.doing, INK) if app.doing else (app.last, MUTED))
    if app.bar:
        progress = Text.assemble(bar(app.bar[0]), (f" {app.bar[0] * 100:3.0f}%  ", INK), (app.bar[1], MUTED))
    elif app.steps:
        done = sum(app.state[i] in ("done", "warn") for i in range(1, 9))
        progress = Text.assemble(bar(done / 8), (f"  {done} of 8 steps done", MUTED))
    else:
        progress = Text("–", MUTED)
    return card([("Step", Text(f"{n} of 8 · {name}" if 0 < n < 9 else name, INK)), ("Now", now),
                 ("Progress", progress)])


def serving_card(s: dict, spin: str) -> Text:
    """The card while Mantiq runs on the screen: `s` is what the screen knows of its server (setup_tui, ev_serve)."""
    took = clock(time.monotonic() - s["since"])
    if s["state"] == "ready":
        status = Text.assemble(("● ", OK), ("Ready", f"bold {OK}"), (f" for {took}", MUTED),
                               (f" · last answer {s['speed']}", MUTED) if s["speed"] else "")
    elif s["state"].startswith("Stopped"):
        status = Text(s["state"], f"bold {DANGER}")
    else:
        status = Text.assemble((spin + " ", ACCENT), (s["state"], INK), (f" · {took}", MUTED))
    model = " · ".join(x for x in (s.get("label") or s["model"], None if s.get("label") else s.get("quant"),
                                   s.get("context") and f"{int(s['context']) // 1024}K context") if x)
    rows = [("Model", Text(model, INK)), ("Dashboard", Text(s["dashboard"], f"bold {INFO_TEXT}")),
            ("API", Text.assemble((s["api"], INK), (" (OpenAI) · ", MUTED), ("/v1/messages", INK),
                                  (" (Anthropic)", MUTED))), ("Status", status)]
    return card(rows)


# the screen's layout: the top bar (tabs), the main box (logo, card, output, a question), the keys in its border
SETUP_CSS = """
    Screen { background: $background; }
    #top { height: 3; border: solid $mntq-edge; border-title-align: center; padding: 0 1; }
    #tabs { width: 1fr; text-wrap: nowrap; text-overflow: ellipsis; }
    #where { width: auto; max-width: 40%; padding-left: 2; color: $mntq-muted; text-style: italic; text-wrap: nowrap; }
    .narrow #where { display: none; }
    #main { height: 1fr; border: solid $mntq-edge; border-subtitle-align: center; padding: 1 2 0 2; }
    #page { height: 1fr; }
    #splash { width: 100%; height: auto; margin-bottom: 1; text-align: center; }
    .short #splash, .tight.asking #splash, .short.asking #card { display: none; }
    #lower { height: 1fr; align-horizontal: center; }
    #card { width: 76; max-width: 100%; height: auto; border: solid $mntq-line; padding: 0 1; margin-bottom: 1; }
    #output { width: 100%; max-width: 120; height: 1fr; min-height: 3; border: solid $mntq-line; padding: 0 1;
              border-title-align: center; border-subtitle-align: right; background: $background;
              scrollbar-size-vertical: 1; scrollbar-color: $mntq-edge; scrollbar-background: $background; }
    #ask { width: 100%; max-width: 120; height: auto; max-height: 75%; margin-top: 1; border: solid $mntq-edge;
           border-title-align: center; padding: 0 1; display: none; overflow-y: auto; scrollbar-size-vertical: 1;
           scrollbar-color: $mntq-edge; scrollbar-background: $background; }
    #ask.on { display: block; }
    #ask.fail { border: solid $error; }
    #intro { margin-bottom: 1; }
    #options, #options:focus { height: auto; max-height: 12; border: none; padding: 0; background: $background;
                               background-tint: $background 0%; }
    #options > .option-list--option-highlighted,
    #options:focus > .option-list--option-highlighted { background: $mntq-tint; color: $mntq-accent-text;
                                                        text-style: bold; }
    #about { margin-top: 1; color: $mntq-muted; text-style: italic; }
"""
# Ctrl+C's question, over the screen
STOP_CSS = """
    StopScreen { align: center middle; background: $background 60%; }
    #box { width: 64; height: auto; border: solid $mntq-edge; border-title-align: center; border-subtitle-align: center;
           background: $background; padding: 1 2; }
    #box OptionList, #box OptionList:focus { height: auto; margin-top: 1; border: none; padding: 0;
                                             background: $background; background-tint: $background 0%; }
    #box OptionList > .option-list--option-highlighted,
    #box OptionList:focus > .option-list--option-highlighted { background: $mntq-tint; color: $mntq-accent-text;
                                                               text-style: bold; }
"""


class StopScreen(ModalScreen):
    """Ctrl+C: stop the setup - or Mantiq, once it runs?  (Ctrl+C again: yes.)"""

    CSS = STOP_CSS
    BINDINGS = [Binding("escape", "dismiss(False)", show=False)]

    def __init__(self, me: str, serving: bool):
        super().__init__()
        self.me, self.serving = me, serving

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static(Text(f"The dashboard and the API close, and the model leaves the GPUs. {self.me} starts it "
                              "again." if self.serving else f"Everything finished so far is kept: {self.me} continues "
                              "where it stopped. The command running now ends.", INK_SOFT))
            yield OptionList(*(("Keep running", "Stop Mantiq") if self.serving else ("Keep going", "Stop the setup")))

    def on_mount(self) -> None:
        box = self.query_one("#box")
        box.border_title = title("Stop Mantiq?" if self.serving else "Stop the setup?")
        box.border_subtitle = hints(("Enter", "Choose"), ("Esc", "Back"))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option_index == 1)


class Output(RichLog):
    """The output box: back at its last line when its size changes (a question opens or closes below it).  No links in
    it: the mouse over a line would draw it again (and a terminal that underlines URLs blinks them)."""

    def on_mount(self) -> None:
        self.auto_links = False

    def on_resize(self, event: events.Resize) -> None:
        super().on_resize(event)
        if getattr(self.app, "view", None) is None:
            self.scroll_end(animate=False)
