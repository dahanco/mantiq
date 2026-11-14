"""tools/mntq_quants.py - the quants' names users see (published under Maya): the model first, then the tier by its routed experts'
bits, the same meaning for every model - XS ~1.6-1.7, S ~2-2.5, M ~3-3.5, L ~4+ (a rebuild: -v2, -v3).

    on screen       "MiniMax M3.1 · Maya-XS"                        shown(family, key), config_label(cfg)
    setup ids       --model minimax-mntq-xs                         resolve(name)
    files           <Model>-Maya-<Tier>-<maintype>-0000N-of-0000M.gguf, one repo per model (<Model>-Maya-GGUF)
    configs         mntq-minimax-mntq-xs.json                       config_stem(family, key)

Every name a quant had before stays an alias, so nothing installed or typed before breaks: a GLM download's name
(--model Mantiq-S24, Mantiq-S-v2-IQ2_XXS, EXL3-3.05bpw ...), --ds4-quant mntq-xs, --mm3-quant gguf-xs, and the installer
records of configs written before (their "quant" / "ds4_quant" / "mm3_quant" keys are what the setup still writes).
The files already published keep their names (their manifests list them).
"""
from __future__ import annotations

import re
from pathlib import Path

FAMILIES = ("glm", "deepseek", "minimax", "qwen")
# ("qwen": CYBER-FROST-3.8-PS, the Qwen3.8-Flash-Next fine-tune - qwen4exp, on the `mntq` engine's own Qwen path)
MODEL = {"glm": "GLM-5.3-Flash", "deepseek": "DeepSeek-V4.1-Flash-1017", "minimax": "MiniMax-M3.1-Flash",
         "qwen": "CYBER-FROST-3.8-PS"}
SHOWN = {"glm": "GLM-5.3", "deepseek": "DeepSeek V4.1", "minimax": "MiniMax M3.1",
         "qwen": "Qwen-3.8-Flash-Next (CYBER-FROST)"}   # the model's name on screen (a qwen config whose GGUF names a
                                                         # fine-tune is shown by that fine-tune instead: config_label);
                                                         # the chat header's short form is serve/web/lib.js's
                                                         # modelName(id, true), "Qwen-3.8-Flash-Next"; its ids stay
SEP = " · "

# Each model's own defaults - what its config gets (the setup, the dashboard's switch to it) and what a config without
# them runs with: its sampling as its developer publishes it (no top-k, min-p or penalties unless they name one) and the
# thinking level a request without one gets.  Every quant of a model the same (Maya's, unsloth's, orcarouter's, EXL3).
#   GLM-5.3-Flash: zai-org/GLM-5.3-Flash generation_config.json - temperature 1.0, top_p 0.95; the level its card's
#     default, Max (the owner, 2026-10-27; it was Maya's High)
#   DeepSeek-V4.1-Flash-1017: deepseek-ai's card ("Recommended sampling parameters") - temperature 1.0, top_p 1.0;
#     the level its encoder's default, high (a reasoning effort of 75); orcarouter's card names none (its base's)
#   MiniMax-M3.1-Flash: MiniMaxAI's card and generation_config.json (54e4e69) - temperature 1.0, top_p 0.95; no
#     default effort named, so "auto" (the template's own: the model decides whether to think)
MODEL_DEFAULTS = {
    "glm": ({"temperature": 1.0, "top_p": 0.95}, "max"),
    "deepseek": ({"temperature": 1.0, "top_p": 1.0}, "high"),
    "minimax": ({"temperature": 1.0, "top_p": 0.95}, "auto"),
    # CYBER-FROST-3.8-PS: Blackfrost-AI/CYBER-FROST-3.8-BF16 generation_config.json and card - temperature 1.0, top_p
    # 0.95, top_k 20; thinking on, its template's default effort xhigh
    "qwen": ({"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "xhigh"),
}


# A quant with its own sampling (the owner validated it on that quant, 2026-11-02): GLM-5.3-Flash Maya-XS - temperature
# 0.7, top_p 0.95, min_p 0.05, no penalties.  Every other quant keeps its model's (MODEL_DEFAULTS).
QUANT_SAMPLING = {("glm", "Maya-XS"): {"temperature": 0.7, "top_p": 0.95, "min_p": 0.05}}


def model_defaults(cfg: dict) -> tuple:
    """(sampling, reasoning_effort): a config's model's own defaults (MODEL_DEFAULTS by its family, the quant's own
    sampling where it has one: QUANT_SAMPLING), copies."""
    fam = family_of(cfg)
    sampling, effort = MODEL_DEFAULTS.get(fam, MODEL_DEFAULTS["glm"])
    return dict(QUANT_SAMPLING.get((fam, tier_of(cfg)), sampling)), effort


# The thinking budget (the config's "thinking_budget": tokens, 0 = no limit; the reasoning ends there and the answer
# starts).  GLM-5.3-Flash Maya-XS thinks past 32K on a plain three.js prompt and never answers; 8192 gave complete,
# valid answers fastest.  Only the quants listed here get another default (the owner, 2026-11-02: Maya-XS only); every
# other quant and model keeps 32768.  Keyed by (family, tier): a quant's catalog entry in QUANTS (and mntq.MODELS)
# carries the tier, so adding the GLM Maya-XS entry there is all it needs to pick this up.
OLD_THINK_BUDGET = 32768
GLM_THINK_BUDGET = 8192
QUANT_THINK_BUDGET = {("glm", "Maya-XS"): GLM_THINK_BUDGET}


def tier_of(cfg: dict) -> str:
    """The tier ("Maya-XS") of the quant a config was set up with (installer.quant: a catalog key or a tier)."""
    fam, quant = family_of(cfg), str((cfg.get("installer") or {}).get("quant") or "")
    for f, key, tier, _ in QUANTS:
        if f == fam and quant in (key, tier):
            return tier
    return quant
THINK_BUDGET_MIN, THINK_BUDGET_MAX = 256, 262144


def default_thinking_budget(cfg: dict) -> int:
    """The thinking budget a config without one runs with: its quant's (QUANT_THINK_BUDGET), else 32768."""
    return QUANT_THINK_BUDGET.get((family_of(cfg), tier_of(cfg)), OLD_THINK_BUDGET)


def thinking_budget_for(cfg: dict) -> int | None:
    """The "thinking_budget" a setup or a switch writes into `cfg`, or None for 'leave it as it is'.  A value the user
    set (installer.thinking_budget_set, written by the dashboard's field) or any value but the old default stays; a
    config of a quant with its own default (QUANT_THINK_BUDGET) and none set, or still at the old 32768, gets it."""
    new = default_thinking_budget(cfg)
    if new == OLD_THINK_BUDGET or (cfg.get("installer") or {}).get("thinking_budget_set"):
        return None
    value = cfg.get("thinking_budget")
    return new if value is None or value == OLD_THINK_BUDGET else None


def check_thinking_budget(value) -> int:
    """A thinking budget from the dashboard: 0 (no limit) or 256..262144 tokens; ValueError otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ValueError("the thinking budget must be a whole number of tokens")
    value = int(value)
    if value != 0 and not THINK_BUDGET_MIN <= value <= THINK_BUDGET_MAX:
        raise ValueError(f"the thinking budget is 0 (no limit) or {THINK_BUDGET_MIN} to {THINK_BUDGET_MAX} tokens")
    return value

# Maya's own quants: (family, the setup's key for it, its tier, its setup id).  The key is what the setup and the
# configs' installer records use: a GLM download's MODELS key, a --ds4-quant / --mm3-quant value
QUANTS = (
    ("glm", "Mantiq-S-v2-IQ2_XXS", "Maya-S", "glm-mntq-s"),
    ("glm", "Mantiq-S24", "Maya-S24", "glm-mntq-s24"),
    ("glm", "Mantiq-XS-IQ1_S", "Maya-XS", "glm-mntq-xs"),
    ("glm", "Mantiq-M", "Maya-M", "glm-mntq-m"),
    ("glm", "Mantiq-M-Derisked", "Maya-M-Derisked", "glm-mntq-m-derisked"),
    ("glm", "Mantiq-L", "Maya-L", "glm-mntq-l"),
    ("deepseek", "mntq-xs", "Maya-XS", "deepseek-mntq-xs"),
    ("deepseek", "mntq-s", "Maya-S", "deepseek-mntq-s"),
    ("deepseek", "mntq-m", "Maya-M", "deepseek-mntq-m"),
    ("minimax", "gguf-xs", "Maya-XS", "minimax-mntq-xs"),
    ("minimax", "gguf-s", "Maya-S", "minimax-mntq-s"),
)
# the other files the setup offers, as shown after the model's name (a GLM download not listed: its own name); also
# the ones an install can still have though the list no longer offers them (DeepSeek's UD-Q4_K_XL: mntq.DS4_BY_HAND)
OTHERS = {("deepseek", "gguf-q8"): "UD-Q8_K_XL", ("deepseek", "gguf-q4"): "UD-Q4_K_XL",
          # (unsloth's smaller UD quants: mntq.DS4_UD, chosen by hand too)
          ("deepseek", "gguf-iq1_s"): "UD-IQ1_S", ("deepseek", "gguf-iq1_m"): "UD-IQ1_M",
          ("deepseek", "gguf-iq2_xxs"): "UD-IQ2_XXS", ("deepseek", "gguf-q2_k_xl"): "UD-Q2_K_XL",
          ("deepseek", "gguf-iq3_xxs"): "UD-IQ3_XXS", ("deepseek", "gguf-iq3_s"): "UD-IQ3_S",
          ("deepseek", "gguf-q3_k_xl"): "UD-Q3_K_XL",
          ("deepseek", "exl3"): "EXL3 2.9 bpw", ("deepseek", "exl3-3.5"): "EXL3 3.5 bpw",
          ("deepseek", "abliterated-128g"): "orcarouter Abliterated Mix-128G",
          ("deepseek", "abliterated-256g"): "orcarouter Abliterated Mix-256G",
          ("minimax", "exl3"): "EXL3 3.5 bpw", ("qwen", "q5_k_m"): "Q5_K_M",
          # the qwen family's small model, Qwen-3.6-35B-A3B-Escha-W2: the setup offers its first published quant
          ("qwen", "iq2_xxs"): "IQ2_XXS",
          # ... and its distillation, Qwen-3.8-35B-A3B (Distill): its one quant here (its id is its key - a single GGUF)
          ("qwen", "qwen38-distill"): "Q4_K_M"}
# CYBER-FROST-3.8-PS: its one GGUF (peasantsmith/CYBER-FROST-3.8-PS-GUFF) - the setup ids it is asked for by
QWEN_IDS = ("cyber-frost", "cyber-frost-3.8", "cyber-frost-3.8-ps", "cyber-frost-3.8-ps-q5_k_m", "cyber-frost-q5_k_m")
# The qwen family's small model: Qwen3.6-35B-A3B-Escha-W2, a Qwen3.5/3.6-MoE (qwen35moe) fine-tune in peasantsmith's
# GGUF (tools/qwen/download-small.json: 35B, 8 of 256 experts a token, Apache-2.0).  The qwen path serves it without
# qwen4exp's n-gram table (--ple-gguf: a qwen35moe file has none to open) and without the MTP drafter (its NextN block
# is not the shape this engine's drafter takes), so prompts are written whole (--prefill 0) and its drafts are prompt
# lookup alone.  Its installer record carries "qwen_model": QWEN_SMALL - what tells a config of it from the other two.
QWEN_SMALL = "escha-w2"                            # (its id: --model, the config's installer; never shown)
QWEN_SMALL_SHOWN = "Qwen-3.6-35B-A3B (Escha-W2)"   # (its name on screen)
# the setup ids it is asked for by (--model)
QWEN_SMALL_IDS = ("escha-w2", "escha", "escha-w2-iq2_xxs", "qwen3.6-35b-a3b-escha-w2", "qwen-3.6-35b-a3b-escha-w2")
# the quants its repo is published in; the setup offers the first, the smallest (the one tools/qwen/download-small.json
# pins) - and what its config's name and its quant on screen are keyed by
QWEN_SMALL_QUANTS = ("iq2_xxs", "iq3_xxs", "q3_k_m", "q4_k_m")
# The qwen family's third model, and the model list's first and a first setup's default: Qwen-3.8-35B-A3B (Distill),
# empero-ai's distillation of Qwen/Qwen3.8-35B-A3B in its own GGUF (tools/qwen/download-distill.json: 35B, 8 of 256
# experts a token, Apache-2.0).  A qwen35moe file exactly as Escha-W2's, on the same Qwen path (no --ple-gguf, no --mtp;
# prompts written whole, drafts the engine's prompt lookup).  Its installer record carries "qwen_model": QWEN_DISTILL -
# what tells its config from Escha-W2's and CYBER-FROST's.  It has one published quant here (its id is its key: the
# setup offers and downloads that file).
QWEN_DISTILL = "qwen38-distill"                    # (its id: --model, the config's installer; never shown)
QWEN_DISTILL_SHOWN = "Qwen-3.8-35B-A3B (Distill)"  # (its name on screen)
# the setup ids it is asked for by (--model)
QWEN_DISTILL_IDS = ("qwen38-distill", "qwen3.8-35b-a3b-distill", "qwen-3.8-35b-a3b-distill", "distill")
# --mm3-quant also takes the tier as DeepSeek's --ds4-quant does (mntq-xs), beside its first names (gguf-xs)
MM3_QUANT_ALIASES = {"mntq-xs": "gguf-xs", "mntq-s": "gguf-s"}


def tier(family: str, key: str | None) -> str | None:
    """A Maya quant's tier ("Maya-XS"), or None for any other files."""
    return next((t for f, k, t, _ in QUANTS if f == family and k == key), None)


def setup_id(family: str, key: str | None) -> str | None:
    """A Maya quant's setup id ("minimax-mntq-xs"), or None."""
    return next((i for f, k, _, i in QUANTS if f == family and k == key), None)


def full_name(family: str, key: str | None) -> str | None:
    """A Maya quant's full name, model first ("MiniMax-M3.1-Flash-Maya-XS"), or None."""
    t = tier(family, key)
    return f"{MODEL[family]}-{t}" if t else None


def shown(family: str, key: str | None = None, gb: float | None = None, name: str | None = None) -> str:
    """The files as users see them, the model's name first: "MiniMax M3.1 · Maya-XS", with gb "... (93 GB)"; never a
    bare tier.  Files that are not a Maya quant by their own name ("DeepSeek V4.1 · UD-Q8_K_XL"); no key: the model.
    `name` names the model instead of the family's own (the fine-tune a qwen config's GGUF is); a qwen key in
    QWEN_SMALL_QUANTS is the family's small model and QWEN_DISTILL its distillation, each shown by its own name (never
    CYBER-FROST's)."""
    if name is None and family == "qwen":
        if key in QWEN_SMALL_QUANTS:
            name = QWEN_SMALL_SHOWN
        elif key == QWEN_DISTILL:
            name = QWEN_DISTILL_SHOWN
    name = name or SHOWN.get(family, family)
    what = tier(family, key) or OTHERS.get((family, key)) or key
    if family == "qwen" and what:
        what = str(what).upper()                            # (a qwen config's quant, from its GGUF: Q4_K_M, not q4_k_m)
    return (name + SEP + what if what else name) + (f" ({gb:.0f} GB)" if gb else "")


def config_stem(family: str, key: str | None) -> str | None:
    """A Maya quant's config name under the scheme (mntq-minimax-mntq-xs), or None where the config keeps the name it
    always had (GLM's mntq-<download>.json, DeepSeek's mntq-ds4.json, MiniMax's EXL3 mntq-mm3.json - installs and
    their run scripts point at them).  MiniMax's GGUFs, new with the scheme, are the first named by it."""
    if family == "qwen":                               # (the family's three published models: one config each)
        if key == QWEN_DISTILL:
            return "mntq-qwen38-distill"
        return "mntq-escha-w2" if key in QWEN_SMALL_QUANTS else "mntq-cyber-frost"
    return f"mntq-{setup_id(family, key)}" if family == "minimax" and setup_id(family, key) else None


def resolve(name: str, glm_downloads=()) -> tuple | None:
    """(family, the setup's key) for a model named on the command line, or None for a name Maya does not know:
    a setup id (glm-mntq-s, deepseek-mntq-xs, minimax-mntq-xs ...) or a quant's full name (MiniMax-M3.1-Flash-Maya-XS),
    any case; a GLM download by its name before (glm_downloads: the setup's MODELS keys, Mantiq-S24 ...); or a model's
    own name (DeepSeek-V4.1-Flash-1017, MiniMax-M3.1-Flash: its files then asked, or --ds4-quant / --mm3-quant) -
    key None."""
    low = (name or "").strip().lower()
    for f, k, _, i in QUANTS:
        if low in (i, full_name(f, k).lower()):
            return f, k
    for k in glm_downloads:
        if low == k.lower():
            return "glm", k
    if low in QWEN_IDS:
        return "qwen", "q5_k_m"
    if low in QWEN_DISTILL_IDS or low == QWEN_DISTILL_SHOWN.lower():
        return "qwen", QWEN_DISTILL
    if low in QWEN_SMALL_IDS or low == QWEN_SMALL_SHOWN.lower():
        return "qwen", QWEN_SMALL_QUANTS[0]
    for f in ("deepseek", "minimax", "qwen"):
        if low == MODEL[f].lower():
            return f, None
    return None


def family_of(cfg: dict) -> str:
    """"minimax" for a MiniMax-M3.1-Flash config (mntq-mm3), "qwen" for CYBER-FROST-3.8-PS (the `mntq` engine with its
    n-gram table, --ple-gguf), "deepseek" for a DeepSeek-V4.1-Flash config (mntq-ds4's --pack: also the configs from
    before the programs' names of v1.1.16), else "glm"."""
    inst = cfg.get("installer") or {}
    exe = Path(str(cfg.get("exe", ""))).name
    if inst.get("family") == "minimax" or exe.startswith("mntq-mm3"):
        return "minimax"
    args = cfg.get("args") or []
    if inst.get("family") == "qwen" or "--ple-gguf" in args[:-1]:
        return "qwen"
    return "deepseek" if inst.get("family") == "deepseek" or "--pack" in args[:-1] or \
        exe.startswith(("mntq-ds4", "strata-ds4")) else "glm"   # (the engine's name before v1.1.16)


def config_key(cfg: dict) -> str | None:
    """The setup's key of a config's files, from its installer record (configs of every release: DeepSeek's and
    MiniMax's before their quants were recorded are the EXL3 quant's); None for a config the setup did not write."""
    inst = cfg.get("installer")
    if not isinstance(inst, dict):
        return None
    fam = family_of(cfg)
    if fam == "deepseek":
        return inst.get("ds4_quant") or "exl3"
    if fam == "minimax":
        return inst.get("mm3_quant") or "exl3"
    if fam == "qwen":
        return inst.get("qwen_quant") or "q5_k_m"
    return inst.get("quant")


# A GGUF file's name, what it is: the parameter sizes, the quant and the format's own words dropped, the rest left
# ("SignOfFour Coder" for Qwen-35B-A3B-SignOfFour-Coder-Q4_K_M.gguf).  A qwen config runs whatever fine-tune its GGUF
# is, so the file names the model; the other families' GGUFs are quants of their one model (SHOWN).
GGUF_NOISE = re.compile(r"^(?:[a-z]?\d+(?:\.\d+)?b|mtp|ttq|gguf|\d+(?:\.\d+)?bpw|i?q\d+(?:_[a-z0-9]+)*)$", re.I)


def qwen_model(cfg: dict) -> str | None:
    """The qwen model a config runs, from its installer record: QWEN_DISTILL for Qwen-3.8-35B-A3B (Distill),
    QWEN_SMALL for Qwen-3.6-35B-A3B-Escha-W2 (the family's qwen35moe models), None for CYBER-FROST-3.8-PS - the
    family's own, and every qwen config written before the moe models existed."""
    return (cfg.get("installer") or {}).get("qwen_model") or None


def gguf_model_name(path) -> str | None:
    """The model a GGUF file is, by its name ("SignOfFour Coder"), or None for no file or nothing left of its name."""
    name = Path(str(path or "")).name
    if not name.lower().endswith(".gguf"):
        return None
    words = [w for w in name[:-5].split("-") if w and not GGUF_NOISE.match(w)]
    return " ".join(words) or None


def config_label(cfg: dict) -> str | None:
    """What a config runs, as users see it ("GLM-5.3 · Maya-S24"), or None for a config the setup did not write.  A
    qwen config is named by the fine-tune its GGUF is ("SignOfFour Coder · Q4_K_M"), or by a published moe model's own
    name (Qwen-3.8-35B-A3B (Distill) / Qwen-3.6-35B-A3B (Escha-W2)); the other families run one model."""
    if not isinstance(cfg.get("installer"), dict):
        return None
    fam = family_of(cfg)
    name = None
    if fam == "qwen" and not qwen_model(cfg):            # (a moe model's name comes from its own key, in shown)
        name = gguf_model_name(cfg["installer"].get("gguf"))
    return shown(fam, config_key(cfg), name=name)
