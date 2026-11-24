<p align="center"><img src="assets/icon.svg" width="96" height="96" alt="Mantiq"></p>

<h1 align="center">Mantiq</h1>

<p align="center">
  <a href="https://github.com/dahanco/mantiq/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dahanco/mantiq?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://github.com/dahanco/mantiq/releases/latest"><img src="https://img.shields.io/github/v/release/dahanco/mantiq?style=for-the-badge" alt="Latest release"></a>
  <a href="https://github.com/dahanco/mantiq#quick-start"><img src="https://img.shields.io/badge/CUDA-13%2B-76B900?style=for-the-badge&amp;logo=nvidia&amp;logoColor=white" alt="CUDA 13+"></a>
  <a href="https://github.com/dahanco/mantiq#quick-start"><img src="https://img.shields.io/badge/Platforms-Windows%20%7C%20Linux%20%7C%20macOS-2f80ed?style=for-the-badge" alt="Windows, Linux, macOS"></a>
</p>

<p align="center"><b>Run 35B-parameter MoE models on the desktop you already own &mdash; privately, and 1.9&times; faster at decode than llama.cpp.</b><br>
Mantiq is a <b>standalone inference engine</b>, not a wrapper: expert tiering across VRAM &rarr; RAM &rarr; NVMe, a live web
dashboard, and an OpenAI-compatible API &mdash; all in one process, on one machine.</p>

<p align="center"><sub>Qwen3.5/3.6-MoE (35B-A3B) · GLM-5.3-Flash · DeepSeek-V4.1-Flash-1017 · MiniMax-M3.1-Flash · nothing leaves your machine</sub></p>

<p align="center"><img src="assets/dashboard-demo.gif" alt="Mantiq web dashboard at http://127.0.0.1:7711 streaming tokens with a live tok/s counter" width="760"></p>
<p align="center"><sub>Live from the Mantiq dashboard at <code>http://127.0.0.1:7711</code> &mdash; watch tokens stream in with a live tok/s counter</sub></p>

<p align="center"><b><a href="#download">⚡ Download for Windows &mdash; no build required</a></b></p>

<a id="download"></a>

> [!IMPORTANT]
> ## ⚡ Try Mantiq right now &mdash; Windows prebuilt, zero build steps
> Download, unzip, double-click `START-MNTQ.bat`. The first launch asks which model to fetch &mdash; no compiler, no CUDA toolkit, no Python install, nothing added to `PATH`.
>
> | Package | Get it (latest: v1.8.0) | Runs on |
> | --- | --- | --- |
> | 🚀 **CUDA** &mdash; full speed | **[Mantiq-Portable-CUDA.zip](https://github.com/dahanco/mantiq/releases/latest/download/Mantiq-Portable-CUDA.zip)** (~600 MB) | NVIDIA GPU, compute capability 7.5+, CUDA 13-capable driver |
> | 💻 **CPU** &mdash; any 64-bit PC | **[Mantiq-Portable-CPU.zip](https://github.com/dahanco/mantiq/releases/latest/download/Mantiq-Portable-CPU.zip)** (~45 MB) | AVX2 CPU, no GPU needed |
>
> Both ship the engine, Python and the dashboard; **models download on first start**. [All releases](https://github.com/dahanco/mantiq/releases) · [v1.8.0 checksums](https://github.com/dahanco/mantiq/releases/tag/v1.8.0)

## Measured speed: decode, same model, same PC

| Engine | Run 1 | Run 2 | Run 3 | Mean | vs. llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Mantiq**, as shipped (MTP + prompt lookup) | 62.2 | 71.7 | 74.3 | **69.4 tok/s** | **1.9×** |
| **Mantiq**, drafting off (apples to apples) | 47.8 | 52.9 | 53.4 | **51.4 tok/s** | **1.4×** |
| **llama.cpp**, Vulkan, `-ngl 99 -ncmoe 32`, `tg128` | - | - | - | 36.31 tok/s | baseline |
| **llama.cpp**, Vulkan, `-ngl 99 -ncmoe 32`, `tg256` | - | - | - | 31.43 ± 3.47 tok/s | baseline |

Tokens per second, decoding only, on one file: `Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf` (Qwen 35B-A3B MoE, Q4_K_M),
RTX 3070 8 GB, i7-12700K, 32 GB RAM, Windows. Mantiq ran its CUDA path; the llama.cpp build at hand is Vulkan-only (no
`ggml-cuda.dll`), and `-ncmoe 32` was the best of a 28-38 sweep because 20.2 GiB of weights do not fit in 8 GB of VRAM.
Reading the prompt is the other half of the story. The batched Qwen MoE prompt path now sizes its chunk to the free
VRAM (4096 tokens on an 8 GB card; up to 8192 where it fits), raising the 5311-token test prompt from 295.7 tok/s at
the old fixed 2048-token chunk to 301-309 tok/s. llama.cpp CUDA reads the same prompt at 356.4 tok/s (`pp5312`), so
Mantiq is now roughly 13-15% behind on prefill instead of 17%. Generating the answer - what a person waiting at the
keyboard sees - is where the gap goes the other way.

Mantiq is an inference engine. It is not a wrapper around someone else's runtime, and it is not a model server that
hands the work to a library: it *is* the runtime - the kernels, the memory tiers, the tuner, the server and the
dashboard, in one process on one machine.

## What the engine does

A model this size does not fit in VRAM. Mantiq plans around that instead of pretending otherwise:

- **It tiers the experts.** The ones your conversation keeps touching stay in VRAM, the next ones in RAM, the rest on
  the NVMe SSD, and they move between the tiers as you chat - so a 321 B, a 428 B or a 748 B model runs on a desktop.
- **It measures the machine and tunes itself.** GPU count and bandwidth, PCIe share, CPU lanes, read chunk size and RAM
  tier split are calibrated per PC, per model and per context (`--calibrate`), then kept.
- **It moves as few bytes per token as it can.** The decode path is bound by memory traffic, so the quant formats, the
  cache and the expert reads are what the engine is built around - not the arithmetic.
- **It is one runtime end to end.** From the browser to the kernel there is no second server, no HTTP hop in the middle
  and no Python in the decode loop.
- **It serves you, not just the model.** A chat and a live Monitor in the browser, pictures on every model, thinking
  levels, an OpenAI- and Anthropic-compatible API with streamed tool calls, and several requests answered at once.
- **It remembers the shared prompt.** The native `--serve` path keeps a chain of conversation checkpoints at turn
  boundaries, generation seams and periodic chunk ends. An agent that re-sends the whole conversation every request
  (OpenCode and other OpenAI-compatible clients) mounts the longest matching one, so a tool call, a reasoning turn or
  a rewritten older message re-reads only the suffix after the divergence - not the whole session. Retention is set by
  `--prompt-cache`, `--prompt-cache-every` and `--turn-token`.

## How it works

One prompt, one process, and a token that is read from the tier the engine decided on:

```mermaid
flowchart TD
  CAL["once, at setup: the PC is measured and tuned -<br/>GPU count and bandwidth, PCIe share, CPU lanes,<br/>read chunk size, the RAM and SSD tier split"]
  Q["a prompt: the dashboard, or the OpenAI / Anthropic API"] --> A["one process: the server and the engine together -<br/>no second server, no HTTP hop, no Python in the decode loop"]
  A --> B["tokenize, then the decode graph layer by layer<br/>CUDA, HIP, Metal or ggml's CPU backend"]
  B --> C{"the router: which experts<br/>does this token touch?"}
  CAL -.->|"sizes the tiers"| C
  C -->|"the ones the conversation keeps using"| D["VRAM: kept resident"]
  C -->|"the next ones"| E["RAM"]
  C -->|"the rest"| F["NVMe SSD: read ahead"]
  D --> H["the layer's attention, and its KV cache -<br/>a window in VRAM, the rest streamed from RAM"]
  E --> H
  F --> H
  H --> I["the next layer - after the last one, a token"]
  I --> J["drafting: the MTP block or the prompt lookup<br/>proposes a few tokens"]
  J --> K["the full model checks them -<br/>only the accepted ones count"]
  K --> L["the token is streamed to the browser"]
  L -->|"the next token"| B
```

Where a token is read from is the engine's own decision, made per layer and per token: nothing about it is left to a
library, and the bytes it does not need are bytes it never reads.

## Built on three projects

Mantiq did not start from nothing. It stands on three engines, and it takes a specific thing from each of them:

| Project | What Mantiq keeps |
| --- | --- |
| **Strata** | The engine's backbone: the kernel suite and the layer composition - the decode graph, the backend dispatch across CUDA / HIP / Metal, and the shape of a layer done in the right order with the right buffers. |
| **Maya** | The project Mantiq carried over from and renamed: the MoE runtime around the kernels - the expert tiers, the setup that checks a PC and builds once, the calibration, and the browser dashboard. Every "Maya-*" quant name is still accepted as an alias. |
| **llama.cpp / ggml** | The quantization formats and the CPU side of the engine: `ggml`'s K-quants and I-quants, the CPU dot-product kernels, the MMQ prompt-prefill kernels and the GGUF reader, built from a pinned upstream commit. |

The three are not glued together at the edges - they meet inside one engine. Strata's layers call into `ggml`'s
quantized kernels, Maya's tiering decides which of them a token ever touches, and Mantiq's own kernels
(per-architecture CPU expert paths, EXL3 and IQ/Q2_K decode on the GPU, MTP and prompt-lookup drafting) fill the gaps
the other three leave on desktop hardware.

## Why it is the fastest of the three

```mermaid
flowchart LR
  L["llama.cpp / ggml<br/>the quant formats, the CPU kernels, the MMQ prompt path<br/>- general-purpose: not built for a model that does not fit in memory at all"]
  S["Strata<br/>the kernel suite and the layer composition across CUDA, HIP and Metal<br/>- a pure kernel library: no idea which experts you actually use"]
  Y["Maya<br/>the MoE runtime: expert tiers, the setup, the calibration, the dashboard<br/>- a tiering runtime: no idea what the hardware could do with a better kernel"]
  L --> M
  S --> M
  Y --> M
  M["Mantiq"] --> M1["the three meet inside one decode loop, not at its edges"]
  M1 --> M2["then its own parts, all of them on the critical path:<br/>quants chosen for the bottleneck, AVX-2 CPU expert paths,<br/>grouped-expert GPU kernels, isolated and measured drafting"]
  M2 --> E["the fastest of the three:<br/>what you pay is bytes moved per token, not FLOPs"]
```

Taken one at a time, each of the three is fast at the thing it was built for and slow at the rest: a pure kernel
library has no idea which experts you actually use, a tiering runtime has no idea what the hardware could do with a
better kernel, and a general-purpose CPU engine is not built for a model that does not fit in memory at all. Mantiq is
the fastest of the three because it is the three *combined and tuned*, and because the parts it adds are all on the
critical path:

- **The quant is chosen for the bottleneck.** Decode speed is bytes moved, so the engine ships its own quants
  (IQ2_XXS-class experts, Q4_K attention, EXL3 tiles) that read less per token than a stock format while staying close
  to the original's answers.
- **The tiering is what makes the model run at all.** Speculative read-ahead, expert caching and a calibrated RAM /
  PCIe / SSD split turn "does not fit" into a steady decode rate - a runtime that cannot do this does not finish the
  sentence.
- **The kernels are tuned for this class of GPU.** AVX-2 CPU expert paths (including hand-written Q6_K down
  projections), grouped-expert GPU kernels for IQ/Q2_K, EXL3 decoding and fused MoE prefill - work aimed at cards of
  8-24 GB, not at datacenter shapes.
- **Drafting is a first-class, measured switch.** MTP and prompt-lookup drafting are isolated from each other so each
  one's acceptance rate is measured rather than guessed, and the fastest settings for a model are one button away.
- **It never leaves the machine.** Every token is served from your own GPU, RAM and disk - no cloud round trip, no
  queue, no bill.

Measure yours with `./mntq.sh --bench` (decode and prefill), and `./mntq.sh --report` to write the machine's numbers to
`mntq-report.txt`. Speeds from other users' machines are welcome in a
[GitHub issue](https://github.com/dahanco/mantiq/issues).

## Quick start

```sh
git clone https://github.com/dahanco/mantiq.git && cd mantiq
./setup.sh
```

The setup asks which model you want, downloads it, builds everything once, **starts the model**, and leaves a dashboard
at `http://127.0.0.1:7711`. Windows: `START-MNTQ.bat` instead of `./setup.sh`. Every option (GPUs, context, an existing
model folder, remote access, ...) is a question the setup asks; Enter takes the recommended answer.

**Requirements in short:** an NVIDIA GPU of compute capability 7.0 or newer for a build from source (compute
capability 6.0 is experimental), 32 GB of RAM, and **a fast NVMe SSD** with room for the model. The prebuilt CUDA
package targets compute capability 7.5+ and needs an NVIDIA driver that supports CUDA 13. AMD (ROCm) and Windows are
supported experimentally, and Apple Silicon through Metal for GLM-5.3-Flash. The setup checks all of it and prints the
exact command for anything missing; it installs nothing system-wide.

**Prebuilt packages (Windows, nothing to build):** the CUDA and CPU portable zips are linked in the
[download callout](#download) at the top &mdash; unzip, double-click `START-MNTQ.bat`, and the first start downloads the
model. A Vulkan package also exists but is experimental and is not refreshed with every release.

## Models that run on it

Every link goes to the publisher's own repository, where the files, the license and the quant's card are. The setup
downloads the ones it offers itself when it asks which model you want; a GGUF you already have is added by hand.

**The smaller ones: 35B-A3B Qwen MoE.** The setup ships two of them (Qwen-3.8-35B-A3B, the distill, and
Qwen-3.6-35B-A3B, Escha-W2), and any other file of the same shape runs just as well: download the GGUF, then the
dashboard's Models page has the button, or `./mntq.sh --add-local-model <file.gguf>`. Mantiq serves it on the Qwen path:

| Model | Repository |
| --- | --- |
| Qwopus3.6-35B-A3B Coder, with its MTP draft block | [Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF](https://huggingface.co/Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF/tree/main) |
| Qwen-35B-A3B SignOfFour Coder | [pragmaticcs/Qwen-35B-A3B-SignOfFour-Coder-GGUF](https://huggingface.co/pragmaticcs/Qwen-35B-A3B-SignOfFour-Coder-GGUF/tree/main) |

**The Maya line: GLM-5.3-Flash quantized for this engine.** These are the large ones the tiering and the calibration
exist for; the names above are the files' own, and the setup accepts them by its `Mantiq-*` ids or by the Maya names the
Hugging Face folders carry.

| Quant | Size | What it is | Repository |
| --- | --- | --- | --- |
| `GLM-5.3-Flash-Maya-XS-IQ1_S` | 73.4 GB | the smallest of them: a 64 GB PC with a 24 GB card at 128K context | [Maya-XS](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/tree/main/Maya-XS) |
| `GLM-5.3-Flash-Maya-S24` | 94.7 GB | about 1.5 GB less that stays on the GPU, so 24 GB cards hold more experts (about 14% faster decode there) | [Maya-S24](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/tree/main/Maya-S24) |
| `GLM-5.3-Flash-Maya-S-v2-IQ2_XXS` | 96.5 GB | the compact one: error-feedback-rounded IQ2_XXS experts, IQ2_S / IQ3_XXS down projections, Q6_K attention, the MTP draft block | [Maya-S](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/tree/main/Maya-S-v2-IQ2_XXS) |
| `GLM-5.3-Flash-Maya-M` | 116.0 GB | IQ2_S gate / up experts, IQ3_XXS down projections (IQ3_S in the most sensitive layers) | [Maya-M](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/tree/main/Maya-M) |
| `GLM-5.3-Flash-Maya-M-Derisked` | 116.0 GB | Maya-M with a directional weight edit by Blackfrost_AI that reduces blanket refusals (experimental) | [Maya-M-Derisked](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF) |
| `GLM-5.3-Flash-Maya-L` | 156.3 GB | the largest: IQ3_S gate / up experts, IQ4_XS down projections (Q5_K in the most sensitive layers) | [Maya-L](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/tree/main/Maya-L) |

GLM-5.3-Flash has published EXL3 quants too - [3.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw),
[4.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/4.05bpw),
[sc_4.10 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/sc_4.10bpw) - and a
[DFlash2 drafter](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF). DeepSeek-V4.1-Flash-1017 and
MiniMax-M3.1-Flash have quants for this engine of their own, in `peasantsmith`'s `-Mantiq-GGUF` repositories (both are
gated: ask their publisher for access), and the setup offers them like any other model.

## Credits and license

- Mantiq is created and led by [@dahanco](https://github.com/dahanco).
- Built on **[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT License, Copyright (c) 2023-2026 The ggml
  authors): the quantization formats, the CPU dot products and the prompt path's MMQ kernels, from a pinned commit
  (`third_party/ggml/LICENSE`).
- **EXL3 support** follows **[exllamav3](https://github.com/turboderp-org/exllamav3)**'s format, codebooks and
  tensor-core tile decoding (MIT License, Copyright (c) 2025 Turboderp; `third_party/exllamav3/LICENSE`). **The DFlash2
  drafter's forward** follows **[beellama.cpp](https://github.com/Anbeeld/beellama.cpp)** (MIT License, Copyright (c)
  2023-2026 The ggml authors, Anbeeld; `third_party/beellama.cpp/LICENSE`).
- The models and their licenses (GLM-5.3-Flash by zai-org, DeepSeek-V4.1-Flash-1017 by DeepSeek, MiniMax-M3.1-Flash
  under the MiniMax Community License) belong to their authors; the quants here are made from their released weights
  and keep their terms.
- Mantiq is open source under the [MIT License](LICENSE); the notices in [LICENSE](LICENSE) and `third_party/` stay
  with every copy.
