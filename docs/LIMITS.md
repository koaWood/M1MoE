# Model Limits — M1 / 16 GB Reference

Benchmarks run on **Apple M1, 16 GB unified memory**, release build (`-c release`).

---

## OLMoE-1B-7B-0924

| Metric | Value |
|---|---|
| Generation speed (release) | **~6 tok/s** (163–173 ms/tok) |
| Generation speed (debug) | ~1.9 tok/s (535–540 ms/tok) |
| Prefill speed (release) | ~174–189 ms/tok |
| Max context | 4 096 tokens |
| DRAM for KV cache | 1 GB (16 layers × 64 MB) |
| Expert DRAM window | 1 024 expert slots — all 1 024 experts fit in DRAM, so no SSD streaming during generation |

**Summary:** OLMoE is a good choice for quick model verification on base M1. All experts fit in the 16 GB window, so generation is steady at roughly **6 tok/s** in release mode. However, the model is basically unusable, as simple prompts like "The capital of France is?" results in a looping output of "The".  LoL.  Quality degrades noticeably at higher temperatures on short or ambiguous prompts; lower temperatures (0.3–0.5) give more coherent output, but don't expect miracles.

---

## Mixtral-8x7B-Instruct-v0.1

| Metric | Value |
|---|---|
| Generation speed (release) | **~0.39 tok/s** (~2 550–2 575 ms/tok) |
| Prefill speed (release) | ~2 900–4 500 ms/tok |
| Max context | 32 768 tokens |
| DRAM for KV cache | 2 GB (32 layers × 64 MB) |
| Expert DRAM window | 78 of 256 total expert slots fit — remaining experts must be streamed from SSD |
| Each expert size | 94 MB |

**Summary:** Mixtral is runnable on M1/16 GB but it is not really practical for interactive use. Only 78 of the 256 expert weight blocks fit in DRAM simultaneously; every cache miss requires loading a 94 MB chunk from SSD, which dominates latency. At **~0.39 tok/s** (roughly one token every 2.5 seconds), it is better suited for batch or offline use. A Mac with **32 GB or more** of unified memory is the realistic minimum for a tolerable Mixtral experience.  Notwithstanding it's interminably slow response, the model itself is quite capable.  Not only did it recognize the Capital of France, but it can go on to explain what's amazing about the city and if you'd like, an interesting monologue on it's history as well.  But, it's about 3 mintues per paragrah on my M1 16GB RAM Macbook Air.

---

## Build mode matters

Running the debug build (`swift run bench ...` without `-c release`) is **3–4× slower** for OLMoE and should be avoided for any real workload. Always use `swift run -c release` or `swift build -c release`.
