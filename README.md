# haas-asr-cpu

Run [Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0), a Japanese speech recognition
model, **on CPU** — no NVIDIA GPU, no CUDA, no Triton.

[日本語の説明は下にあります](#日本語)

Officially, [Haas-ASR](https://github.com/third-intelligence/Haas-ASR) needs Linux with an NVIDIA GPU:
the decoder's prediction network uses Kimi Delta Attention (KDA) from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention), whose kernels are Triton +
CUDA only. That network is 2 layers and 3.9M of the model's 615.8M parameters; the FastConformer encoder
already runs on CPU. This package swaps in a pure-PyTorch KDA layer with the same parameter names, so the
released weights load unchanged, and adds sentence-level timestamps for Japanese.

## Results

Five Japanese TEDx talks (42 min in total), human-made Japanese subtitles as the reference, all on CPU
(AMD Ryzen AI MAX+ 395, 16 threads, WSL2). CER is pooled over the five talks.

| Model | CER | kana CER | kana CER, fillers excluded | Real-time factor | Time for 42 min |
|---|---|---|---|---|---|
| **Haas-ja-v1.0 (this package, float32)** | **10.41%** | **7.13%** | **6.86%** | 0.06 | 163 s |
| Whisper large-v3 (faster-whisper, int8) | 11.24% | 8.02% | 8.02% | 0.45 | 1,129 s |
| Parakeet-ja (`nvidia/parakeet-tdt_ctc-0.6b-ja`) | 14.13% | 9.42% | 9.41% | 0.07 | 173 s |

- **kana CER** compares readings (both sides converted to hiragana with pykakasi), so kanji/kana spelling
  differences such as 有る/ある are not counted as errors.
- **fillers excluded** does not count fillers (あの, えー, …) that a model added but the subtitles omit.
  On two of the five talks Haas wrote many fillers; Whisper and Parakeet wrote almost none.
- Haas won on 3 of the 5 talks. Whisper was slightly better on one (9.19% vs 9.39%) and Parakeet on
  another (8.16% vs 8.94%).
- Haas's time includes loading the model (about 9 s per file).

This is one machine and five talks of one genre, not a benchmark. The official results are on
[asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (recommended, it applies the dependency
overrides Haas-ASR needs).

```bash
git clone https://github.com/nakata-data-sci/haas-asr-cpu.git
cd haas-asr-cpu
uv sync
```

The model (1.2 GB) is downloaded from Hugging Face on first use.

## Usage

Command line:

```bash
uv run haas-cpu-transcribe talk.wav --json talk.json --srt talk.srt --threads 8
```

```text
[00:00:50.427 --> 00:00:54.911] 私の家では…
```

Python:

```python
import haas_asr_cpu

segments = haas_asr_cpu.transcribe("talk.wav", threads=8)
# [{"start": 50.427, "end": 54.911, "text": "..."}, ...]
```

Or patch Haas-ASR and use its own API:

```python
import haas_asr_cpu
haas_asr_cpu.install()  # before loading the model

import haas_asr
model = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", device="cpu")
```

## How it works

### Kimi Delta Attention on CPU (`kda.py`)

`KimiDeltaAttentionCPU` has the same constructor arguments and parameter names as
`fla.layers.kda.KimiDeltaAttention` (0.5.2). The math follows fla-core 0.5.2:

| Part | Source in fla-core 0.5.2 |
|---|---|
| State update and output | `ops/kda/naive.py` `naive_recurrent_kda` |
| Gate `g = -exp(A_log) * softplus(g + dt_bias)` | `ops/kda/gate.py` `naive_kda_gate` |
| L2-normalised q, k (eps 1e-6), sigmoid on beta | `ops/kda/fused_recurrent.py` |
| Causal depthwise conv + SiLU with a cache | `modules/conv/short_conv.py` |
| RMSNorm * weight * sigmoid(g) | `modules/fused_norm_gate.py` |

`install()` replaces `haas_asr.prednet.load_kda_class`, so Haas builds its prediction network with this
layer. The prediction network runs one token at a time during decoding, so a plain loop is fast enough:
the real-time factor matches Parakeet-ja, which has the same architecture with an LSTM.

The tests check the layer against fla's own reference implementations, loaded as files so that fla's
Triton code is never imported:

- the recurrence equals `naive_recurrent_kda` and also `naive_chunk_kda` (a different, chunk-parallel
  algorithm);
- decoding one token at a time with a cache equals running the whole sequence at once, for both the
  short convolution and the full layer.

```bash
uv run pytest
```

### Sentence-level timestamps (`segments.py`)

NeMo builds segments by grouping space-separated words at `.` `!` `?`. Japanese has no spaces, so each
30-second chunk comes back as one segment. This package rebuilds segments from the per-token timestamps
and closes a segment at:

- a sentence-final mark (。？！?!);
- a pause of 0.8 s or more;
- after 8 s, a comma (、,) or a pause of 0.3 s or more;
- 20 s at the latest.

On a 477-second talk this gives 62 segments (median 4.1 s, longest 15.4 s) instead of 16, with start
times 0.6–0.8 s earlier than the human subtitle cues.

Rare kanji such as 兎 are encoded as several SentencePiece byte-fallback tokens, each decoding to U+FFFD.
`repair_tokens` aligns the tokens with the full hypothesis text and restores the characters. If the tokens
still do not match the text, that chunk falls back to one chunk-level segment.

## Limitations

- **Not compared with the GPU version.** The official numbers are measured in bfloat16 on NVIDIA GPUs;
  this runs in float32 on CPU, so transcripts may differ slightly. A side-by-side check on a GPU has not
  been done yet.
- **Tested on Linux (WSL2) only.** macOS and Windows should work since everything is plain PyTorch, but
  they are untested.
- **Pinned to Haas-ASR `acc614f`** (2026-10-30). `install()` relies on `haas_asr.prednet.load_kda_class`;
  a later Haas-ASR may change it.
- Only the inference path used by Haas is implemented (no attention mask, no `allow_neg_eigval` /
  `lower_bound`); unsupported options raise `NotImplementedError`.

## License

Apache License 2.0 (see `LICENSE`). `kda.py` follows flash-linear-attention (MIT License); its notice is
in `NOTICE`. Haas-ASR and the Haas-ja-v1.0 weights are Apache License 2.0 and are downloaded, not
redistributed.

This is an unofficial project, not affiliated with Third Intelligence.

---

## 日本語

日本語の音声認識モデル [Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) を、
**NVIDIA GPU なし・CPU だけ** で動かすためのパッケージです。

公式の Haas-ASR は NVIDIA GPU が必須です。デコーダーの予測ネットワークが使う Kimi Delta Attention の
実装(flash-linear-attention)が、Triton と CUDA でしか動かないためです。この部品は 2 層・390 万
パラメータで、モデル全体(6 億 1,580 万)の 1% もありません。ここを同じパラメータ名・同じ計算の純 PyTorch の層に
差し替えて、公開されている重みをそのまま CPU で動かします。あわせて、日本語の文ごとのタイムスタンプも出します。

- **精度**: 日本語の TEDx 講演 5 本(計 42 分)で、表記ゆれをならしたかな CER が 6.86%。
  同じ CPU の Whisper large-v3(8.02%)と Parakeet-ja(9.41%)より低い値でした(上の表)。
- **速さ**: 実時間比 0.06(42 分を 163 秒)。Whisper large-v3 の約 7 倍です。
- **確かめ方**: flash-linear-attention 自身の参照実装 2 種と出力が一致することを、テストで確かめています。
- **未確認**: GPU 版(bfloat16)との出力の突き合わせ、macOS・Windows での動作。

```bash
git clone https://github.com/nakata-data-sci/haas-asr-cpu.git
cd haas-asr-cpu
uv sync
uv run haas-cpu-transcribe talk.wav --srt talk.srt --threads 8
```

非公式のプロジェクトです。Third Intelligence 社とは関係ありません。
