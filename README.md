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

The same layer also runs on a GPU without fla. On an AMD GPU (ROCm on Windows) it gave exactly the CPU
transcripts about 4 times faster; see [On a GPU](#on-a-gpu).

## Results

Five Japanese TEDx talks (42 min in total), human-made Japanese subtitles as the reference, all on CPU
(AMD Ryzen AI MAX+ 395, 16 threads, WSL2). CER is pooled over the five talks.

| Model | CER | kana CER | kana CER, fillers excluded |
|---|---|---|---|
| **Haas-ja-v1.0 (this package, float32)** | **9.36%** | **6.10%** | **5.82%** |
| Whisper large-v3 (faster-whisper, int8; 3 runs) | 10.19–10.89% | 7.00–7.63% | 7.00–7.62% |
| Parakeet-ja (`nvidia/parakeet-tdt_ctc-0.6b-ja`) | 13.19% | 8.51% | 8.49% |

Time for the 42 minutes, with the model loaded again for each file:

| Model | Total | Loading | Inference | Inference real-time factor |
|---|---|---|---|---|
| **Haas-ja-v1.0 (this package)** | 158 s | 82 s | **76 s** | **0.030** |
| Whisper large-v3 | 1,055–1,153 s | 21–34 s | 1,034–1,119 s | 0.41–0.44 |
| Parakeet-ja | 165 s | 87 s | 78 s | 0.031 |

- **kana CER** compares readings (both sides converted to hiragana with pykakasi), so kanji/kana spelling
  differences such as 有る/ある are not counted as errors.
- **fillers excluded** does not count fillers (あの, えー, …) that a model added but the subtitles omit.
  On two of the five talks Haas wrote many fillers; Whisper and Parakeet wrote almost none.
- Per talk (kana CER, fillers excluded), Haas was best on two talks in every run. On one talk the winner against Whisper changed between
  Whisper runs, Whisper was slightly better on another (8.45% vs 8.59%) and Parakeet on the last
  (7.40% vs 8.11%).
- Whisper's results changed between identical runs on two of the five talks (for example 5.31%, 10.27%
  and 6.23%); the bad runs contain repeated sentences and garbage text. faster-whisper retries with a
  higher sampling temperature when an output looks wrong, which is likely the cause. Haas and Parakeet ran twice
  and gave identical results both times.
- Loading is importing the libraries and loading the model. Averaged per file: 16 s for Haas (including
  starting a subprocess, since it runs in its own environment), 17 s for Parakeet, 4–7 s for Whisper. With
  `haas_asr_cpu.load()` the model is loaded once for many files.
- Subtitle credit lines (翻訳: … / 校正: …) are removed from the reference.

This is one machine and five talks of one genre, not a benchmark. The official results are on
[asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench).

A write-up of this comparison (in Japanese), with how the port was made:
[日本語の文字おこしはどれが正確？ Haas-ja・Whisper large-v3・Parakeet をCPUだけで比べた実測](https://congaroo.org/media/technology/haas-ja-whisper-parakeet-cpu-comparison).

## Install

Requires Python 3.11 or 3.12 (tested with 3.12, which `.python-version` selects) and [uv](https://docs.astral.sh/uv/).
Several NeMo dependencies have no wheels for Python 3.13+, so newer versions are excluded. uv is needed, not just recommended: the
CPU-only PyTorch index and the same dependency settings as Haas-ASR (overrides to versions with security
fixes, bytecode compiled at install) are uv settings in `pyproject.toml`. A plain `pip install` does not
apply them, and pulls the CUDA build of PyTorch on Linux.

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
[00:00:50.427 --> 00:01:00.107] 私の家では…
```

The transcript goes to stdout and progress to stderr (`-q` silences it). NeMo's log level is set to
ERROR (pass `nemo_logs=True` to `load()` / `transcribe()` to keep its logs); a few import-time messages
from other libraries can still appear on stderr.

Python:

```python
import haas_asr_cpu

segments = haas_asr_cpu.transcribe("talk.wav", threads=8)
# [{"start": 50.427, "end": 60.107, "text": "..."}, ...]

# several files: load the model once (9–14 s on the test machine) and reuse it
model = haas_asr_cpu.load(threads=8)
for path in ["a.wav", "b.wav"]:
    segments = haas_asr_cpu.transcribe(path, model=model)
```

On a GPU, pass `device="cuda"` to `load()` / `transcribe()`, or `--device cuda` to the command; see
[On a GPU](#on-a-gpu).

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

The tests check:

- the recurrence equals fla's own `naive_recurrent_kda` and also `naive_chunk_kda` (a different,
  chunk-parallel algorithm), loaded as files so that fla's Triton code is never imported;
- the whole layer equals a reference written out in the tests from the fla formulas above (gate, L2 norm,
  beta, short convolution, gated norm) around fla's `naive_recurrent_kda`. Those other parts are not run
  from fla's code itself, because their files import Triton;
- decoding one token at a time with a cache equals running the whole sequence at once, for both the
  short convolution and the full layer.

```bash
uv run pytest
```

### Sentence-level timestamps (`segments.py`)

NeMo builds segments by grouping space-separated words at `.` `!` `?`. Japanese has no spaces, so each
chunk (about 30 s, cut at quiet points) comes back as one segment. This package rebuilds segments from the per-token timestamps
and closes a segment at:

- a sentence-final mark (。？！?!);
- a pause of 0.8 s or more;
- after 8 s, a comma (、，,) or a pause of 0.3 s or more;
- 20 s at the latest (only a single token can be longer).

A fragment of 2 characters or fewer joins the next segment of the same chunk, unless a pause of 0.8 s
separates them or the joined segment would pass 20 s.

TDT sometimes places the first token of a chunk at 0.0 s although it is spoken much later (one talk gave
"こ" at 0.0 s and "んにちは" from 16.4 s). When the first token is at 0.0 s, is followed by a pause of 0.8 s
or more and does not end a sentence or clause, it is moved to the next token. This is a heuristic: real
speech at the very start of a chunk followed by a pause is moved too.

On a 477-second talk this gives 62 segments (median 4.1 s, longest 15.4 s) instead of 16. Matched to the
human subtitle cue that starts with the same text, segments start a median 0.5 s earlier on that talk;
on the other four talks the median difference is about 0.3 s or less (depending on how cues are matched).

Rare kanji such as 兎 are encoded as several SentencePiece byte-fallback tokens, each decoding to U+FFFD.
`repair_tokens` aligns the tokens with the full hypothesis text and restores the characters. If the tokens
still do not match the text, that chunk falls back to one chunk-level segment.

## On a GPU

`device="cuda"` (`--device cuda`) loads the model on a GPU; PyTorch calls AMD GPUs with ROCm "cuda" too.
`dtype="bfloat16"` (`--dtype bfloat16`) uses the precision of the official results; the default is float32.
Tested only on one machine: AMD Radeon 8060S (Ryzen AI MAX+ 395, gfx1151), Windows 11, ROCm 10.1 with
PyTorch 2.14.0. Same five talks as above:

| | Inference (42 min) | Inference real-time factor | CER | kana CER, fillers excluded |
|---|---|---|---|---|
| CPU, float32 (above) | 76 s | 0.030 | 9.36% | 5.82% |
| **AMD GPU, float32** | **18 s** | **0.0072** | 9.36% | 5.82% |
| AMD GPU, bfloat16 | 13 s | 0.0051 | 9.35% | 5.84% |

- In float32 the GPU gave the same segments as the CPU, text and times, on all five talks. In bfloat16 a
  few characters changed on two talks.
- Loading the model took about 15 s, as on the CPU.
- With a ROCm build of PyTorch, `load()` makes two more process-wide changes, both needed on that
  machine. NeMo's decoder checks NVIDIA's driver whenever PyTorch sees a GPU, even for `device="cpu"`,
  and that check is made to report "not supported". On the GPU, `torch.backends.cudnn` is also disabled,
  because MIOpen could not compile its batch-norm kernel (`miopenStatusUnknownError`), and the decoder
  runs without CUDA (HIP) graphs: with them the transcript of one talk changed and decoding was slower.
- An NVIDIA GPU is not tested. There, the official Haas-ASR with fla's kernels is the better choice.
- `device="cpu"` works in the same ROCm environment but is slow there: one 354 s talk took 276 s of
  inference with 8 threads, against 11 s with 16 threads for the CPU build of PyTorch on WSL.
- `uv sync` installs the CPU build of PyTorch. For ROCm on Windows, install PyTorch from AMD's index first,
  then this package without letting it replace PyTorch, for example:

  ```powershell
  uv venv --python 3.12
  uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1151]==2.14.0+rocm10.1.0" "torchaudio==2.11.0.3+rocm10.1.0"
  # haas-asr and haas-asr-cpu built as wheels (uv build); overrides.txt pins torch/torchaudio to the
  # versions above plus the dependency overrides in pyproject.toml
  # --compile-bytecode as in pyproject.toml: NeMo 2.7.3 can break when modules are compiled at run time
  uv pip install --compile-bytecode --override overrides.txt --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ haas_asr-0.1.0-py3-none-any.whl
  uv pip install --no-deps haas_asr_cpu-0.1.0-py3-none-any.whl
  ```

## Limitations

- **Not compared with the official GPU version.** The official numbers are measured in bfloat16 on NVIDIA
  GPUs with fla's kernels; this runs in float32 on CPU, so transcripts may differ slightly. Only an AMD GPU
  has been tried (see [On a GPU](#on-a-gpu)).
- **CPU: tested on Linux (WSL2) only.** With Python 3.12, Windows x64 and macOS 14+ on Apple Silicon have
  wheels for every compiled dependency in `uv.lock` and should work, but are untested. Intel Macs and
  macOS 13 or earlier cannot install (PyTorch 2.15 has no wheels for them), nor can Windows on ARM
  (several compiled dependencies, such as kaldialign and pyarrow, have no wheels for it).
- **Process-wide effects.** `load()` calls `install()`, which replaces `haas_asr.prednet.load_kda_class`
  for the rest of the process (a later `haas_asr.load_model(..., device="cuda")` would also get the CPU
  layer). `threads` sets `torch.set_num_threads` for the process. Unless `nemo_logs=True`, NeMo's logger is
  set to ERROR. Decoding with timestamps leaves the model's `compute_timestamps` setting on.
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
実装(flash-linear-attention)が、Triton と CUDA でしか動かないためです。この予測ネットワークは 2 層・
390 万パラメータで、モデル全体(6 億 1,580 万)の 1% もありません。ここを同じパラメータ名・同じ計算の純 PyTorch の層に
差し替えて、公開されている重みをそのまま CPU で動かします。あわせて、日本語の文ごとのタイムスタンプも出します。

- **精度**: 日本語の TEDx 講演 5 本(計 42 分)で、表記ゆれとフィラーの差を除いたかな CER が 5.82%。
  同じ CPU の Whisper large-v3(3 回の実行で 7.00〜7.62%)と Parakeet-ja(8.49%)より低い値でした(上の表)。
- **速さ**: 推論の実時間比 0.030(42 分を 76 秒)。推論は Whisper large-v3 の約 14 倍の速さです(読み込みを含めると約 7 倍)。
- **GPU でも**: fla 無しで GPU でも動きます(`--device cuda`)。AMD の GPU(Radeon 8060S・Windows の ROCm)では、
  float32 で CPU と同じ文字おこし・同じ時刻のまま、推論が約 4 倍速くなりました(42 分を 18 秒)。
- **確かめ方**: 中心の再帰計算が flash-linear-attention 自身の参照実装 2 種と一致すること、層全体が fla の式どおりに書いた参照と一致すること、1 トークンずつ流した結果 = 一括の結果になることを、テストで確かめています。
- **未確認**: 公式の GPU 版(NVIDIA・bfloat16)との出力の突き合わせ、NVIDIA の GPU、macOS と Windows の CPU での動作。
- **記事**: 比べ方と結果の詳しい説明は [日本語の文字おこしはどれが正確？ Haas-ja・Whisper large-v3・Parakeet をCPUだけで比べた実測](https://congaroo.org/media/technology/haas-ja-whisper-parakeet-cpu-comparison)。

```bash
git clone https://github.com/nakata-data-sci/haas-asr-cpu.git
cd haas-asr-cpu
uv sync
uv run haas-cpu-transcribe talk.wav --srt talk.srt --threads 8
```

非公式のプロジェクトです。Third Intelligence 社とは関係ありません。
