"""Run the Haas-ja speech recognition model on CPU (no CUDA, no Triton).

    import haas_asr_cpu
    segments = haas_asr_cpu.transcribe("talk.wav")

or, to use haas_asr directly:

    import haas_asr_cpu
    haas_asr_cpu.install()          # before haas_asr.load_model
    import haas_asr
    model = haas_asr.load_model("third-intelligence/Haas-ja-v1.0", device="cpu")
"""

from .kda import KimiDeltaAttentionCPU, install
from .segments import build_segments, repair_tokens
from .transcribe import DEFAULT_MODEL, transcribe

__all__ = ["DEFAULT_MODEL", "KimiDeltaAttentionCPU", "build_segments", "install", "repair_tokens", "transcribe"]
__version__ = "0.1.0"
