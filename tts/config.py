from dataclasses import dataclass, field
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    root: Path = ROOT
    model: str = field(default_factory=lambda: os.getenv('TTS_MODEL', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'))
    device: str = field(default_factory=lambda: os.getenv('TTS_DEVICE', 'auto'))
    recycle_chunks: int = field(default_factory=lambda: int(os.getenv('TTS_RECYCLE_CHUNKS', '1')))
    chunk_timeout: int = field(default_factory=lambda: int(os.getenv('TTS_CHUNK_TIMEOUT', '1800')))
    mps_fraction: float = field(default_factory=lambda: float(os.getenv('TTS_MPS_MEMORY_FRACTION', '0.6')))
    max_jobs: int = 100
    queue_size: int = 10
    generation: dict = field(default_factory=lambda: {'max_new_tokens': 2048, 'do_sample': True})

    def __post_init__(self):
        if not 1 <= self.recycle_chunks <= 20:
            raise ValueError('TTS_RECYCLE_CHUNKS must be between 1 and 20.')
        if self.device not in ('auto', 'mps', 'cpu'):
            raise ValueError('TTS_DEVICE must be auto, mps, or cpu.')
        if not 0 < self.mps_fraction <= 1 or self.chunk_timeout < 1:
            raise ValueError('Invalid memory fraction or chunk timeout.')

    @property
    def cache(self):
        return self.root / 'cache'

    @property
    def output(self):
        return self.root / 'output'

    @property
    def voices(self):
        return self.root / 'voices'
