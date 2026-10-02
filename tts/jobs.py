"""A bounded FIFO serviced by exactly one background thread."""
from collections import deque
from dataclasses import dataclass, field, replace
import hashlib
import logging
from pathlib import Path
import shutil
import threading
import uuid

from tts.audio import assemble, export_mp3, valid_audio
from tts.engine import Cancelled, SubprocessEngine, check_cancel
from tts.voices import cache_key

TERMINAL = {'complete', 'failed', 'cancelled'}
logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    chunks: list
    voices: dict
    language: str
    pause_ms: int
    state: str = 'queued'
    completed: int = 0
    cached: int = 0
    current: dict | None = None
    error: str | None = None
    revision: int = 0
    downloads: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)


class JobManager:
    def __init__(self, config, engine_factory=SubprocessEngine):
        self.config, self.engine_factory = config, engine_factory
        config.cache.mkdir(parents=True, exist_ok=True)
        config.output.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.pending = deque()
        self.condition = threading.Condition(threading.RLock())
        self.stopping = False
        self.thread = threading.Thread(target=self._work, name='tts-queue', daemon=False)
        self.thread.start()

    def submit(self, chunks, voices, language, pause_ms):
        with self.condition:
            if self.stopping:
                raise RuntimeError('Server is shutting down.')
            if len(self.pending) >= self.config.queue_size:
                raise OverflowError('Queue is full. Wait for a job to finish.')
            # Keep immutable reference copies, so editing the library cannot poison caches.
            snapshots = {}
            reference_dir = self.config.cache / 'references'
            reference_dir.mkdir(exist_ok=True)
            for speaker, voice in voices.items():
                target = reference_dir / f'{voice.fingerprint}.wav'
                if not target.exists():
                    source = Path(voice.reference)
                    digest = hashlib.sha256((source.parent / 'voice.json').read_bytes())
                    temp = target.with_suffix('.partial.wav')
                    try:
                        with source.open('rb') as src, temp.open('wb') as dst:
                            for block in iter(lambda: src.read(65536), b''):
                                digest.update(block)
                                dst.write(block)
                        if digest.hexdigest() != voice.fingerprint:
                            raise ValueError('Voice changed while submitting; please retry.')
                        temp.replace(target)
                    finally:
                        temp.unlink(missing_ok=True)
                snapshots[speaker] = replace(voice, reference=str(target))
            while len(self.jobs) >= self.config.max_jobs:
                oldest = next((key for key, job in self.jobs.items() if job.state in TERMINAL), None)
                if oldest is None:
                    raise OverflowError('Too many active jobs.')
                del self.jobs[oldest]
            job = Job(uuid.uuid4().hex, chunks, snapshots, language, pause_ms)
            self.jobs[job.id] = job
            self.pending.append(job.id)
            self.condition.notify()
            return self.snapshot(job.id)

    def snapshot(self, job_id):
        with self.condition:
            job = self.jobs[job_id]
            return {'id': job.id, 'state': job.state, 'completed': job.completed,
                    'total': len(job.chunks), 'progress': round(100 * job.completed / len(job.chunks)),
                    'cached': job.cached, 'current': job.current, 'error': job.error,
                    'revision': job.revision, 'downloads': dict(job.downloads),
                    'queue_position': list(self.pending).index(job_id) + 1 if job_id in self.pending else 0}

    def list_jobs(self):
        with self.condition:
            return [self.snapshot(key) for key in reversed(self.jobs)]

    def update(self, job, **changes):
        with self.condition:
            for key, value in changes.items():
                setattr(job, key, value)
            job.revision += 1

    def cancel_job(self, job_id):
        with self.condition:
            job = self.jobs[job_id]
            if job.state not in TERMINAL:
                job.cancel.set()
                if job.id in self.pending:
                    self.pending.remove(job.id)
                    self.update(job, state='cancelled')
                else:
                    self.update(job, state='cancelling')
            return self.snapshot(job_id)

    def close(self):
        with self.condition:
            self.stopping = True
            for job in list(self.jobs.values()):
                if job.state not in TERMINAL:
                    self.cancel_job(job.id)
            self.condition.notify_all()
        self.thread.join()

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopping or self.pending)
                if self.stopping:
                    return
                job = self.jobs[self.pending.popleft()]
                self.update(job, state='generating')
            self._run(job)

    def _run(self, job):
        engine = None
        directory = self.config.output / job.id
        try:
            directory.mkdir()
            engine = self.engine_factory(self.config)
            paths = []
            settings = {**self.config.generation, 'device': self.config.device,
                        'mps_fraction': self.config.mps_fraction}
            for chunk in job.chunks:
                check_cancel(job.cancel)
                voice = job.voices[chunk.speaker]
                self.update(job, current={'speaker': chunk.speaker, 'voice': voice.name, 'text': chunk.text})
                path = self.config.cache / f'{cache_key(self.config.model, voice, job.language, chunk.text, settings)}.wav'
                cached = valid_audio(path)
                if not cached:
                    engine.synthesize(chunk.text, voice, job.language, path, job.cancel, directory / 'worker.log')
                    if not valid_audio(path):
                        raise RuntimeError('Worker did not produce a valid WAV.')
                paths.append(path)
                self.update(job, completed=job.completed + 1, cached=job.cached + int(cached))
            # Release the model before assembly/export, including optional multi-chunk workers.
            engine.close()
            check_cancel(job.cancel)
            self.update(job, state='assembling', current=None)
            wav = directory / 'conversation.wav'
            assemble(paths, job.chunks, wav, job.pause_ms, job.cancel)
            self.update(job, downloads={'wav': f'/jobs/{job.id}/audio/wav'}, state='encoding')
            export_mp3(wav, directory / 'conversation.mp3', job.cancel, directory / 'worker.log')
            check_cancel(job.cancel)
            self.update(job, state='complete', downloads={**job.downloads, 'mp3': f'/jobs/{job.id}/audio/mp3'})
        except Cancelled:
            self.update(job, state='cancelled', current=None)
        except Exception as exc:
            logger.exception('Job %s failed', job.id)
            self.update(job, state='failed', error=str(exc), current=None)
        finally:
            if engine:
                engine.close()
            # Crashed/killed workers can leave an incomplete chunk, never a cache hit.
            for partial in self.config.cache.glob('*.partial.wav'):
                partial.unlink(missing_ok=True)
