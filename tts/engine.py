"""One disposable Qwen worker at a time; no torch imports in the server."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


class Cancelled(Exception):
    pass


@contextmanager
def application_lock(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / '.app.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Speakerpy server or voice designer is already running.') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def check_cancel(cancel):
    if cancel.is_set():
        raise Cancelled('Job cancelled.')


class SubprocessEngine:
    def __init__(self, config):
        self.config = config
        self.process = None
        self.count = 0
        self.log = None

    def synthesize(self, text, voice, language, target, cancel, log_path):
        check_cancel(cancel)
        if self.process is None:
            self.log = log_path.open('a')
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'tts.worker'], cwd=ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, bufsize=1,
                env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '2'})
        request = {'text': text, 'voice': voice.__dict__, 'language': language,
                   'target': str(target), 'model': self.config.model,
                   'settings': self.config.generation, 'device': self.config.device,
                   'mps_fraction': self.config.mps_fraction}
        try:
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
            deadline = time.monotonic() + self.config.chunk_timeout
            while True:
                check_cancel(cancel)
                if time.monotonic() > deadline:
                    raise TimeoutError('Synthesis timed out. See the job log; completed chunks remain cached.')
                ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                if ready:
                    line = self.process.stdout.readline()
                    if not line:
                        raise RuntimeError('TTS worker exited unexpectedly. See the job log.')
                    result = json.loads(line)
                    if not result.get('ok'):
                        raise RuntimeError(result.get('error', 'TTS worker failed.'))
                    break
            self.count += 1
            if self.count >= self.config.recycle_chunks:
                self.close()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process:
            process, self.process = self.process, None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdin.close()
            process.stdout.close()
        if self.log:
            self.log.close()
            self.log = None
        self.count = 0
