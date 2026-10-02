"""Use `python -m invoke --list` to see local setup, launch and test commands."""
from pathlib import Path
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from invoke import task

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv' / 'bin' / 'python'


def command(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


@task
def setup(c, lite=False):
    """Install with pip in .venv (use --lite for API development without Qwen)."""
    if not PYTHON.exists():
        command(sys.executable, '-m', 'venv', ROOT / '.venv')
    command(PYTHON, '-m', 'pip', 'install', '-r', 'requirements-app.txt' if lite else 'requirements.txt')
    if not lite:
        (ROOT / '.venv' / '.speakerpy-ready').touch()


@task
def run(c, browser=False):
    """Start the single local server; optionally open the browser after /health succeeds."""
    if not PYTHON.exists():
        raise RuntimeError('Run python -m invoke setup first, or use ./run.sh.')
    process = subprocess.Popen([str(PYTHON), 'app.py'], cwd=ROOT)
    stop = threading.Event()

    def open_when_ready():
        while not stop.wait(0.3) and process.poll() is None:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1) as response:
                    if response.status == 200:
                        webbrowser.open('http://127.0.0.1:8000')
                        return
            except (OSError, urllib.error.URLError):
                pass
    if browser:
        threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        code = process.wait()
        if code:
            raise RuntimeError(f'Server exited with status {code}.')
    except KeyboardInterrupt:
        # Ctrl+C reaches the child too. Allow its lifespan to cancel and reap the worker.
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
    finally:
        stop.set()


@task
def test(c):
    """Install test dependencies, then test API, worker lifecycle, cache and real ffmpeg."""
    command(PYTHON, '-m', 'pip', 'install', '-r', 'requirements-dev.txt')
    command(PYTHON, '-m', 'pytest')


@task
def smoke(c, voice='', url='http://127.0.0.1:8000', timeout=3600):
    """Submit actual speech over HTTP, wait for completion, and verify both downloads."""
    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url.rstrip('/') + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    voices = request('/voices')['voices']
    if not voices:
        raise RuntimeError('Wait for a voice sample to finish, then run smoke again.')
    selected = voice or voices[0]['id']
    text = 'Sprecher 1: Hallo! Dies ist ein kurzer Test der Sprachausgabe.'
    mapping = {'Sprecher 1': selected}
    if not voice and len(voices) > 1:
        text += '\nSprecher 2: Guten Abend! Meine Stimme klingt anders.'
        mapping['Sprecher 2'] = voices[1]['id']
    job = request('/jobs', {'text': text, 'voices': mapping, 'language': 'German'})
    print(f'Job {job["id"]} queued', flush=True)
    deadline, previous = time.monotonic() + int(timeout), None
    while time.monotonic() < deadline:
        job = request(f'/jobs/{job["id"]}')
        status = f'{job["state"]}: {job["completed"]}/{job["total"]}'
        if status != previous:
            print(status, flush=True)
            previous = status
        if job['state'] in ('failed', 'cancelled'):
            raise RuntimeError(job['error'] or job['state'])
        if job['state'] == 'complete':
            for kind, path in job['downloads'].items():
                with urllib.request.urlopen(url.rstrip('/') + path, timeout=30) as audio:
                    header = audio.read(16)
                    if not header or (kind == 'wav' and not header.startswith(b'RIFF')):
                        raise RuntimeError(f'Invalid {kind} download')
            print(f'Speech and both downloads verified: {url}/jobs/{job["id"]}', flush=True)
            return
        time.sleep(1)
    request(f'/jobs/{job["id"]}/cancel', {})
    raise TimeoutError('Smoke test timed out; job cancellation requested.')
