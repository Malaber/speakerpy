import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import create_app
from tts.config import Config
from tts.engine import check_cancel, application_lock


def add_voice(root, voice_id='anna'):
    directory = root / 'voices' / voice_id
    directory.mkdir(parents=True)
    (directory / 'voice.json').write_text(json.dumps({
        'name': voice_id.title(), 'language': 'German', 'ref_text': 'Hallo Welt.'}))
    sf.write(directory / 'reference.wav', np.zeros(2400), 24000)


class FakeEngine:
    calls = []
    active = 0
    peak = 0
    gate = None
    fail_once = False

    def __init__(self, config):
        pass

    def synthesize(self, text, voice, language, target, cancel, log_path):
        type(self).active += 1
        type(self).peak = max(self.peak, self.active)
        try:
            while self.gate and not self.gate.is_set():
                check_cancel(cancel)
                time.sleep(0.01)
            check_cancel(cancel)
            if self.fail_once:
                type(self).fail_once = False
                raise RuntimeError('intentional test failure')
            self.calls.append((text, voice.id))
            sf.write(target, np.full(2400, 0.1), 24000)
        finally:
            type(self).active -= 1

    def close(self):
        pass


@pytest.fixture
def client(tmp_path):
    add_voice(tmp_path)
    add_voice(tmp_path, 'klaus')
    FakeEngine.calls, FakeEngine.active, FakeEngine.peak = [], 0, 0
    FakeEngine.gate, FakeEngine.fail_once = None, False
    with TestClient(create_app(Config(root=tmp_path), FakeEngine)) as client:
        yield client


def payload(text='A: Hallo\nB: Guten Abend'):
    return {'text': text, 'voices': {'A': 'anna', 'B': 'klaus'}, 'pause_ms': 250}


def submit(client, body=None):
    response = client.post('/jobs', json=body or payload())
    assert response.status_code == 202, response.text
    return response.json()['id']


def wait(client, job_id, state=None):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f'/jobs/{job_id}').json()
        if job['state'] == state or (state is None and job['state'] in {'complete', 'failed', 'cancelled'}):
            return job
        time.sleep(0.01)
    pytest.fail(f'Job did not finish: {job}')


def test_api_audio_cache_sse(client, tmp_path):
    first = submit(client)
    result = wait(client, first)
    assert result['state'] == 'complete', result
    assert result['completed'] == result['total'] == 2
    assert FakeEngine.calls == [('Hallo', 'anna'), ('Guten Abend', 'klaus')]
    wav = tmp_path / 'output' / first / 'conversation.wav'
    assert sf.info(wav).frames == 2400 * 2 + 6000
    data, _ = sf.read(wav)
    assert np.all(data[2400:8400] == 0)
    assert client.get(result['downloads']['mp3']).content
    assert client.get(result['downloads']['wav']).content[:4] == b'RIFF'
    assert '"state": "complete"' in client.get(f'/jobs/{first}/events').text
    second = wait(client, submit(client))
    assert second['cached'] == 2
    assert len(FakeEngine.calls) == 2
    changed = payload()
    changed['voices'] = {'A': 'klaus', 'B': 'anna'}
    assert wait(client, submit(client, changed))['cached'] == 0


def test_fifo_responsive_cancel_and_recovery(client):
    FakeEngine.gate = threading.Event()
    first = submit(client)
    wait(client, first, 'generating')
    second = submit(client)
    assert client.get(f'/jobs/{second}').json()['state'] == 'queued'
    start = time.monotonic()
    assert client.get('/health').status_code == 200
    assert client.post('/parse', json={'text': 'A: Still responsive'}).status_code == 200
    assert time.monotonic() - start < 1
    assert client.post(f'/jobs/{first}/cancel').status_code == 200
    assert wait(client, first)['state'] == 'cancelled'
    FakeEngine.gate.set()
    assert wait(client, second)['state'] == 'complete'
    assert FakeEngine.peak == 1


def test_queue_limit_and_cancel_queued(client):
    FakeEngine.gate = threading.Event()
    first = submit(client)
    wait(client, first, 'generating')
    queued = [submit(client) for _ in range(10)]
    assert client.post('/jobs', json=payload()).status_code == 429
    for job in queued:
        assert client.post(f'/jobs/{job}/cancel').json()['state'] == 'cancelled'
    client.post(f'/jobs/{first}/cancel')
    assert wait(client, first)['state'] == 'cancelled'


def test_failure_does_not_stop_queue(client):
    FakeEngine.fail_once = True
    first, second = submit(client), submit(client)
    assert wait(client, first)['state'] == 'failed'
    assert wait(client, second)['state'] == 'complete'


def test_validation(client):
    assert client.post('/parse', json={'text': 'No speaker'}).status_code == 422
    assert client.post('/jobs', json={**payload(), 'voices': {}}).status_code == 422
    assert client.post('/jobs', json={**payload(), 'language': 'unknown'}).status_code == 422
    assert client.post('/jobs', json={**payload(), 'pause_ms': -1}).status_code == 422
    assert client.get('/jobs/unknown').status_code == 404
    assert client.get('/jobs/unknown/events').status_code == 404
    assert len(client.get('/voices').json()['voices']) == 2


def test_corrupt_cache_regenerated(client, tmp_path):
    wait(client, submit(client))
    for path in (tmp_path / 'cache').glob('*.wav'):
        path.write_bytes(b'broken')
    assert wait(client, submit(client))['cached'] == 0


def test_reference_change_invalidates_cache(client, tmp_path):
    wait(client, submit(client))
    sf.write(tmp_path / 'voices/anna/reference.wav', np.ones(2400) * .2, 24000)
    assert wait(client, submit(client))['cached'] == 1


def test_single_server_lock(tmp_path):
    with application_lock(tmp_path):
        with pytest.raises(RuntimeError, match='already running'):
            with application_lock(tmp_path):
                pass
