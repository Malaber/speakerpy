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


def fake_design(self, text, description, language, target, cancel, log_path):
    self.synthesize(text, type('Voice', (), {'id': 'designed'})(), language, target, cancel, log_path)


FakeEngine.design = fake_design


def test_first_startup_creates_three_disk_voices(tmp_path):
    FakeEngine.calls, FakeEngine.active, FakeEngine.peak = [], 0, 0
    FakeEngine.gate, FakeEngine.fail_once = None, False
    with TestClient(create_app(Config(root=tmp_path), FakeEngine)) as client:
        jobs = client.get('/jobs').json()
        assert len(jobs) == 3
        for job in jobs:
            assert wait(client, job['id'])['state'] == 'complete'
        library = client.get('/voices').json()['voices']
        assert len(library) == 3
        assert len({v['ref_text'] for v in library}) == 1
        assert len({v['description'] for v in library}) == 3
        assert FakeEngine.peak == 1
        for voice in library:
            assert client.get(f'/voices/{voice["id"]}/audio').content[:4] == b'RIFF'
    with TestClient(create_app(Config(root=tmp_path), FakeEngine)) as client:
        assert client.get('/jobs').json() == []
        assert len(client.get('/voices').json()['voices']) == 3


def test_add_regenerate_and_voice_snapshot(client, tmp_path):
    created = client.post('/voices', json={'name': 'New voice', 'description': 'Deep calm voice'}).json()
    assert wait(client, created['id'])['state'] == 'complete'
    voice_id = created['voice_id']
    before = client.get(f'/voices/{voice_id}/audio').content
    FakeEngine.gate = threading.Event()
    regeneration = client.post(f'/voices/{voice_id}/regenerate', json={'description': 'Warm expressive voice'}).json()
    wait(client, regeneration['id'], 'generating')
    assert client.post(f'/voices/{voice_id}/regenerate', json={}).status_code == 409
    assert client.get(f'/voices/{voice_id}/audio').content == before
    body = {'text': 'A: Test voice snapshot', 'voices': {'A': voice_id}}
    conversation_id = submit(client, body)
    snapshot_voice = client.app.state.manager.jobs[conversation_id].voices['A']
    FakeEngine.gate.set()
    assert wait(client, regeneration['id'])['state'] == 'complete'
    assert wait(client, conversation_id)['state'] == 'complete'
    voice = next(v for v in client.get('/voices').json()['voices'] if v['id'] == voice_id)
    assert voice['description'] == 'Warm expressive voice'
    assert voice['fingerprint'] != snapshot_voice.fingerprint
    assert Path(snapshot_voice.reference).exists()
    # Failed regeneration must preserve the working sample and metadata.
    FakeEngine.fail_once = True
    failed = client.post(f'/voices/{voice_id}/regenerate', json={'description': 'Will fail this time'}).json()
    assert wait(client, failed['id'])['state'] == 'failed'
    voice_after = next(v for v in client.get('/voices').json()['voices'] if v['id'] == voice_id)
    assert voice_after['fingerprint'] == voice['fingerprint']
    retry = client.post(f'/jobs/{failed["id"]}/retry', json={}).json()
    assert wait(client, retry['id'])['state'] == 'complete'


def test_no_pause_inside_turn(client, tmp_path):
    body = {'text': 'A: ' + 'Hallo Welt. ' * 35 + '\nB: Ende.',
            'max_chars': 100, 'voices': {'A': 'anna', 'B': 'klaus'}, 'pause_ms': 300}
    job_id = submit(client, body)
    status = wait(client, job_id)
    assert status['state'] == 'complete'
    assert status['total'] > 2
    assert sf.info(tmp_path / 'output' / job_id / 'conversation.wav').frames == status['total'] * 2400 + 7200


def test_shutdown_cancels_active_job(tmp_path):
    add_voice(tmp_path)
    FakeEngine.gate = threading.Event()
    FakeEngine.fail_once = False
    app = create_app(Config(root=tmp_path), FakeEngine)
    with TestClient(app) as client:
        job_id = submit(client, {'text': 'A: Still running', 'voices': {'A': 'anna'}})
        wait(client, job_id, 'generating')
    assert not app.state.manager.thread.is_alive()
    assert app.state.manager.snapshot(job_id)['state'] == 'cancelled'
    FakeEngine.gate = None


def test_invalid_disk_metadata_reported(client, tmp_path):
    directory = tmp_path / 'voices/broken'
    directory.mkdir()
    (directory / 'voice.json').write_text('[]')
    result = client.get('/voices')
    assert result.status_code == 200
    assert 'JSON object' in result.json()['errors'][0]


def test_failed_mp3_retains_wav(client, monkeypatch):
    def fail(*args):
        raise RuntimeError('encoder failed')
    monkeypatch.setattr('tts.jobs.export_mp3', fail)
    result = wait(client, submit(client))
    assert result['state'] == 'failed'
    assert client.get(result['downloads']['wav']).content[:4] == b'RIFF'
    assert 'mp3' not in result['downloads']


def test_sse_reports_live_progress(client):
    FakeEngine.gate = threading.Event()
    job_id = submit(client)
    wait(client, job_id, 'generating')
    timer = threading.Timer(0.6, FakeEngine.gate.set)
    timer.start()
    try:
        response = client.get(f'/jobs/{job_id}/events')
    finally:
        timer.join()
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert events[0]['state'] == 'generating'
    assert events[0]['completed'] == 0
    assert events[-1]['state'] == 'complete'
    assert events[-1]['completed'] == events[-1]['total'] == 2
    assert response.headers['content-type'].startswith('text/event-stream')


def test_publish_failure_restores_previous_voice(client, monkeypatch, tmp_path):
    before = (tmp_path / 'voices/anna/voice.json').read_bytes()
    original = Path.replace
    def fail_metadata(source, target):
        if source.parent.name.startswith('.') and source.name == 'voice.json':
            raise OSError('simulated publication failure')
        return original(source, target)
    monkeypatch.setattr(Path, 'replace', fail_metadata)
    job = client.post('/voices/anna/regenerate', json={'description': 'A regenerated warm voice'}).json()
    assert wait(client, job['id'])['state'] == 'failed'
    assert (tmp_path / 'voices/anna/voice.json').read_bytes() == before
    assert client.get('/voices/anna/audio').content[:4] == b'RIFF'


def test_staging_directories_not_listed(client, tmp_path):
    staging = tmp_path / 'voices/.unfinished'
    staging.mkdir()
    (staging / 'voice.json').write_text('{}')
    assert client.get('/voices').json()['errors'] == []
