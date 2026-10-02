"""Real subprocess lifecycle tests without downloading or loading model weights."""
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest

from tts.config import Config
from tts.engine import Cancelled, SubprocessEngine
from tts.voices import Voice

STUB = '''
import json, sys, time
import numpy as np
import soundfile as sf
for line in sys.stdin:
    request = json.loads(line)
    if request['text'] == 'hang':
        time.sleep(60)
    if request['text'] == 'crash':
        sys.exit(2)
    sf.write(request['target'], np.zeros(100), 24000)
    print(json.dumps({'ok': True}), flush=True)
'''


@pytest.fixture
def worker_stub(monkeypatch):
    popen = subprocess.Popen
    children = []

    def launch(args, **kwargs):
        child = popen([sys.executable, '-u', '-c', STUB], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr('tts.engine.subprocess.Popen', launch)
    return children


def generate(engine, tmp_path, text='hi', cancel=None):
    engine.synthesize(text, Voice('a', 'A', 'German', 'Hallo', '/tmp/ref.wav', 'hash'),
                      'German', tmp_path / 'chunk.wav', cancel or threading.Event(), tmp_path / 'log')


def test_worker_recycling(worker_stub, tmp_path):
    engine = SubprocessEngine(Config(root=tmp_path, recycle_chunks=2))
    try:
        generate(engine, tmp_path)
        first = worker_stub[0]
        assert first.poll() is None
        generate(engine, tmp_path)
        assert first.poll() is not None
        generate(engine, tmp_path)
        assert len(worker_stub) == 2
    finally:
        engine.close()
    assert all(p.poll() is not None for p in worker_stub)


@pytest.mark.parametrize('mode', ['cancel', 'timeout', 'crash'])
def test_worker_failure_reaps_process(worker_stub, tmp_path, mode):
    engine = SubprocessEngine(Config(root=tmp_path, chunk_timeout=1))
    cancel = threading.Event()
    timer = threading.Timer(.2, cancel.set) if mode == 'cancel' else None
    if timer:
        timer.start()
    try:
        expected = {'cancel': Cancelled, 'timeout': TimeoutError, 'crash': RuntimeError}[mode]
        with pytest.raises(expected):
            generate(engine, tmp_path, 'crash' if mode == 'crash' else 'hang', cancel)
    finally:
        engine.close()
        if timer:
            timer.join()
    assert worker_stub[0].poll() is not None


@pytest.mark.parametrize('mps', [False, True])
def test_qwen_prompt_reuse_and_protocol(monkeypatch, tmp_path, mps):
    from tts.worker import main
    calls = []
    moves = []
    class Model:
        def __init__(self):
            tokenizer = SimpleNamespace(model=SimpleNamespace(to=lambda d: moves.append(('tokenizer', d))))
            self.model = SimpleNamespace(to=lambda d: moves.append(('model', d)), speech_tokenizer=tokenizer)
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append(('load', model, kwargs))
            print('third party loading log')
            return cls()
        def create_voice_clone_prompt(self, **kwargs):
            calls.append(('prompt', kwargs))
            return 'reusable prompt'
        def generate_voice_clone(self, **kwargs):
            calls.append(('clone', kwargs))
            return [np.zeros(100)], 24000
        def generate_voice_design(self, **kwargs):
            calls.append(('design', kwargs))
            return [np.zeros(100)], 24000
    torch = SimpleNamespace(set_num_threads=lambda n: None, float16='fp16', float32='fp32', device=lambda d: d,
        Tensor=type('Tensor', (), {}), inference_mode=nullcontext,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        mps=SimpleNamespace(set_per_process_memory_fraction=lambda f: None, empty_cache=lambda: None))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'qwen_tts', SimpleNamespace(Qwen3TTSModel=Model))
    request = {'operation': 'clone', 'device': 'auto', 'model': 'base', 'mps_fraction': .6,
               'voice': {'fingerprint': 'a', 'reference': 'ref.wav', 'ref_text': 'Hi'},
               'text': 'Hallo', 'language': 'German', 'settings': {'max_new_tokens': 2048},
               'target': str(tmp_path / 'output.wav')}
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdin', io.StringIO('\n'.join(json.dumps(request) for _ in range(2))))
    monkeypatch.setattr(sys, 'stdout', output)
    main()
    assert [json.loads(line) for line in output.getvalue().splitlines()] == [{'ok': True}] * 2
    assert [c[0] for c in calls] == ['load', 'prompt', 'clone', 'clone']
    assert calls[0][2]['device_map'] == {'': 'cpu'}
    assert calls[0][2]['dtype'] == ('fp16' if mps else 'fp32')
    assert moves == ([('model', 'mps'), ('tokenizer', 'mps')] if mps else [])
    assert calls[-1][1]['voice_clone_prompt'] == 'reusable prompt'
    calls.clear()
    request.update(operation='design', description='warm', model='design')
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(request) + '\n'))
    main()
    assert [c[0] for c in calls] == ['load', 'design']
    assert calls[-1][1]['instruct'] == 'warm'
