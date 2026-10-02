"""Local dialogue TTS API. Run one server process: python app.py."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import shutil

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tts.config import Config, ROOT
from tts.engine import SubprocessEngine, application_lock
from tts.jobs import JobManager, TERMINAL
from tts.parser import parse_dialogue, prepare_chunks
from tts.voices import LANGUAGES, load_voices


class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    max_chars: int = Field(default=500, ge=100, le=1000)


class JobRequest(ParseRequest):
    voices: dict[str, str] = Field(max_length=100)
    language: str = 'German'
    pause_ms: int = Field(default=250, ge=0, le=3000)


def create_app(config=None, engine_factory=SubprocessEngine):
    config = config or Config()

    @asynccontextmanager
    async def lifespan(app):
        with application_lock(config.cache):
            app.state.manager = JobManager(config, engine_factory)
            try:
                yield
            finally:
                await asyncio.to_thread(app.state.manager.close)

    app = FastAPI(title='Speakerpy · Dialogue TTS', lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])

    def parse(body):
        try:
            turns = parse_dialogue(body.text)
            speakers = list(dict.fromkeys(turn.speaker for turn in turns))
            if len(speakers) > 100:
                raise ValueError('Use at most 100 speakers.')
            return turns, speakers, prepare_chunks(turns, body.max_chars)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def snapshot(job_id):
        try:
            return app.state.manager.snapshot(job_id)
        except KeyError:
            raise HTTPException(404, 'Unknown job (jobs reset when the server restarts).') from None

    @app.get('/health')
    def health():
        return {'status': 'ok', 'ffmpeg': bool(shutil.which('ffmpeg')),
                'model': config.model, 'recycle_chunks': config.recycle_chunks,
                'languages': LANGUAGES}

    @app.get('/voices')
    def voices():
        library, errors = load_voices(config.voices)
        return {'voices': [voice.public() for voice in library.values()], 'errors': errors}

    @app.post('/parse')
    def parse_text(body: ParseRequest):
        turns, speakers, chunks = parse(body)
        return {'speakers': speakers, 'turns': len(turns), 'total': len(chunks)}

    @app.post('/jobs', status_code=202)
    def submit(body: JobRequest):
        _, speakers, chunks = parse(body)
        if body.language not in LANGUAGES:
            raise HTTPException(422, 'Unsupported language.')
        library, _ = load_voices(config.voices)
        if set(body.voices) != set(speakers):
            raise HTTPException(422, 'Assign a voice to every detected speaker, then submit again.')
        if any(voice not in library for voice in body.voices.values()):
            raise HTTPException(422, 'Selected voice is missing or invalid. Refresh the voice library.')
        if not shutil.which('ffmpeg'):
            raise HTTPException(503, 'ffmpeg is required for MP3 export. Install it and retry.')
        try:
            return app.state.manager.submit(chunks, {s: library[body.voices[s]] for s in speakers},
                                            body.language, body.pause_ms)
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/jobs')
    def list_jobs():
        return app.state.manager.list_jobs()

    @app.get('/jobs/{job_id}')
    def get_job(job_id: str):
        return snapshot(job_id)

    @app.post('/jobs/{job_id}/cancel')
    def cancel_job(job_id: str):
        snapshot(job_id)
        return app.state.manager.cancel_job(job_id)

    @app.get('/jobs/{job_id}/events')
    async def events(job_id: str, request: Request):
        snapshot(job_id)

        async def stream():
            previous = None
            heartbeat = 0
            while not await request.is_disconnected():
                try:
                    status = app.state.manager.snapshot(job_id)
                except KeyError:
                    return
                data = json.dumps(status, ensure_ascii=False)
                if data != previous:
                    yield f'id: {status["revision"]}\ndata: {data}\n\n'
                    previous = data
                if status['state'] in TERMINAL:
                    return
                heartbeat += 1
                if heartbeat % 40 == 0:
                    yield ': keep-alive\n\n'
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/jobs/{job_id}/audio/{kind}')
    def audio(job_id: str, kind: str):
        status = snapshot(job_id)
        if kind not in ('wav', 'mp3') or kind not in status['downloads']:
            raise HTTPException(404, 'Audio is not available yet.')
        return FileResponse(config.output / job_id / f'conversation.{kind}',
                            media_type='audio/wav' if kind == 'wav' else 'audio/mpeg',
                            filename=f'conversation.{kind}')

    @app.get('/jobs/{job_id}/log')
    def log(job_id: str):
        snapshot(job_id)
        path = config.output / job_id / 'worker.log'
        if not path.exists():
            raise HTTPException(404, 'No worker log yet.')
        return FileResponse(path, media_type='text/plain', filename=f'{job_id}.log')

    @app.get('/')
    def index():
        return FileResponse(ROOT / 'static' / 'index.html')

    app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
    return app


app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000, workers=1)
