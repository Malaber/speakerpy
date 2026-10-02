"""Local dialogue TTS API. Run one server process: python app.py."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tts.config import Config, ROOT
from tts.engine import SubprocessEngine, application_lock
from tts.jobs import JobManager, TERMINAL
from tts.parser import parse_dialogue, prepare_chunks
from tts.voices import LANGUAGES, COMPARISON_TEXT, STARTER_VOICES, load_voices


class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    max_chars: int = Field(default=500, ge=100, le=1000)


class JobRequest(ParseRequest):
    voices: dict[str, str] = Field(max_length=100)
    language: str = 'German'
    pause_ms: int = Field(default=250, ge=0, le=3000)


class VoiceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=5, max_length=1000)
    ref_text: str = Field(default=COMPARISON_TEXT, min_length=10, max_length=600)
    language: str = 'German'


class RegenerateRequest(BaseModel):
    description: str | None = Field(default=None, min_length=5, max_length=1000)


def create_app(config=None, engine_factory=SubprocessEngine):
    config = config or Config()

    @asynccontextmanager
    async def lifespan(app):
        with application_lock(config.cache):
            app.state.manager = JobManager(config, engine_factory)
            try:
                if config.bootstrap_voices and not any(config.voices.glob('*/voice.json')):
                    for voice_id, name, description in STARTER_VOICES:
                        app.state.manager.submit_design(voice_id, name, description, COMPARISON_TEXT, 'German')
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
        with app.state.manager.condition:
            library, errors = load_voices(config.voices)
            return {'voices': [voice.public() for voice in library.values()], 'errors': errors,
                    'comparison_text': COMPARISON_TEXT}

    def enqueue_design(voice_id, name, description, text, language):
        if language not in LANGUAGES or not name.strip() or not description.strip() or not text.strip():
            raise HTTPException(422, 'Provide a name, description, reference text and supported language.')
        try:
            return app.state.manager.submit_design(voice_id, name.strip(), description, text, language)
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/voices', status_code=202)
    def create_voice(body: VoiceRequest):
        return enqueue_design(uuid.uuid4().hex, body.name, body.description, body.ref_text, body.language)

    @app.post('/voices/{voice_id}/regenerate', status_code=202)
    def regenerate_voice(voice_id: str, body: RegenerateRequest):
        with app.state.manager.condition:
            library, _ = load_voices(config.voices)
            if voice_id not in library:
                raise HTTPException(404, 'Unknown voice.')
            voice = library[voice_id]
            description = body.description or voice.description
            if not description:
                raise HTTPException(422, 'Provide a description to regenerate this imported voice.')
            return enqueue_design(voice.id, voice.name, description, voice.ref_text, voice.language)

    @app.get('/voices/{voice_id}/audio')
    def voice_audio(voice_id: str):
        with app.state.manager.condition:
            library, _ = load_voices(config.voices)
            if voice_id not in library:
                raise HTTPException(404, 'Unknown voice.')
            return FileResponse(library[voice_id].reference, media_type='audio/wav',
                                headers={'Cache-Control': 'no-cache'})

    @app.post('/parse')
    def parse_text(body: ParseRequest):
        turns, speakers, chunks = parse(body)
        return {'speakers': speakers, 'turns': len(turns), 'total': len(chunks)}

    @app.post('/jobs', status_code=202)
    def submit(body: JobRequest):
        _, speakers, chunks = parse(body)
        if body.language not in LANGUAGES:
            raise HTTPException(422, 'Unsupported language.')
        if not shutil.which('ffmpeg'):
            raise HTTPException(503, 'ffmpeg is required for MP3 export. Install it and retry.')
        try:
            with app.state.manager.condition:
                library, _ = load_voices(config.voices)
                if set(body.voices) != set(speakers):
                    raise HTTPException(422, 'Assign a voice to every detected speaker, then submit again.')
                if any(voice not in library for voice in body.voices.values()):
                    raise HTTPException(422, 'Selected voice is missing or invalid. Refresh the voice library.')
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

    @app.post('/jobs/{job_id}/retry', status_code=202)
    def retry_job(job_id: str):
        with app.state.manager.condition:
            status = snapshot(job_id)
            if status['state'] not in TERMINAL:
                raise HTTPException(409, 'Wait for this job to finish before retrying.')
            job = app.state.manager.jobs[job_id]
            if job.kind == 'voice':
                d = job.design
                return enqueue_design(d['id'], d['name'], d['description'], d['ref_text'], d['language'])
            try:
                return app.state.manager.submit(job.chunks, job.voices, job.language, job.pause_ms)
            except OverflowError as exc:
                raise HTTPException(429, str(exc)) from exc

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
