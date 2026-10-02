# Speakerpy

Local dialogue-to-speech studio for a Mac. Paste labelled dialogue, assign voices,
queue synthesis, then play or download WAV and MP3. FastAPI, vanilla JavaScript,
SSE and Qwen3-TTS; no Docker, database or frontend build step.

## Start

Install Python 3.12 and ffmpeg (`brew install python@3.12 ffmpeg` on macOS), then:

```sh
./run.sh
```

The launcher creates `.venv`, installs dependencies with **pip** via **Invoke**,
starts one server, and opens [localhost:8000](http://127.0.0.1:8000) once ready.
Press **Ctrl+C** to stop it and its active synthesis worker. Subsequent launches
reuse the virtual environment and downloaded model weights.

Equivalent manual setup:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install invoke
invoke setup
invoke run --browser
# Or: python app.py
```

On an empty first startup, three German samples are queued: a warm calm voice,
a bright energetic voice, and a deep textured narrator. All read the same text.
The first generation downloads `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`; dialogue
synthesis downloads `Qwen/Qwen3-TTS-12Hz-1.7B-Base`. Allow several GB of disk space
per model and time for the first download. The browser stays usable meanwhile.
Internet is needed for dependencies and initial weights; generation then runs
locally. Model files are stored in `cache/huggingface/` (`HF_HOME` can override it).

## Voices and dialogue

- Listen to each reference in **Voice library**. Edit its characteristics and
  choose **Regenerate sample**, or use **Add a voice**. These jobs share the
  conversation queue, so only one model runs at any moment.
- The library is discovered from `voices/<id>/reference.wav` and `voice.json`.
  No fixed voice registry exists. See [voice import instructions](voices/README.md).
- Regeneration preserves the old sample until synthesis succeeds, and keeps one
  previous pair on disk. Queued conversations snapshot references, so later voice
  changes cannot silently alter their output. New references invalidate caches.
- The three starter descriptions are only seed instructions for an empty folder.
  Existing libraries are used as-is on restart. Failed startup jobs have **Retry**
  controls; an entirely empty library is seeded again on the next startup.
- `python pregen.py --name Anna --description "A warm German speaker"` queues a
  voice through the running server's API, just like the browser.

```text
Sprecher 1: Hallo!
Sprecher 2: Guten Abend.
Sprecher 1: Ein längerer Beitrag kann
auf der nächsten Zeile weitergehen.
  Hinweis: Zeilen mit einem Doppelpunkt als Fortsetzung einrücken.
```

Every unindented `Name: text` line starts a new turn, even when the name repeats.
Unlabelled continuation lines belong to the preceding speaker. Blank lines retain
paragraphs; empty turns are rejected. Chunking prefers paragraphs, then sentences,
then words, with a hard 500-character default. Chunks never cross turns. The pause
(default 250 ms) is added **between turns**, not between chunks of a long turn.
**Randomize** assigns distinct voices and explains when more voices are needed.

## Memory and job lifecycle

A single FIFO thread handles both dialogue and voice-design jobs. Torch and Qwen
are imported **only in its child process**. By default that process exits after
every chunk, releasing model, prompt and accelerator memory. Larger batches reuse
one model and per-voice clone prompts, then recycle it:

```sh
TTS_RECYCLE_CHUNKS=4 ./run.sh
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `TTS_RECYCLE_CHUNKS` | `1` | Chunks per subprocess (1–20); larger is faster but holds more memory |
| `TTS_DEVICE` | `auto` | MPS when available, otherwise CPU; accepts `mps` or `cpu` |
| `TTS_MPS_MEMORY_FRACTION` | `0.6` | PyTorch MPS allocation limit; not a total-system RAM guarantee |
| `TTS_CHUNK_TIMEOUT` | `1800` | Seconds per chunk including model download/loading |
| `TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Clone model; the `0.6B-Base` model can reduce memory further |

MPS uses float16, CPU uses float32, and attention uses SDPA. Weights are loaded and
cast on CPU before moving to MPS, avoiding duplicate checkpoint/cast allocations
on the GPU during loading. No FlashAttention
installation is required. Voice design always uses its separate 1.7B model.
Each inference is limited to 2048 new tokens. Very long/unusual utterances may
need smaller chunks or edited text. A subprocess limits accumulated memory;
it cannot make a model's peak memory requirements disappear.

The queue accepts up to ten waiting jobs. Cancellation terminates and reaps the
active worker; timeout/failure also frees it before the next job starts. ffmpeg
export is cancellable. Assembly streams bounded blocks rather than loading the
whole conversation into memory. Jobs that fail during MP3 export retain the WAV.

Job history is in memory (last 100); restarting clears the queue and history.
Completed output stays in `output/<job-id>/`. Completed chunks stay in `cache/`;
resubmitting the same dialogue reuses them after interruption. Cache identities
include model, reference content/metadata, voice ID, language, text and settings.
Failed/partial WAVs are never cache hits. Retry uses the job's original references;
submit a new conversation to use a newly regenerated voice.

Disk cache/output are not automatically deleted. Stop the app before removing
chunk WAVs from `cache/` or old output. Keep `cache/huggingface/` to avoid model
downloads and keep `voices/` to preserve your library. If changing model
weights at the same model path/ID or upgrading synthesis behavior, clear chunk
WAV caches. An OS file lock prevents two servers using the same project cache;
run **one Uvicorn worker**, without reload. Separate checkouts/caches are separate
apps and should not run inference simultaneously on the same Mac.

## API and tests

Interactive API documentation: [localhost:8000/docs](http://127.0.0.1:8000/docs).

| Endpoint | Purpose |
| --- | --- |
| `POST /parse` | Validate text and return speakers, turns, total chunks |
| `GET /voices` | Discover disk voices and report invalid folders |
| `POST /voices` | Queue voice design (`name`, `description`, optional `ref_text`, `language`) |
| `GET /voices/{id}/audio` | Play/download the reference WAV |
| `POST /voices/{id}/regenerate` | Queue replacement; optional new `description` |
| `POST /jobs` | Queue dialogue synthesis |
| `GET /jobs`, `GET /jobs/{id}` | Queue and progress snapshots |
| `GET /jobs/{id}/events` | SSE snapshots, including current speaker/voice/text |
| `POST /jobs/{id}/cancel`, `POST /jobs/{id}/retry` | Cancel or retry |
| `GET /jobs/{id}/audio/wav`, `.../mp3` | Available outputs |
| `GET /jobs/{id}/log` | Worker diagnostics |

```sh
curl -X POST http://127.0.0.1:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"text":"A: Hallo!\nB: Guten Abend!","voices":{"A":"clara","B":"felix"},"language":"German","pause_ms":250}'
```

```sh
# Automated tests: injected model doubles, real child processes and real ffmpeg.
.venv/bin/python -m invoke test

# Real synthesis through HTTP, after a sample is ready and the app is running:
.venv/bin/python -m invoke smoke
```

For lightweight API development, `invoke setup --lite` installs no Torch/Qwen.
Tests run with temporary reference files and fake synthesis; no model download or
GPU is needed. They cover speaker boundaries, queue serialization, responsiveness,
cancellation, child recycling/timeouts/crashes, prompt reuse, reference changes,
first-startup voices, regeneration, cache recovery, silence placement, SSE and
MP3/WAV downloads. The original `test.py` is a legacy PDF prototype, **not** the
test suite or application entry point; its Docling dependencies are not installed.
