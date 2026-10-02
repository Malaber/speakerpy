"""Bounded-memory WAV assembly and cancellable ffmpeg export."""
from pathlib import Path
import subprocess
import time
import numpy as np
import soundfile as sf
from tts.engine import check_cancel


def valid_audio(path: Path) -> bool:
    try:
        info = sf.info(path)
        return info.frames > 0 and info.channels == 1 and info.samplerate > 0
    except (OSError, RuntimeError):
        return False


def assemble(paths, chunks, output, pause_ms, cancel):
    temporary = output.with_suffix('.partial.wav')
    try:
        rate = sf.info(paths[0]).samplerate
        with sf.SoundFile(temporary, 'w', samplerate=rate, channels=1, subtype='PCM_16') as master:
            previous_turn = chunks[0].turn
            for path, chunk in zip(paths, chunks):
                check_cancel(cancel)
                with sf.SoundFile(path) as source:
                    if source.samplerate != rate or source.channels != 1:
                        raise ValueError('Chunk audio formats do not match; cannot assemble safely.')
                    if chunk.turn != previous_turn:
                        master.write(np.zeros(round(rate * pause_ms / 1000), dtype=np.float32))
                    for block in source.blocks(blocksize=65536, dtype='float32'):
                        check_cancel(cancel)
                        master.write(block)
                previous_turn = chunk.turn
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def export_mp3(wav, output, cancel, log_path):
    temporary = output.with_suffix('.partial.mp3')
    process = None
    try:
        with log_path.open('a') as log:
            process = subprocess.Popen(
                ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                 '-i', str(wav), '-codec:a', 'libmp3lame', '-q:a', '2', str(temporary)],
                stdout=log, stderr=log)
            deadline = time.monotonic() + 600
            while process.poll() is None:
                check_cancel(cancel)
                if time.monotonic() > deadline:
                    raise TimeoutError('MP3 conversion timed out.')
                time.sleep(0.1)
            check_cancel(cancel)
            if process.returncode:
                raise RuntimeError('ffmpeg could not encode MP3; the WAV is still available. See job log.')
        temporary.replace(output)
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        temporary.unlink(missing_ok=True)
