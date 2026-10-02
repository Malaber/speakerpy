"""Reference library validation and content-addressed voice identities."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re

LANGUAGES = ("German", "English", "Chinese", "Japanese", "Korean", "French",
             "Russian", "Portuguese", "Spanish", "Italian", "Auto")

COMPARISON_TEXT = (
    "Hallo und herzlich willkommen! Heute entdecken wir gemeinsam etwas Neues. "
    "Draußen scheint die Sonne, und auf dem Marktplatz treffen sich viele Menschen. "
    "Hast du einen Moment Zeit? Dann hör genau zu: Jede Stimme erzählt ihre eigene Geschichte."
)

# Only initial creation instructions. The library itself is always scanned from disk.
STARTER_VOICES = (
    ('clara', 'Clara', 'Eine erwachsene deutsche Sprecherin mit warmer, ruhiger, tiefer Stimme. '
     'Sie spricht deutlich, freundlich und entspannt.'),
    ('felix', 'Felix', 'Ein erwachsener deutscher Sprecher mit heller, lebhafter Stimme. '
     'Er spricht fröhlich, dynamisch und neugierig.'),
    ('theo', 'Theo', 'Ein älterer deutscher Sprecher mit tiefer, leicht rauer Stimme. '
     'Er spricht gelassen und ausdrucksstark wie ein erfahrener Geschichtenerzähler.'),
)


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    language: str
    ref_text: str
    reference: str
    fingerprint: str
    description: str = ''

    def public(self):
        return {key: value for key, value in asdict(self).items()
                if key in {"id", "name", "language", "description", "ref_text", "fingerprint"}}


def load_voices(root: Path) -> tuple[dict[str, Voice], list[str]]:
    voices, errors = {}, []
    for metadata in sorted(root.glob("*/voice.json")):
        voice_id = metadata.parent.name
        if voice_id.startswith('.'):
            continue  # In-progress generation directories are not library entries.
        try:
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", voice_id):
                raise ValueError("folder name must contain only letters, numbers, _ or -")
            data = json.loads(metadata.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError('voice.json must contain a JSON object')
            for field in ("name", "ref_text", "language"):
                if not isinstance(data.get(field), str) or not data[field].strip():
                    raise ValueError(f"{field} must be a non-empty string")
            if data["language"] not in LANGUAGES:
                raise ValueError("unsupported language")
            reference = metadata.parent / "reference.wav"
            # Read in blocks: reference recordings need not fit in server memory.
            digest = hashlib.sha256(metadata.read_bytes())
            with reference.open("rb") as audio:
                for block in iter(lambda: audio.read(65536), b""):
                    digest.update(block)
            # SoundFile supports PCM and floating-point WAV references.
            import soundfile as sf
            info = sf.info(reference)
            if info.frames < 1 or info.duration > 60 or info.channels not in (1, 2):
                raise ValueError("reference must be nonempty mono/stereo WAV, at most 60 seconds")
            voices[voice_id] = Voice(voice_id, data["name"], data["language"],
                                    data["ref_text"], str(reference.resolve()), digest.hexdigest(),
                                    str(data.get('description', '')))
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            errors.append(f"{voice_id}: {exc}")
    return voices, errors


def cache_key(model: str, voice: Voice, language: str, text: str, settings: dict) -> str:
    identity = {"schema": 1, "model": model, "voice": voice.id,
                "reference": voice.fingerprint, "language": language,
                "text": text, "settings": settings}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
