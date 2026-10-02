"""Parse explicit speaker labels before cleaning or chunking their turns."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class Chunk:
    speaker: str
    text: str
    turn: int


def clean_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
    text = text.replace("\xad", "").replace("**", "").replace("__", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text).strip()


def parse_dialogue(text: str) -> list[Turn]:
    turns: list[Turn] = []
    speaker = None
    lines: list[str] = []

    def flush():
        if speaker is not None:
            cleaned = clean_text("\n".join(lines))
            if not cleaned:
                raise ValueError(f"Empty turn for {speaker}.")
            turns.append(Turn(speaker, cleaned))

    for number, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        # Indented lines are explicit continuations, even if they contain colons.
        match = re.match(r"^([^\s:][^:]{0,79}):\s*(.*)$", line)
        if match:
            flush()
            speaker, first = match.groups()
            speaker = speaker.strip()
            lines = [first]
        elif line.strip() and speaker is None:
            raise ValueError(f"Line {number}: start with a speaker label, e.g. Sprecher 1: Hallo.")
        elif speaker is not None:
            lines.append(line.strip())
    flush()
    if not turns:
        raise ValueError("Enter at least one speaker and some dialogue.")
    return turns


def split_text(text: str, limit: int = 500) -> list[str]:
    if limit < 1:
        raise ValueError("Chunk size must be positive.")

    def split(value, level=0):
        if len(value) <= limit:
            return [value] if value else []
        if level == 0:
            parts, joiner = value.split("\n\n"), "\n\n"
        elif level == 1:
            parts, joiner = re.split(r"(?<=[.!?])\s+", value), " "
        elif level == 2:
            parts, joiner = value.split(), " "
        else:
            return [value[i:i + limit] for i in range(0, len(value), limit)]
        result, current = [], ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > limit:
                if current:
                    result.append(current)
                    current = ""
                result.extend(split(part, level + 1))
            elif current and len(current) + len(joiner) + len(part) > limit:
                result.append(current)
                current = part
            else:
                current = current + joiner + part if current else part
        if current:
            result.append(current)
        return result

    return split(text.strip())


def prepare_chunks(turns: list[Turn], limit: int = 500) -> list[Chunk]:
    return [Chunk(turn.speaker, text, index)
            for index, turn in enumerate(turns) for text in split_text(turn.text, limit)]
