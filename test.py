import argparse
import sys
import io
import re
import gc
import os
import hashlib
import subprocess
from pathlib import Path

import torch
import requests
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Optional: for memory monitoring
try:
    import psutil
except ImportError:
    psutil = None

# Suppress "Setting `pad_token_id` to `eos_token_id`" warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import transformers
transformers.logging.set_verbosity_error()


def clean_markdown_for_tts(text):
    """Strips Markdown, HTML comments, and PDF artifacts for a smoother listening experience."""
    # 1. Remove HTML comments
    text = re.sub(r'', '', text, flags=re.DOTALL)

    # 2. Markdown formatting removal
    text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # Remove Header hashes
    text = re.sub(r'#+\s', '', text)
    # Remove links [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'[-*_]{3,}', '', text)
    # Remove HTML image comments
    text = text.replace("<!-- image -->", "")
    # Remove unknown chars
    text = text.replace("/uniF6B7", "")

    # --- PDF Artifact Cleanup ---

    text = text.replace("­ ", "")
    text = text.replace("  ", " ")

    # 3. Remove invisible soft hyphens (\xad)
    text = text.replace('\xad', '')

    # 4. Fix line-break hyphenation ("Lebensmittelzu- bereitung" -> "Lebensmittelzubereitung")
    # Matches a word char, a hyphen, 1+ whitespace chars (including newlines), and a word char
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # 5. Fix multiple horizontal spaces (justified text artifacts)
    # Using [ \t]+ instead of \s+ so we don't destroy \n\n paragraph breaks!
    text = re.sub(r'[ \t]+', ' ', text)

    # 6. Fix hard-wrapped lines inside paragraphs
    # Replaces single newlines with a space, but leaves double newlines (\n\n) alone
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Adaptive PDF to Speech Pipeline")
    parser.add_argument("input_file", help="Path to the PDF file")

    # Mode Toggle
    parser.add_argument("--mode", choices=["api", "local"], default="local", help="Synthesize via API or locally")

    # API Settings
    parser.add_argument("--url", default="http://localhost:5002/api/tts", help="Coqui TTS API URL")
    parser.add_argument("--language", default="German", help="Language for local (German) or API (de)")

    # Local Settings
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Local Qwen3 model path")
    parser.add_argument("--ref_audio", default="german_narrator.wav", help="Fixed reference voice for local mode")
    parser.add_argument("--ref_text", default="Dies ist die feste Stimme für meine Ernährungsbildungs-Präsentation. Ich erläutere ihnen hiermit die komplizierten Fakten des Lebens ganz simpel und mit etwas Witz.",
                        help="Transcription of ref_audio")
    parser.add_argument("--no-subprocess", action="store_true", help="Run local synthesis in-process (uses more RAM)")

    # Processing Settings
    parser.add_argument("--max_chars", type=int, default=500, help="Wait for this many chars before chunking")
    parser.add_argument("--snippet", type=int, nargs='?', const=3, help="Only process the first N chunks")
    parser.add_argument("--export-chunks", action="store_true", help="Save the chunks to a file and exit")

    # Internal Worker Settings
    parser.add_argument("--chunk-worker", action="store_true", help="Internal: Run as a single-chunk worker")
    parser.add_argument("--chunk-output", help="Internal: Output path for the chunk worker")

    args = parser.parse_args()

    # --- Worker Mode Handler ---
    if args.chunk_worker:
        from qwen_tts import Qwen3TTSModel
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.bfloat16
        
        # Suppress warnings in worker
        logging.getLogger("transformers").setLevel(logging.ERROR)
        transformers.logging.set_verbosity_error()

        model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map={"": device},
            torch_dtype=dtype,
            attn_implementation="sdpa"
        )
        if hasattr(model.model, "config"):
            model.model.config.pad_token_id = model.model.config.eos_token_id
            
        chunk_text = sys.stdin.read().strip()
        if not chunk_text:
            sys.exit(0)
            
        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(
                text=chunk_text,
                language=args.language,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text
            )
            
        if isinstance(wavs[0], torch.Tensor):
            data = np.array(wavs[0].detach().cpu().numpy(), copy=True)
        else:
            data = np.array(wavs[0], copy=True)
            
        sf.write(args.chunk_output, data, sr)
        sys.exit(0)

    input_path = Path(args.input_file)
    md_path = input_path.with_suffix(".md")

    suffix = "-snippet.wav" if args.snippet is not None else ".wav"
    audio_output = input_path.parent / (input_path.stem + suffix)

    # --- Step 1: Text Extraction ---
    if md_path.exists():
        print(f"♻️  Found cached Markdown: {md_path.name}")
        with open(md_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        text_to_read = clean_markdown_for_tts(raw_text)
    else:
        print(f"🔍 Extracting PDF text...")
        from docling.document_converter import DocumentConverter
        doc_converter = DocumentConverter()
        doc_result = doc_converter.convert(str(input_path))
        raw_text = doc_result.document.export_to_markdown()
        text_to_read = clean_markdown_for_tts(raw_text)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(raw_text) # Save raw text for caching
        
        # 🗑️ KILL Docling immediately to free up baseline RAM
        del doc_result
        del doc_converter
        gc.collect()

    # --- Step 2: Buffered Chunking (with sub-splitting for large blocks) ---
    def split_large_text(text, max_chars, separators):
        """Recursively splits a text block using a list of separators."""
        if len(text) <= max_chars:
            return [text]
        if not separators:
            # Fallback: split by words
            words = text.split()
            res, curr = [], ""
            for w in words:
                if len(curr) + len(w) + 1 <= max_chars:
                    curr += (w + " ") if curr else w
                else:
                    if curr: res.append(curr)
                    curr = w
            if curr: res.append(curr)
            return res

        sep_type = separators[0]
        if sep_type == "\n":
            parts = text.split("\n")
            joiner = "\n"
        elif sep_type == "SENTENCE":
            parts = re.split(r'(?<=[.!?])\s+', text)
            joiner = " "
        else:
            parts = [text]
            joiner = ""

        res, curr = [], ""
        for p in parts:
            p = p.strip()
            if not p: continue
            if len(p) > max_chars:
                if curr: res.append(curr)
                res.extend(split_large_text(p, max_chars, separators[1:]))
                curr = ""
                continue
            if not curr:
                curr = p
            elif len(curr) + len(p) + len(joiner) <= max_chars:
                curr += joiner + p
            else:
                res.append(curr)
                curr = p
        if curr: res.append(curr)
        return res

    paragraphs = [p.strip() for p in text_to_read.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        # Correctly check if the COMBINED length fits the limit
        combined_len = len(current_chunk) + len(p) + (2 if current_chunk else 0)
        if combined_len <= args.max_chars:
            current_chunk += ("\n\n" if current_chunk else "") + p
        else:
            # Flush the current buffer first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # If the single paragraph itself is too large, split it sub-sectionally
            if len(p) > args.max_chars:
                sub_chunks = split_large_text(p, args.max_chars, ["\n", "SENTENCE"])
                # Add all but the last sub-chunk directly
                for sc in sub_chunks[:-1]:
                    chunks.append(sc.strip())
                # Keep the last sub-chunk in the buffer to potentially join with next paragraph
                current_chunk = sub_chunks[-1]
            else:
                current_chunk = p

    if current_chunk: chunks.append(current_chunk.strip())
    if args.snippet is not None: chunks = chunks[:args.snippet]

    if args.export_chunks:
        chunks_file = input_path.with_suffix(".chunks.txt")
        print(f"📝 Exporting {len(chunks)} chunks to: {chunks_file}")
        oversized = []
        limit_buffer = args.max_chars * 1.1
        with open(chunks_file, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks):
                chunk_len = len(chunk)
                f.write(f"--- CHUNK {idx} ({chunk_len} chars) ---\n")
                f.write(chunk)
                f.write("\n\n")
                if chunk_len > limit_buffer:
                    oversized.append((idx, chunk_len))
        
        if oversized:
            print(f"⚠️  WARNING: {len(oversized)} chunks exceed the 10% buffer ({int(limit_buffer)} chars):")
            for idx, length in oversized:
                print(f"   - Chunk {idx}: {length} chars")
        
        print("✅ Export complete. Exiting.")
        sys.exit(0)

    # --- Step 3: Initialization (Orchestrator Mode) ---
    model = None
    if args.mode == "local":
        if args.no_subprocess:
            from qwen_tts import Qwen3TTSModel
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"🚀 Loading Local Model on {device} (In-Process)...")
            dtype = torch.float16 if device == "mps" else torch.bfloat16
            model = Qwen3TTSModel.from_pretrained(
                args.model,
                device_map={"": device},
                torch_dtype=dtype,
                attn_implementation="sdpa"
            )
            if hasattr(model.model, "config"):
                model.model.config.pad_token_id = model.model.config.eos_token_id
        else:
            print(f"🚀 Using Subprocess Synthesis to manage RAM.")
    else:
        print(f"🌐 Using API at {args.url}...")

    # --- Step 4: Synthesis Loop (Resumable + Memory Optimized) ---
    print(f"🎙️ Starting synthesis. Mode: {args.mode}")

    # Create a dedicated cache directory for this specific PDF
    cache_dir = input_path.parent / f"{input_path.stem}_audio_cache"
    cache_dir.mkdir(exist_ok=True)
    print(f"📁 Using cache directory: {cache_dir}")

    first_chunk = True
    output_file = None

    try:
        pbar = tqdm(chunks, desc=f"Synthesizing ({args.mode})")
        for i, chunk_text in enumerate(pbar):
            # --- 0. Memory Monitoring ---
            if psutil:
                # RSS (Resident Set Size) matches Activity Monitor's "Memory" column
                total_rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
                stats = {"Total": f"{total_rss_gb:.1f}GB"}
                if torch.backends.mps.is_available():
                    # Only tracks current tensors
                    mps_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
                    stats["MPS-Alloc"] = f"{mps_gb:.1f}GB"
                pbar.set_postfix(stats)

            # Create a unique MD5 hash for this exact text snippet
            text_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
            # We include the index (i:04d) so the files sort alphabetically
            chunk_file_path = cache_dir / f"chunk_{i:04d}_{text_hash}.wav"

            # --- 1. Generation or Cache Loading ---
            if chunk_file_path.exists():
                # ♻️ CACHE HIT: Load the existing audio from disk
                data, sr = sf.read(chunk_file_path)
            else:
                # ⚙️ CACHE MISS: Generate the audio
                if args.mode == "api":
                    payload = {'text': chunk_text, 'language_id': 'de'}
                    response = requests.get(args.url, params=payload, timeout=300)
                    if response.status_code == 200:
                        data, sr = sf.read(io.BytesIO(response.content))
                    else:
                        print(f"\nAPI Error on chunk {i}. Skipping...")
                        continue
                else:
                    # ⚙️ SUBPROCESS WORKER: Call ourselves to process this single chunk
                    cmd = [
                        sys.executable, __file__, str(input_path),
                        "--mode", "local",
                        "--model", args.model,
                        "--ref_audio", args.ref_audio,
                        "--ref_text", args.ref_text,
                        "--language", args.language,
                        "--chunk-worker",
                        "--chunk-output", str(chunk_file_path)
                    ]
                    # Pass chunk text via stdin to avoid shell limits
                    subprocess.run(cmd, input=chunk_text.encode('utf-8'), check=True, capture_output=True)
                    data, sr = sf.read(chunk_file_path)

            # --- 2. Stream to Final Master File ---
            if first_chunk:
                # On the first chunk, create the file and define the format
                output_file = sf.SoundFile(audio_output, mode='w', samplerate=sr,
                                           channels=1, subtype='PCM_16')
                output_file.write(data)
                first_chunk = False
            else:
                # On later chunks, just write data (metadata is already set)
                output_file.write(data)

            # --- 3. Clean up Orchestrator Memory ---
            if 'data' in locals():
                del data
            
            # Force Python to destroy unreferenced objects NOW
            gc.collect()

        if output_file:
            output_file.close()
            print(f"✨ Success! Master audio saved to: {audio_output}")
            print(f"💾 Individual snippets preserved in: {cache_dir}")

    except Exception as e:
        if output_file: output_file.close()
        print(f"\nPipeline failed: {e}")
        print("Don't worry, your progress is saved in the cache directory. Just run the script again!")
        sys.exit(1)

        if output_file:
            output_file.close()
            print(f"✨ Success! Master audio saved to: {audio_output}")
            print(f"💾 Individual snippets preserved in: {cache_dir}")

    except Exception as e:
        if output_file: output_file.close()
        print(f"\nPipeline failed: {e}")
        print("Don't worry, your progress is saved in the cache directory. Just run the script again!")
        sys.exit(1)


if __name__ == "__main__":
    main()

