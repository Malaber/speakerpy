"""Private line protocol. This process alone owns torch/model/prompt memory."""
from contextlib import redirect_stdout
import gc
import json
from pathlib import Path
import sys
import traceback


def main():
    protocol = sys.stdout
    model, prompts = None, {}
    # Third-party libraries print diagnostics; reserve stdout for protocol replies.
    with redirect_stdout(sys.stderr):
        for line in sys.stdin:
            target = None
            try:
                request = json.loads(line)
                import numpy as np
                import soundfile as sf
                import torch
                from qwen_tts import Qwen3TTSModel
                if model is None:
                    torch.set_num_threads(2)
                    device = request['device']
                    if device == 'auto':
                        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
                    if device == 'mps':
                        torch.mps.set_per_process_memory_fraction(request['mps_fraction'])
                    dtype = torch.float16 if device == 'mps' else torch.float32
                    model = Qwen3TTSModel.from_pretrained(
                        request['model'], device_map={'': device}, dtype=dtype,
                        attn_implementation='sdpa')
                voice = request['voice']
                with torch.inference_mode():
                    key = voice['fingerprint']
                    if key not in prompts:
                        prompts[key] = model.create_voice_clone_prompt(
                            ref_audio=voice['reference'], ref_text=voice['ref_text'],
                            x_vector_only_mode=False)
                    wavs, rate = model.generate_voice_clone(
                        text=request['text'], language=request['language'],
                        voice_clone_prompt=prompts[key], **request['settings'])
                data = wavs[0]
                if isinstance(data, torch.Tensor):
                    data = data.detach().float().cpu().numpy()
                data = np.asarray(data).reshape(-1)
                if data.size == 0 or not np.isfinite(data).all():
                    raise ValueError('Model returned empty or non-finite audio.')
                target = Path(request['target'])
                temporary = target.with_suffix('.partial.wav')
                sf.write(temporary, data, rate, subtype='PCM_16')
                temporary.replace(target)
                del wavs, data
                gc.collect()
                if device == 'mps':
                    torch.mps.empty_cache()
                protocol.write(json.dumps({'ok': True}) + '\n')
                protocol.flush()
            except Exception as exc:
                traceback.print_exc()
                if target:
                    target.with_suffix('.partial.wav').unlink(missing_ok=True)
                protocol.write(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
                protocol.flush()
                return


if __name__ == '__main__':
    main()
