import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Use a specific device map dictionary for Mac
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Targeted device: {device}")

# We use device_map as a dict to bypass the 'auto' logic that causes meta-tensor errors
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map={"": device},
    torch_dtype=torch.bfloat16
)

# Generate your master reference voice
ref_text = "Dies ist die feste Stimme für meine Ernährungsbildungs-Präsentation. Ich erläutere ihnen hiermit die komplizierten Fakten des Lebens ganz simpel und mit etwas Witz."
description = "Ein professioneller, sachlicher deutscher Sprecher mit einer warmen Stimme."

print("🎙️ Designing voice...")
wavs, sr = model.generate_voice_design(text=ref_text, instruct=description)
sf.write("german_narrator.wav", wavs[0], sr)
print("✅ Success! 'german_narrator.wav' created.")
