import torch
from TTS.api import TTS


class Xtts:

    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def tts_to_file(self, text, audio_prompt_path, output_path):
        self.model.tts_to_file(text=text, speaker_wav=str(audio_prompt_path), language="en", file_path=str(output_path))

    def tts(self, text, audio_prompt_path):
        self.model.tts(text=text, speaker_wav=audio_prompt_path, language="en")
