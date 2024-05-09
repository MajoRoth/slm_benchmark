import torch
from TTS.api import TTS
import numpy as np

from utils.dataset import EmotionDataset
from utils.wer import Whisper

from utils.audio import safe_write_audio, write_audio



class Xtts:

    SR = 24000

    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.whisper = None

    def tts_to_file(self, text, audio_prompt_path, output_path):
        self.model.tts_to_file(text=text, speaker_wav=str(audio_prompt_path), language="en", file_path=str(output_path))

    def tts(self, text, audio_prompt_path):
        return self.model.tts(text=text, speaker_wav=str(audio_prompt_path), language="en")

    def get_whisper(self):
        if self.whisper is None:
            self.whisper = Whisper()
        return self.whisper

    def top_wer_sampling(self, text, audio_prompt_path, output_path, k):
        """
            samples from XTTS k times and chooses the recording with the best wer.
        """
        whisper = self.get_whisper()
        samples_scores = list()

        for i in range(k):
            sample = self.tts(text, audio_prompt_path)
            sample = np.array(sample).astype(np.float32)
            w = whisper.wer(sample, text)
            samples_scores.append(list((sample, w)))

        wav, _ = min(samples_scores, key=lambda pair: pair[1])
        write_audio(output_path, wav, Xtts.SR)
        return output_path



if __name__ == '__main__':
    tts = Xtts()

    emotions = EmotionDataset()
    p = emotions.get_audio_path("happy", speaker=2, text="dogs")
    tts.top_wer_sampling(
        "I received a random act of kindness from someone today.", p, "/cs/labs/adiyoss/amitroth/slm-benchmark/best.wav", k=3
    )

