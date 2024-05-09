import emotions as emotions
import string
from pathlib import Path
import json
import whisper
import torch
from jiwer import wer, cer
import numpy as np

from utils.audio import safe_load_audio


def clean_string(input_string):
    translation_table = str.maketrans('', '', '!@#$%^&*()_+=-1234567890`~<>?/.,;:[]{}\\|')
    cleaned_string = input_string.translate(translation_table)
    return cleaned_string


class Whisper:

    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = whisper.load_model("large-v2").to(self.device)

    def transcribe(self, audio):
        if isinstance(audio, (str, Path)):
            return self.model.transcribe(str(audio))['text']
        else:
            # raw waveform
            return self.model.transcribe(audio)['text']

    def wer(self, audio, text):
        transcribed_text = clean_string(self.transcribe(audio))
        cleaned_text = clean_string(text)
        w = wer(transcribed_text, cleaned_text)
        print(f"wer: {w} - transcribed text: {transcribed_text}")
        return w


if __name__ == '__main__':
    w = Whisper()

    # w.wer("/cs/labs/adiyoss/amitroth/slm-benchmark/utils/sample_1_sad_8_happy.wav", "I'm feeling so blessed and grateful")
    # w.wer("/cs/labs/adiyoss/amitroth/slm-benchmark/utils/sample_1_sad_22_happy.wav", "I'm feeling so blessed and grateful")