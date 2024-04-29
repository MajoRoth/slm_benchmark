import argparse
import os

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from utils import FastSpeech
from tqdm import tqdm

def get_text_for_class(text_class):
    texts = {
        "laugh": "that was funny",
        "waterfall": "i love the sound of water",
        "phone": "come here you have a phone call"
    }

    return texts[text_class]


def background_noise_alignment(text_class, background_noise_directory, output_path):
    fastspeech = FastSpeech()
    bg_dir_path = Path(background_noise_directory)

    bg_paths = list(bg_dir_path.glob("*.wav"))
    assert len(bg_paths) > 0

    if not os.path.exists(Path(output_path) / f"text_{text_class}"):
        os.makedirs(Path(output_path) / f"text_{text_class}")

    speech_wav, speech_sr = fastspeech.tts(get_text_for_class(text_class))

    for bg_path in tqdm(bg_paths, desc="generating audio samples"):
        bg_wav, bg_sr = sf.read(bg_path)
        bg_wav = librosa.resample(y=bg_wav, orig_sr=bg_sr, target_sr=speech_sr)

        min_length = min(len(speech_wav), len(bg_wav))

        merged_data = np.vstack((speech_wav[:min_length], bg_wav[:min_length])).T
        merged_data = merged_data.sum(axis=1) / 2
        print(merged_data)

        sf.write(Path(output_path) / f"text_{text_class}" / bg_path.name, merged_data, speech_sr)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-class",
        type=str,
        default="waterfall"
    )

    parser.add_argument(
        "--background-noise-dir",
        type=str,
        default="audio/background_noises"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_audio/background_noise_alignment/"
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    background_noise_alignment(
        text_class=args.text_class,
        background_noise_directory=os.path.join(os.path.dirname(__file__), "..", args.background_noise_dir),
        output_path=os.path.join(os.path.dirname(__file__), "..", args.output_dir)
    )
