from __future__ import annotations

import sys
import csv
from pathlib import Path
from glob import iglob, glob
import shutil
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.general import regex_rglob
from utils.audio import safe_load_audio, get_length, safe_write_audio, concat_audio, load_audio

BACKGROUND_NOISE_DATASET_PATH = Path("../audio/background_noises")


class EmotionDataset:

    EMOTIONAL_DATASET_PATH = Path("../audio/emotion_prompts")
    EMOTION_DICT = {"neutral": "01-01", "happy": "03-02", "sad": "04-02"}
    TEXT_DICT = {"kids": "01", "dogs": "02", "concat": "03"}

    def __init__(self):
        self.emotions = list([p.name for p in EmotionDataset.EMOTIONAL_DATASET_PATH.glob("*/") if p.is_dir()])

    def get_emotions(self):
        return self.emotions

    def get_audio_path(self, emotion: str, speaker: str | int = None, text: str = None):
        """
                emotion: the emotion of the speech
                speaker: can be [1-24, male, female, None]
                        1-24 will bring the desired speaker by index
                        male or female will sample randomly between the males / females
                        None will sample randomly from all speakers
                text: can be "dogs" or "kids". None will sample randomly
            """

        if emotion in self.emotions:
            emotion_code = EmotionDataset.EMOTION_DICT[emotion]
        else:
            raise Exception(f"{emotion} is not a valid emotion. choose an emotion from {self.emotions}")

        if text in EmotionDataset.TEXT_DICT.keys():
            text_code = EmotionDataset.TEXT_DICT[text]
        elif text is None:
            text_code = random.choice(EmotionDataset.TEXT_DICT.values())
        else:
            raise Exception(f"{text} is not a valid text. choose from: {EmotionDataset.TEXT_DICT.keys()}")

        if isinstance(speaker, int) and 1 <= speaker <= 24:
            speaker_code = '{:02d}'.format(speaker)
        elif speaker == "male":
            speaker_id = random.choice(range(1, 24, 2))
            speaker_code = '{:02d}'.format(speaker_id)
        elif speaker == "female":
            speaker_id = random.choice(range(2, 25, 2))
            speaker_code = '{:02d}'.format(speaker_id)
        elif speaker == None:
            speaker_id = random.choice(range(1, 25))
            speaker_code = '{:02d}'.format(speaker_id)
        else:
            raise Exception(
                f"{speaker} is not a valid speaker. choose a speaker between [1-24], specify 'male' or 'female', or choose randomly using 'None'")

        wav_path = EmotionDataset.EMOTIONAL_DATASET_PATH / emotion / f"03-01-{emotion_code}-{text_code}-01-{speaker_code}.wav"
        return wav_path


class BackgroundDataset:

    def __init__(self):
        self.classes = list([p.name for p in BACKGROUND_NOISE_DATASET_PATH.glob("*/") if p.is_dir()])

    def get_classes(self):
        return self.classes

    def get_audio_path(self, noise_class):
        if noise_class not in self.classes:
            raise Exception(f"{noise_class} is not a valid background noise class. choose a class from {classes}")

        all_class = (BACKGROUND_NOISE_DATASET_PATH / noise_class).glob("*.wav")
        wav_path = random.choice(all_class)

        return wav_path




class TextDataset:

    def __init__(self, txt_path):
        self.path = txt_path

        with open(txt_path, 'r') as f:
            self.lines = f.readlines()

        self.len = len(self.lines)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if item >= self.len:
            raise Exception(f"Dataset contains {self.len} lines. [0-{self.len - 1}]")

        return self.lines[item]


def parse_RAVDESS(original_dataset_path, parsed_dataset_path):
    """
        dont need to run this function, wavs appended to git.
    """
    original_dataset_path = Path(original_dataset_path)
    parsed_dataset_path = Path(parsed_dataset_path)
    neutral_dataset_path = parsed_dataset_path / "neutral"
    happy_dataset_path = parsed_dataset_path / "happy"
    sad_dataset_path = parsed_dataset_path / "sad"

    for p in [neutral_dataset_path, happy_dataset_path, sad_dataset_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    neutral_wavs = regex_rglob(original_dataset_path, "03-01-01-01-0.-01-..\.wav")
    happy_wavs = regex_rglob(original_dataset_path, "03-01-03-02-0.-01-..\.wav")
    sad_wavs = regex_rglob(original_dataset_path, "03-01-04-02-0.-01-..\.wav")

    for wav_path in neutral_wavs:
        wav = safe_load_audio(wav_path)
        safe_write_audio(neutral_dataset_path / wav_path.name, wav)

    for speaker in range(1, 25):
        wav_1 = safe_load_audio(
            parsed_dataset_path / "neutral" / "03-01-01-01-01-01-{:02d}.wav".format(speaker)
        )

        wav_2 = safe_load_audio(
            parsed_dataset_path / "neutral" / "03-01-01-01-02-01-{:02d}.wav".format(speaker)
        )

        wav_3 = concat_audio(wav_1, wav_2)

        safe_write_audio(parsed_dataset_path / "neutral" / "03-01-01-01-03-01-{:02d}.wav".format(speaker), wav_3)



    for wav_path in happy_wavs:
        wav = safe_load_audio(wav_path)
        safe_write_audio(happy_dataset_path / wav_path.name, wav)

    for speaker in range(1, 25):
        wav_1 = safe_load_audio(
            parsed_dataset_path / "happy" / "03-01-03-02-01-01-{:02d}.wav".format(speaker)
        )

        wav_2 = safe_load_audio(
            parsed_dataset_path / "happy" / "03-01-03-02-02-01-{:02d}.wav".format(speaker)
        )

        wav_3 = concat_audio(wav_1, wav_2)

        safe_write_audio(parsed_dataset_path / "happy" / "03-01-03-02-03-01-{:02d}.wav".format(speaker), wav_3)

    for wav_path in sad_wavs:
        wav = safe_load_audio(wav_path)
        safe_write_audio(sad_dataset_path / wav_path.name, wav)

    for speaker in range(1, 25):
        wav_1 = safe_load_audio(
            parsed_dataset_path / "sad" / "03-01-04-02-01-01-{:02d}.wav".format(speaker)
        )

        wav_2 = safe_load_audio(
            parsed_dataset_path / "sad" / "03-01-04-02-02-01-{:02d}.wav".format(speaker)
        )

        wav_3 = concat_audio(wav_1, wav_2)

        safe_write_audio(parsed_dataset_path / "sad" / "03-01-04-02-03-01-{:02d}.wav".format(speaker), wav_3)


def parse_FSD(original_dataset_path, parsed_dataset_path):
    original_dataset_path = Path(original_dataset_path)
    parsed_dataset_path = Path(parsed_dataset_path)

    with open(original_dataset_path, 'r') as f:
        gt_csv = csv.reader(f, delimiter=',')

        lines = list(gt_csv)
        lines_1_class = [l for l in lines if len(l[1].split(',')) == 1]

        classes = set()
        for l in lines_1_class[1:]:
            classes.add(l[1])
            p = BACKGROUND_NOISE_DATASET_PATH / l[1]
            if not os.path.exists(p):
                os.makedirs(p)

            wav = safe_load_audio(f"/cs/labs/adiyoss/amitroth/datasets/FSD50K.dev_audio/{l[0]}.wav")
            if get_length(wav) > 8:
                print(get_length(wav), l[1])
                shutil.copy(f"/cs/labs/adiyoss/amitroth/datasets/FSD50K.dev_audio/{l[0]}.wav",
                            p / f"{l[0]}.wav")

        print(classes)
        print(len(classes))


if __name__ == '__main__':
    parse_RAVDESS("/Users/amitroth/Downloads/Audio_Speech_Actors_01-24", "/Users/amitroth/PycharmProjects/slm-benchnark/audio/emotion_prompts")
    # wav = get_emotion_prompt("happy", speaker="male")

    # d = TextDataset("/Users/amitroth/PycharmProjects/slm-benchnark/txt/sentiment/happy.txt")
    # shutil.copy("/cs/labs/adiyoss/amitroth/datasets/FSD50K.dev_audio/16845.wav", "/cs/labs/adiyoss/amitroth/slm-benchmark/16845.wav")
    # parse_FSD("/cs/labs/adiyoss/amitroth/datasets/FSD50K.ground_truth/dev.csv",
    #           "/cs/labs/adiyoss/amitroth/slm-benchmark/audio/background_noises")
