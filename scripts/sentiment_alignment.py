import argparse
import json
import os
import random
from pathlib import Path
from utils.xtts import Xtts
from tqdm import tqdm
from utils.dataset import EmotionDataset, TextDataset


def sentiment_alignment(output_path, args):
    output_path = Path(output_path)

    speakers = [
        {'speaker': 3, 'prompt': 'concat'},
        {'speaker': 5, 'prompt': 'concat'},
        {'speaker': 19, 'prompt': 'concat'},
        {'speaker': 23, 'prompt': 'concat'},
        {'speaker': 2, 'prompt': 'concat'},
        {'speaker': 8, 'prompt': 'kids'},
        {'speaker': 12, 'prompt': 'concat'},
        {'speaker': 22, 'prompt': 'concat'},
        {'speaker': 24, 'prompt': 'concat'}
    ]

    sad_dataset = TextDataset("./../txt/sentiment/sad.txt")
    happy_dataset = TextDataset("./../txt/sentiment/happy.txt")

    xtts = Xtts()

    emotions_prompts_dataset = EmotionDataset()

    metadata = list()

    for i in tqdm(range(args.samples_num)):
        sad_text = sad_dataset.get_random_text()
        happy_text = happy_dataset.get_random_text()

        speaker = random.choice(speakers)
        sample_metadata = {
            "index": i,
            "speaker": speaker['speaker'],
            "prompt": speaker['prompt'],
            "prompts_paths": {},
            "sad_text": sad_text,
            "happy_text": happy_text,
            "generated_audio": {}
        }

        for emotion in emotions_prompts_dataset.get_emotions():
            prompt_path = emotions_prompts_dataset.get_audio_path(emotion=emotion, speaker=speaker['speaker'], text=speaker['prompt'])
            sample_metadata['prompts_paths'][emotion] = str(prompt_path)
            xtts.tts_to_file(happy_text, prompt_path, output_path / f"sample_{i}_{emotion}_happy.wav")  # happy text
            xtts.tts_to_file(sad_text, prompt_path, output_path / f"sample_{i}_{emotion}_sad.wav")  # sad text
            sample_metadata['generated_audio'][f"{emotion}_happy"] = str(output_path / f"sample_{i}_{emotion}_happy.wav")
            sample_metadata['generated_audio'][f"{emotion}_sad"] = str(output_path / f"sample_{i}_{emotion}_sad.wav")



        metadata.append(sample_metadata)

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f)


def eval(output_path):
    xtts = Xtts()

    dataset = EmotionDataset()

    for emotion in dataset.get_emotions():
        for speaker in range(1, 25):
            for i, t in enumerate(["this is the best day ever", "this is the worst day ever"]):
                prompt_path = dataset.get_audio_path(emotion=emotion, speaker=speaker, text='concat')
                xtts.tts_to_file(t, prompt_path, Path(output_path) / f"eval_concat_{emotion}-{speaker}-{i}.wav")



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_audio/sentiment_alignment/new"
    )

    parser.add_argument(
        "--samples-num",
        type=int,
        help="amount of samples to generate",
        default=10,
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # eval("/Users/amitroth/PycharmProjects/slm-benchnark/generated_audio/sentiment_alignment/eval")

    sentiment_alignment(
        output_path=os.path.join(os.path.dirname(__file__), "..", args.output_dir),
        args=args
    )
