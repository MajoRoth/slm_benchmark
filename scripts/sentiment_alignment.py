import argparse
import os
from pathlib import Path
from utils import Xtts
from tqdm import tqdm


def sentiment_alignment(texts, prompts_directory, output_path):
    xtts = Xtts()
    prompts = Path(prompts_directory)

    sentiment_paths = list(prompts.glob("*.wav"))
    assert len(sentiment_paths) > 0

    for i, t in enumerate(texts):
        os.makedirs(Path(output_path) / f"text_{i}")

        for sentiment_path in tqdm(sentiment_paths, desc="generating audio samples"):
            xtts.tts(t, sentiment_path, Path(output_path) / f"text_{i}" / sentiment_path.name)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompts-dir",
        type=str,
        default="audio/emotion_prompts/male"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_audio/sentiment_alignment/male"
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    sentiment_alignment(
        texts=["this is the best day ever", "this is the worst day ever"],
        prompts_directory=os.path.join(os.path.dirname(__file__), "..", args.prompts_dir),
        output_path=os.path.join(os.path.dirname(__file__), "..", args.output_dir)
    )
