import argparse
import os
import random
import resampy
import soundfile as sf
from pathlib import Path
from glob import iglob, glob
from utils.utils import seed_everything


def _get_bgs(args) -> list:
    """
    This function returns a list of file path, with the first being the true one, and the rest being distractors.
    args.distract_method determines which approach is used.
    """
    if args.distract_method == 'in_domain':
        domain = random.sample(os.listdir(args.bg_path), 1)[0]
        return random.sample(glob(f'{args.bg_path}/{domain}/**/*.{args.ext}', recursive=True), args.n_options)

    return random.sample(glob(f'{args.bg_path}/**/*.{args.ext}', recursive=True), args.n_options)


def generate_all_samples(args):
    files = iglob(f'{args.data_path}/**/*.{args.ext}', recursive=True)
    for j, f in enumerate(files):
        if j >= args.n_samples:
            return
        generate_sample(f, j, args)


def generate_sample(gt_path, ind, args):
    wav, sr = sf.read(gt_path, dtype='float32')
    bg_paths = _get_bgs(args)

    gt_bg, _sr = sf.read(bg_paths[0], dtype='float32')
    gt_bg = resampy.resample(gt_bg, _sr, sr)
    assert len(gt_bg) >= len(wav), "Background is too short"
    wav[:int(args.start_ratio*len(wav))] += gt_bg[:int(args.start_ratio*len(wav))]  # TODO: better address SNR and clipping

    for i, bg in enumerate(bg_paths):
        bg_wav, _sr = sf.read(bg, dtype='float32')
        bg_wav = resampy.resample(bg_wav, _sr, sr)
        assert len(bg_wav) >= len(wav), "Background is too short"
        out_wav = wav.copy()
        out_wav[int(args.start_ratio*len(wav)):] += bg_wav[int(args.start_ratio*len(out_wav)):len(out_wav)]
        sf.write(f'{args.out_path}/bg-cons_{ind}_{i}.wav', out_wav, sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../audio/speech', help='Path to foreground samples, can have hierarchy')
    parser.add_argument('--bg_path', default='../audio/background_noises', help='Path to background samples, can have hierarchy')
    parser.add_argument('--out_path', default='../generated_audio/background_consistency/', help='Path to save outputs')
    parser.add_argument('--ext', default='wav', help='Path to save outputs')
    parser.add_argument('--n_samples', default=10, type=int, help='Number of samples for benchmark')
    parser.add_argument('--n_options', default=4, type=int, help='Number of options in each sample including the true')
    parser.add_argument('--distract_method', default='in_domain', help='Which method to use for generating distractors from [in_domain, random]')
    parser.add_argument('--start_ratio', default=0.7, type=float, help='Number of options in each sample including the true')
    parser.add_argument('--seed', default=42, type=int, help='random seed, use -1 for non-determinism')
    args = parser.parse_args()

    os.makedirs(Path(args.out_path), exist_ok=True)
    seed_everything(args.seed)

    generate_all_samples(args)
