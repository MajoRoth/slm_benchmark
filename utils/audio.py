import soundfile as sf
import numpy as np
import librosa

GENERAL_SAMPLING_RATE = 16000


def safe_load_audio(path):
    """
        normalizes and resamples audio on load
    """
    data, sr = load_audio(path)

    if sr != GENERAL_SAMPLING_RATE:
        data = resample(data, sr, GENERAL_SAMPLING_RATE)

    data = normalize_audio(data)

    return data


def safe_write_audio(path, data):
    """
        note that this functions assumes sr == GENERAL_SAMPLING_RATE
    """
    write_audio(path, data, GENERAL_SAMPLING_RATE)

def get_length(audio):
    """
        assumes GENERAL_SAMPLING_RATE
    """
    return len(audio) / GENERAL_SAMPLING_RATE


def merge_audio(data_1, data_2):
    """
        audio should be in same sampling rates
    """
    min_length = min(len(data_1), len(data_2))

    merged_data = np.vstack((data_1[:min_length], data_2[:min_length])).T
    merged_data = merged_data.sum(axis=1) / 2

    return merged_data


def concat_audio(data_1, data_2):
    """
        audio should be in same sampling rates
    """
    combined_audio = np.concatenate((data_1, data_2))
    return combined_audio


def resample(y, orig_sr, target_sr):
    return librosa.resample(y=y, orig_sr=orig_sr, target_sr=target_sr)


def load_audio(path):
    data, sr = sf.read(path)
    return data, sr


def write_audio(path, data, sr):
    sf.write(file=path, data=data, samplerate=sr)


def normalize_audio(data):
    maximum = np.max(np.abs(data))

    normalized_signal = np.array([(data / maximum) * 32767], np.int16)[0] # TODO check if np.int is necessary

    return normalized_signal


if __name__ == '__main__':
    wav, sr = load_audio("/Users/amitroth/PycharmProjects/slm-benchnark/audio/background_noises/laugh.wav")
    print(wav)
    norm_wav = normalize_audio(wav)
    print(norm_wav)
    write_audio("/Users/amitroth/PycharmProjects/slm-benchnark/audio/background_noises/laugh_norm.wav", norm_wav, sr)