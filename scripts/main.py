
import numpy as np
import librosa





def merge(path_1, path_2, output_path):
    wav_1, fs_1 = sf.read(path_1)
    wav_2, fs_2 = sf.read(path_2)

    wav_2 = wav_2.sum(axis=1) / 2
    # wav_1 = wav_1.sum(axis=1) / 2

    wav_2 = librosa.resample(y=wav_2, orig_sr=fs_2, target_sr=fs_1)

    min_length = min(len(wav_1), len(wav_2))

    merged_data = np.vstack((wav_1[:min_length], wav_2[:min_length])).T
    merged_data = merged_data.sum(axis=1) / 2
    print(merged_data)

    sf.write(output_path, merged_data, fs_1)



if __name__ == '__main__':
    # tts("the constructions is so loud", "construction.wav")
    merge("construction.wav", "background_construction.wav", "merged.wav")