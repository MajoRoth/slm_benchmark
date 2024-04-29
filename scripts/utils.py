import soundfile as sf
import torch
from TTS.api import TTS
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface



class FastSpeech:

    def __init__(self):
        self.models, cfg, self.task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False}
        )
        TTSHubInterface.update_cfg_with_data_cfg(cfg, self.task.data_cfg)
        self.generator = self.task.build_generator(self.models, cfg)

    def tts(self, text):
        sample = TTSHubInterface.get_model_input(self.task, text)
        wav, rate = TTSHubInterface.get_prediction(self.task, self.models[0], self.generator, sample)
        return wav, rate


class Xtts:

    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def tts(self, text, audio_prompt_path, output_path):
        self.model.tts_to_file(text=text, speaker_wav=audio_prompt_path, language="en", file_path=output_path)

