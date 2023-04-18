import torchaudio
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from configs import config
import soundfile as sf
import torchvision



df_test = pd.DataFrame([(path.stem,*path.stem.split("_"),path) for path in Path('full_test/').glob("*.ogg")],
                       columns = ['filename','name','id','path'])


class TestDataset(Dataset):
    def __init__(self, data, sr, transform=None):
        self.data = data
        self.audio_dir = data.path.values[0]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        signal, sr = torchaudio.load(self.audio_dir)
        signal = self._get_signal_chunk(signal.squeeze(), sr, item)
        signal = self._resample_if_necessary(signal, sr)
        # signal = self._mix_down_if_necessary(signal)
        mel = self._melspec(signal)
        mel = self._mono_to_color(mel)
        mel = self._normalize(mel)
        image = self._signal_to_image(mel)
        if self.transform != None:
            image = self.transform(image)
            return image

        return image

    def _get_signal_chunk(self, signal, sr, item):
        return signal[sr * (item * 5): sr * (5 * (item + 1))]

    def _resample_if_necessary(self, signal, sr):
        if sr != config.sr:
            resampler = torchaudio.transforms.Resample(sr, config.sr)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdims=True)
        return signal

    def _melspec(self, signal):
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.sr,
                                                               n_fft=config.n_fft,
                                                               hop_length=config.hop_length,
                                                               n_mels=config.n_mels,
                                                               win_length=config.win_length,
                                                               f_min=300
                                                               )
        mel = mel_spectrogram(signal)
        transform_to_db = torchaudio.transforms.AmplitudeToDB()
        mel = transform_to_db(mel)

        return mel

    def _mono_to_color(self, X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)

        _min, _max = X.min(), X.max()

        if (_max - _min) > eps:
            V = np.clip(X, _min, _max)
            V = 255 * (V - _min) / (_max - _min)
            V = V.type(torch.uint8)
        else:
            V = torch.zeros_like(X, dtype=torch.uint8)

        return V

    def _normalize(self, signal):
        signal = (signal.type(torch.float32) / 255.0).squeeze(dim=0)
        signal = torch.stack([signal, signal, signal])
        return signal

    def _signal_to_image(self, signal):
        trans = torchvision.transforms.ToPILImage()
        image = trans(signal)
        return image




dataset = TestDataset(df_test,21952)

if __name__ == '__main__':
    print(dataset[0])