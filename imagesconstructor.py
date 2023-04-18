import os
import glob
from tqdm import tqdm
import torch,torchaudio,torchvision
from configs import config
import numpy as np
from pathlib import Path
import librosa




def image_preprocess(train=True):
    if train:
        os.makedirs('train_mel/', exist_ok=True)
        dataset = glob.glob(os.path.join("train_audio/", '**/*.ogg'), recursive=True)
        for train_path in tqdm(dataset):

            train_path = Path(train_path)
            ogg_name = os.path.basename(train_path)
            dirname = train_path.parent.stem
            os.makedirs(f'train_mel/{dirname}', exist_ok=True)

            mel_image = get_image(train_path)
            ogg_path = Path('train_mel/', dirname, ogg_name)
            save_path = ogg_path.with_suffix('.png')

            mel_image.save(save_path)

    else:
        os.makedirs('valid_mel/',exist_ok=True)
        dataset = glob.glob(os.path.join('valid_audio', '**/*.mp3'), recursive=True)
        for train_path in tqdm(dataset):

            train_path = Path(train_path)
            ogg_name = os.path.basename(train_path)
            dirname = train_path.parent.stem
            os.makedirs(f'valid_mel/{dirname}', exist_ok=True)

            mel_image = get_image(train_path,train=False)
            ogg_path = Path('valid_mel/', dirname, ogg_name)
            save_path = ogg_path.with_suffix('.png')

            mel_image.save(save_path)

def test_mels():
    os.makedirs('test_mel/', exist_ok=True)
    dataset = glob.glob(os.path.join("test_audio/", '*.ogg'), recursive=True)
    for train_path in tqdm(dataset):
        train_path = Path(train_path)
        ogg_name = os.path.basename(train_path)
        os.makedirs(f'test_mel/', exist_ok=True)

        mel_image = get_image(train_path)
        ogg_path = Path('test_mel/', ogg_name)
        save_path = ogg_path.with_suffix('.png')

        mel_image.save(save_path)




def get_image(image_path,train = True):
    if train:
        signal,sr = torchaudio.load(image_path)

        signal = _resample_if_necessary(signal,sr)
        signal = _mix_down_if_necessary(signal)
        signal = _right_pad_if_necessary(signal)
        signal = _clip_signal(signal,sr)
        mel = _melspec(signal)
        mel = _mono_to_color(mel)
        mel = _normalize(mel)
        image = _signal_to_image(mel)
        return image
    else:
        signal,sr = librosa.load(image_path)
        signal,sr = torch.from_numpy(signal).unsqueeze(dim=0),torch.tensor(sr)
        signal = _resample_if_necessary(signal,sr)
        signal = _mix_down_if_necessary(signal)
        signal = _right_pad_if_necessary(signal)
        signal = _clip_signal(signal,sr)
        mel = _melspec(signal)
        mel = _mono_to_color(mel)
        mel = _normalize(mel)
        image = _signal_to_image(mel)
        return image



def _clip_signal(signal,sr):
    signal = signal[:, 0 * sr: 5 * sr]
    return signal





def _resample_if_necessary(signal, sr):
    if sr != config.sr:
        resampler = torchaudio.transforms.Resample(sr, config.sr)
        signal = resampler(signal)
    return signal

def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdims=True)
    return signal



def _right_pad_if_necessary(signal):
    length_signal = signal.shape[1]
    if length_signal < config.num_samples:
        num_missing_samples = config.num_samples - length_signal  # define how much we want right pad
        last_dim_padding = (0, num_missing_samples)  # (0,2)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def _melspec(signal):
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


def _mono_to_color(X, eps=1e-6, mean=None, std=None):
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


def _normalize(signal):
    signal = (signal.type(torch.float32) / 255.0).squeeze(dim=0)
    signal = torch.stack([signal, signal, signal])
    return signal

def _signal_to_image(signal):
    trans = torchvision.transforms.ToPILImage()
    image = trans(signal)
    return image


if __name__ == '__main__':
    # image_preprocess(train=False)
    # test_mels()
    # dataset = glob.glob(os.path.join("test_audio/", '*.ogg'), recursive=True)
    # for train_path in dataset[:2]:
    #     print(train_path)
    img = get_image('train_auido_cut/bawhor2/XC113261-Part 1.ogg')
    img.show()

  #   os.makedirs('valid_mel/', exist_ok=True)
  #   dataset = glob.glob(os.path.join('valid_audio', '**/*.mp3'), recursive=True)
  #   for train_path in tqdm(dataset[:2]):
  #       train_path = Path(train_path)
  #       print(train_path)
  #       mel_image = get_image(train_path)
  #       mel_image.show()
  #