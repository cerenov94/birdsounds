import torch
import pandas as pd




class Starter_configuration():
    def __init__(self):
        self.n_mels = 224
        self.n_fft = 892
        self.win_length = self.n_fft
        self.batch_size = 10
        self.sr = 32000
        self.hop_length = 245
        self.len_chak = 448
        self.num_samples = self.sr*5
        self.lr = 0.001


config = Starter_configuration()