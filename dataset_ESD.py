import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob

import sys
sys.path.append("..")
from prepare_utils import crop_data

import numpy as np
import re




class Dataset_ESD(Dataset):
    def __init__(self, mel_cropLength, dataset_mode='train', cropping=True, wav_return=False):
        self.wav_return = wav_return
        
        """ List of Dataset """
        txt_list = sorted(glob("../Dataset/**/*.txt"))
        txt_list = [pth for pth in txt_list if int(re.sub('^0.', "", pth.split("/")[-2])) > 10]

        _mel_list = sorted(glob(f"../ESD_preprocessed/mel/{dataset_mode}/**.npy"))
        _pitch_list = sorted(glob(f"../ESD_preprocessed/pitch/{dataset_mode}/**.npy"))
        _energy_list = sorted(glob(f"../ESD_preprocessed/energy/{dataset_mode}/**.npy"))

        remove_list = []

        self.bad_file_pth = "../ESD_preprocessed/bad_file_pitch.txt"
        with open(self.bad_file_pth, "r") as f:
            _list = f.readlines()
            for txt in _list:
                remove_list.append(txt.split("/")[-1].replace("\n", ""))


        self.mel_list, self.pitch_list, self.energy_list = [], [], []

        for pth1, pth2, pth3 in zip(_mel_list, _pitch_list, _energy_list):
            if not pth1.split("/")[-1] in remove_list:
                self.mel_list.append(pth1)
                self.pitch_list.append(pth2)
                self.energy_list.append(pth3)
        

        print(len(self.mel_list), len(remove_list))
        self.mode = dataset_mode
        self.crop_len = mel_cropLength

        
        self.mel_stats = np.load("../ESD_preprocessed/mel_stats.npy").astype(np.float64)
        self.f0_mean = np.load("../ESD_preprocessed/f0_mean.npy").astype(np.float64)
        self.f0_std = np.load("../ESD_preprocessed/f0_std.npy").astype(np.float64)
        self.e_mean = np.load("../ESD_preprocessed/e_mean.npy").astype(np.float64)
        self.e_std = np.load("../ESD_preprocessed/e_std.npy").astype(np.float64)

        self.cropping = False if dataset_mode == 'test' else cropping

    def __len__(self):
        return len(self.mel_list)

    def __getitem__(self, idx):
        """
        Total 15000 Train Set, 1000 Eval Set, and 1500 Test Set.
        In the train set (ESD), there are 1500 set of each of speakers, and 3000 set of each of emotion.

        #=== Contents
        - mel: mel-spectrogram, cropped by setting length
        - speaker_id
        - emotion_id: [neutral, Angry, Happy, Sad, Surprise] -> [0, 1, 2, 3, 4]
        """

        ### log-mel spectrogram
        _mel = np.load(self.mel_list[idx])
        _pitch = np.load(self.pitch_list[idx])
        _energy = np.load(self.energy_list[idx])

        if self.cropping:
            mel, pitch, energy = crop_data(_mel, _pitch, _energy, self.crop_len)
            mel = mel.T
        else:
            mel, pitch, energy = _mel, _pitch, _energy
            mel = mel.T

        mel = self.normalize(mel, self.mel_stats[0], self.mel_stats[1])
        pitch = self.feature_normalize(pitch, self.f0_mean, self.f0_std)
        energy = self.feature_normalize(energy, self.e_mean, self.e_std)

        ### Speaker ID
        spk_id = int(re.sub('^0.', "", self.mel_list[idx].split("/")[-1].split("_")[0])) - 11

        ### Emotion ID
        emo_id = self.mel_list[idx].split("/")[-1].split("_")[1].replace(".npy", "")
        emo_id = (int(emo_id) - 1) // 350
        
        if self.wav_return:    
            return {
                'mel': mel,
                'pitch': pitch,
                'energy': energy,
                'spk_id': spk_id,
                'emo_id': emo_id,
                'wav': self.mel_list[idx]
            }
        else:
            return {
                'mel': mel,
                'pitch': pitch,
                'energy': energy,
                'spk_id': spk_id,
                'emo_id': emo_id
            }

    def normalize(self, mel, mean, scale):
        return (mel - mean) / scale

    def feature_normalize(self, x, mean, scale):
        zero_idxs = np.where(x == 0.0)[0]
        x = (x - mean) / scale
        x[zero_idxs] = 0.0
        return x



def collate_fn(batch, only_mel=True):
    mel_list = torch.tensor([b['mel'] for b in batch]).float()
    f0_list = torch.tensor([b['pitch'] for b in batch]).float()
    e_list = torch.tensor([b['energy'] for b in batch]).float()
    spkID_list = torch.tensor([b['spk_id'] for b in batch]).long()
    emoID_list = torch.tensor([b['emo_id'] for b in batch]).long()

    if only_mel:
        return mel_list, spkID_list, emoID_list
    else:
        return mel_list, spkID_list, emoID_list, f0_list, e_list




if __name__ == "__main__":
    dataset = Dataset_ESD(128)
    loader = DataLoader(
        dataset, batch_size=64, 
        shuffle=False, collate_fn=collate_fn
    )

    for data in loader:
        print(data[3].shape, data[4].shape)
