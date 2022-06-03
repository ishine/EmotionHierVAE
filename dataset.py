import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob

import sys
sys.path.append("..")
from prepare_utils import crop_mel

import numpy as np
import re

class Dataset_ESD(Dataset):
    def __init__(self, mel_cropLength, dataset_mode='train', cropping=True):
        """ List of Dataset """
        txt_list = sorted(glob("../Dataset/**/*.txt"))
        txt_list = [pth for pth in txt_list if int(re.sub('^0.', "", pth.split("/")[-2])) > 10]

        self.mel_list = sorted(glob(f"../ESD_preprocessed/mel/{dataset_mode}/**.npy"))
        self.pitch_list = sorted(glob(f"../ESD_preprocessed/mel/{dataset_mode}/**.npy"))
        self.energy_list = sorted(glob(f"../ESD_preprocessed/mel/{dataset_mode}/**.npy"))

        self.mode = dataset_mode
        self.crop_len = mel_cropLength

        stats_file = "../ESD_preprocessed/mel_stats.npy"
        self.mel_stats = np.load(stats_file).astype(np.float64)

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
        if self.cropping:
            mel = crop_mel(np.load(self.mel_list[idx]), self.crop_len).T # (n_mel, cropped_len)
        else:
            mel = np.load(self.mel_list[idx]).T
        mel = self.normalize(mel, self.mel_stats[0], self.mel_stats[1])

        ### Speaker ID
        spk_id = int(re.sub('^0.', "", self.mel_list[idx].split("/")[-1].split("_")[0])) - 11

        ### Emotion ID
        emo_id = (idx % 1500) // 300
        
        return {
            'mel': mel,
            'spk_id': spk_id,
            'emo_id': emo_id
        }

    def normalize(self, mel, mean, scale):
        return (mel - mean) / scale



def collate_fn(batch):
    batch_size = len(batch)

    mel_list = torch.tensor([b['mel'] for b in batch]).float()
    spkID_list = torch.tensor([b['spk_id'] for b in batch]).long()
    emoID_list = torch.tensor([b['emo_id'] for b in batch]).long()

    return mel_list, spkID_list, emoID_list



if __name__ == "__main__":
    dataset = Dataset_ESD(128)
    loader = DataLoader(
        dataset, batch_size=16, 
        shuffle=False, collate_fn=collate_fn
    )

    for data in loader:
        print(data[0].shape, data[1].shape, data[2].shape)
        break
