import sys
sys.path.append("..")

import os
from prepare_utils import get_mel_from_audio
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(44)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np

import argparse
import time

from dataset_ESD import Dataset_ESD
from torch.utils.data import DataLoader

from module.vc_model import VoiceConversionModel

from utils import check_recon_mel, makedirs
import soundfile as sf
import yaml


""" Configuration """

def _argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--best_path', type=str,
        default="./assets/220320/1666957262.7049165"        
        # 1655554544.38614 (0.2d)       
        # 1655546299.8871748 (0.5d)
        # 1655560155.123934 (0.0d)
    )
    parser.add_argument(
        '--saved_model', type=str,
        default="checkpoint_100000.pth.tar"
    )
    
    args = parser.parse_args()
    return args

def get_wave_from_vocoder(mel, pth_saving, mode="recon", sr=22050):
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    # Load Model
    
    if sr==16000:
        vocoder_pth = "./MelGAN_16kHz/best_netG.pt"
        
        ckpt = torch.load(vocoder_pth, map_location=device)
        vocoder.mel2wav.load_state_dict(ckpt)

    wav = vocoder.inverse(mel.transpose(0, 1).unsqueeze(0))

    sf.write(pth_saving + f"/{mode}.wav", wav[0].to('cpu').detach().numpy(), sr)




class Synthesizer():
    def __init__(self, args):
        """ Model & Checkpoint """
        # Checkpoint
        ckpt_pth = os.path.join(args.best_path, args.saved_model)
        ckpt = torch.load(ckpt_pth)

        # Yaml Config
        config = yaml.load(open(os.path.join(args.best_path, "config.yaml")), Loader=yaml.FullLoader)
        self.config = config
        
        # Model
        self.model = VoiceConversionModel(config).to(device)
        self.model.eval()

        self.model.load_state_dict(ckpt['model'])

        # Test Dataset
        self.dataset = Dataset_ESD(None, 'test', False, True)

        # Statistics
        stats_file = "../VCTK_preprocessed/mel_stats.npy"
        self.mel_mean, self.mel_std = np.load(stats_file).astype(np.float64)


    def test_step(self, src_mel, tar_mel, src_SpkID=None, tar_SpkID=None, src_emoID=None, tar_emoID=None):
        with torch.no_grad():
            ### Get Representations
            src_spk_emb, src_VQlist = self.model.encoder(src_mel, src_SpkID)[0:2]
            tar_spk_emb, tar_VQlist = self.model.encoder(tar_mel, tar_SpkID)[0:2]

            ### Convert
            converse_mel = self.model.decoder(tar_spk_emb, tar_emoID, src_VQlist)
            converse_mel = self._mel_denormalize(converse_mel[0])
            converse_mel_npy = converse_mel.to('cpu').detach().numpy().T

            recon_mel = self.model.decoder(src_spk_emb, src_emoID, src_VQlist)
            recon_mel = self._mel_denormalize(recon_mel[0])
            recon_mel_npy = recon_mel.to('cpu').detach().numpy().T

            src_mel = self._mel_denormalize(src_mel[0])
            src_mel_npy = src_mel.to('cpu').detach().numpy().T

            tar_mel = self._mel_denormalize(tar_mel[0])
            tar_mel_npy = tar_mel.to('cpu').detach().numpy().T
            ### ==================================================================

        data_list = [converse_mel, recon_mel, src_mel, tar_mel]
        npy_list = [converse_mel_npy, recon_mel_npy, src_mel_npy, tar_mel_npy]

        print("Forward End!")

        return data_list, npy_list


    def Converse_unseen_to_unseen(self, src_ind, tar_ind):
        # Make directory to save
        self._makedir()

        """ Data Preparing """
        src_mel, src_SpkID, src_emoID, tar_mel, tar_SpkID, tar_emoID = self.prepare_from_ESD(src_ind, tar_ind)        



        """ Forward """
        torch_list, npy_list = self.test_step(src_mel, tar_mel, src_SpkID, tar_SpkID, src_emoID, tar_emoID)
        
        converse_mel, recon_mel, src_mel, tar_mel = torch_list
        converse_mel_npy, recon_mel_npy, src_mel_npy, tar_mel_npy = npy_list


        # Check Mel
        check_recon_mel(converse_mel_npy, self.test_path, 0, mode='converse')
        check_recon_mel(recon_mel_npy, self.test_path, 0, mode='recon')
        check_recon_mel(src_mel_npy, self.test_path, 0, mode='GT')
        check_recon_mel(tar_mel_npy, self.test_path, 0, mode='target')

        # Get wav
        get_wave_from_vocoder(converse_mel, self.test_path, mode='converse', sr=16000)
        get_wave_from_vocoder(recon_mel, self.test_path, mode='recon', sr=16000)
        get_wave_from_vocoder(src_mel, self.test_path, mode='GT', sr=16000)
        get_wave_from_vocoder(tar_mel, self.test_path, mode='target', sr=16000)



    def Converse_custom_to_custom(self, src_path, tar_path):
        ### read audio
        src_audio, src_fs = sf.read(src_path, samplerate=None)
        tar_audio, tar_fs = sf.read(tar_path, samplerate=None)

        if len(src_audio.shape) == 2:
            src_audio = (src_audio[:, 0] + src_audio[:, 1]) / 2

        if len(tar_audio.shape) == 2:
            tar_audio = (tar_audio[:, 0] + tar_audio[:, 1]) / 2

        assert src_fs == tar_fs == 44100, "[Synthesizer.py] You need to prepare an audio of rate 44.1kHz"

        # Downsample
        src_audio, tar_audio = src_audio[::2], tar_audio[::2]

        ### convert to mel
        src_mel = get_mel_from_audio(self.config, src_audio)
        tar_mel = get_mel_from_audio(self.config, tar_audio)

        src_mel = torch.tensor(src_mel.T).float().to(device)
        tar_mel = torch.tensor(tar_mel.T).float().to(device)

        src_len = (src_mel.shape[0] // 16) * 16
        tar_len = (tar_mel.shape[0] // 16) * 16

        src_mel = self._mel_normalize(src_mel)[:src_len, :].unsqueeze(0)
        tar_mel = self._mel_normalize(tar_mel)[:tar_len, :].unsqueeze(0)

        #print(src_len, tar_len)


        """ Step """

        # Build directory to save
        self._makedir()

        torch_list, npy_list = self.test_step(src_mel, tar_mel)
        
        converse_mel, recon_mel, src_mel, tar_mel = torch_list
        converse_mel_npy, recon_mel_npy, src_mel_npy, tar_mel_npy = npy_list

        # Check Mel
        check_recon_mel(converse_mel_npy, self.test_path, 0, mode='converse')
        check_recon_mel(recon_mel_npy, self.test_path, 0, mode='recon')
        check_recon_mel(src_mel_npy, self.test_path, 0, mode='GT')
        check_recon_mel(tar_mel_npy, self.test_path, 0, mode='target')

        # Get wav
        get_wave_from_vocoder(converse_mel, self.test_path, mode='converse')
        get_wave_from_vocoder(recon_mel, self.test_path, mode='recon')
        get_wave_from_vocoder(src_mel, self.test_path, mode='GT')
        get_wave_from_vocoder(tar_mel, self.test_path, mode='target')




    def prepare_from_ESD(self, src_ind, tar_ind):
        src_mel = torch.tensor(self.dataset[src_ind]['mel']).float().to(device)
        tar_mel = torch.tensor(self.dataset[tar_ind]['mel']).float().to(device)

        src_len = (src_mel.shape[0] // 16) * 16
        src_len = 96
        tar_len = (tar_mel.shape[0] // 16) * 16
        
        src_mel = src_mel[:src_len, :].unsqueeze(0)
        tar_mel = tar_mel[:tar_len, :].unsqueeze(0)
        
        src_SpkID = torch.tensor(self.dataset[src_ind]['spk_id']).long().to(device).unsqueeze(0)
        tar_SpkID = torch.tensor(self.dataset[tar_ind]['spk_id']).long().to(device).unsqueeze(0)
        
        src_emoID = torch.tensor(self.dataset[src_ind]['emo_id']).long().to(device).unsqueeze(0)
        tar_emoID = torch.tensor(self.dataset[tar_ind]['emo_id']).long().to(device).unsqueeze(0)

        print(src_len, tar_len)
        print(src_SpkID.item(), tar_SpkID.item())
        print(src_emoID.item(), tar_emoID.item())
        print(self.dataset[src_ind]['wav'])

        return src_mel, src_SpkID, src_emoID, tar_mel, tar_SpkID, tar_emoID


    def _makedir(self):
        ts = time.time()
        self.test_path = os.path.join("./assets_test", str(ts))
        makedirs(self.test_path)

    def _mel_denormalize(self, mel):
        if isinstance(mel, torch.Tensor):
            _mean, _std = torch.tensor(self.mel_mean).float().to(device), torch.tensor(self.mel_std).float().to(device)
            return mel * _std + _mean
        else:
            return mel * self.mel_std + self.mel_mean

    def _mel_normalize(self, mel):
        if isinstance(mel, torch.Tensor):
            _mean, _std = torch.tensor(self.mel_mean).float().to(device), torch.tensor(self.mel_std).float().to(device)
            return (mel - _mean) / _std
        else:
            return (mel - self.mel_mean) / self.mel_std

   # def _M4a2Wav(self, m4a_path):



    
args = _argparse()
synthesizer = Synthesizer(args)

dir_custom = "../Dataset_Custom/Custom"

synthesizer.Converse_unseen_to_unseen(555, 1000)
# synthesizer.Converse_custom_to_custom(
#     os.path.join(dir_custom, "hamzi_002_eng.wav"),
#     os.path.join(dir_custom, "male2_neutral_5b_2.wav"),
# )


