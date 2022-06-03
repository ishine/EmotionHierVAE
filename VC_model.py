import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

import numpy as np

from module import *
from module.adaptor import VarianceAdaptor
from train_utils import AMC_loss


class VoiceConversionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        """ Configuration """
        # Structure
        d_cnt = config['Model']['Reconstruction']['d_contents']
        d_spk = config['Model']['Reconstruction']['d_speaker']

        # number of speaker
        n_spk = config['Dataset']['num_speaker']


        """ Architecture """ 
        # 1. ConvBlock Encoder
        self.encoder = Encoder(config)

        # 2. Style Embedding
        self.style_embedding = StyleNetwork(config)

        # 3. Attention Decoder
        self.decoder = Decoder(config)

        # 4. PostNet
        self.postnet = PostNet(config)


        """ Speaker loss"""
        # which is unsupervised loss for speaker embedding to express a speaker more general.
        self.SPK_loss = lambda spk_emb, spk_id: AMC_loss(spk_emb, spk_id, tot_class = n_spk)


    def forward(self, mel, spk_id, emo_id):
        list_VQEmb, list_SpkEmb, vq_loss, last_emb = self.encoder(mel)

        # output speaker embedding, and style embedding
        spk_emb, style_emb = self.style_embedding(list_SpkEmb, emo_id)
        spk_loss = self.SPK_loss(spk_emb, spk_id)

        recon_mel = self.decoder(list_VQEmb, last_emb, style_emb)
        post_mel = self.postnet(recon_mel)


        return recon_mel, post_mel, vq_loss, spk_loss, list_VQEmb, last_emb, style_emb






