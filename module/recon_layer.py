import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from .utils import *


class MultiHeadAttention(nn.Module):
    def __init__(self, d_hid, d_head, decoder):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_hid, d_head)
        self.decoder = decoder


    def forward(self, query, key=None):
        if key is None:
            key = query
        
        tot_timeStep = query.shape[1]       # (B, T, C)
        
        query = query.contiguous().transpose(0, 1)
        key = key.contiguous().transpose(0, 1)

        if self.decoder:
            query = self.attn(query, key, key, attn_mask = src_mask(tot_timeStep).to(query.device))[0]
        else:
            query = self.attn(query, key, key)[0]

        query = query.contiguous().transpose(0, 1)
        return query


class ConvFeedForward(nn.Module):
    def __init__(self, config, d_hid, kernel_size=None, conditional=False):
        super().__init__()

        """ Parameter """
        d_spk = config["Model"]["Reconstruction"]["d_speaker"]
        dropout = config["Model"]["Reconstruction"]["dropout"]

        if kernel_size is None:
            kernel_size = config["Model"]["Reconstruction"]["kernel_size"]
        padding = (kernel_size - 1) // 2

        """ Layer """
        self.conv1 = nn.Conv1d(d_hid, 2 * d_hid, kernel_size = kernel_size, padding = padding)
        self.conv2 = nn.Conv1d(2 * d_hid, d_hid, kernel_size = kernel_size, padding = padding)

        self.dropout = nn.Dropout(dropout)
        self.norm = AdaIN(d_hid, d_spk) if conditional else nn.LayerNorm(d_hid)
        

    def forward(self, x, cond=None):
        """ (B, T, C) -> (B, T, C) """
        out = x.contiguous().transpose(1, 2)

        out = self.conv2(F.gelu(self.conv1(out)))
        out = out.contiguous().transpose(1, 2)

        if cond is not None:
            out = self.norm(self.dropout(out) + x, cond)
        else:
            out = self.norm(self.dropout(out) + x)
        return out


class LinearFeedForward(nn.Module):
    def __init__(self, config, d_hid, spec_norm=True):
        super().__init__()

        """ Parameter """
        dropout = config["Model"]["Reconstruction"]["dropout"]

        """ Layer """
        f = spectral_norm if spec_norm else lambda x: x
        self.linear1 = f(nn.Linear(d_hid, 2 * d_hid))
        self.linear2 = f(nn.Linear(2 * d_hid, d_hid))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_hid)
        

    def forward(self, x, cond=None):
        """ (B, T, C) -> (B, T, C) """

        out = self.linear2(F.gelu(self.linear1(x)))
        out = self.norm(self.dropout(out) + x)
        return out



""" Attnetion Blocks """

class EncConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        """ Parameter """
        d_hid = config["Model"]["Reconstruction"]["d_contents"]
        scale_factor = config["Model"]["Reconstruction"]["scale_factor"]
        
        """ Architecture """
        self.downsample = nn.AvgPool1d(kernel_size=scale_factor, stride=scale_factor)

        self.conv1 = Conv(config, d_hid, d_hid)
        self.conv2 = Conv(config, d_hid, d_hid)
        self.conv3 = Conv(config, d_hid, d_hid)
        self.conv4 = Conv(config, d_hid, d_hid, stride=2)


    def forward(self, x):
        """ (B, T, C) -> (B, T, C) """

        hid = x + self.conv2(self.conv1(x))

        res = self.downsample(hid.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)

        return res + self.conv4(self.conv3(hid))


class EncVCBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        """ Parameter """
        d_head = config["Model"]["Reconstruction"]["d_head"]
        n_codebook = config["Model"]["Reconstruction"]["n_codebook"]
        n_tokens = config["Model"]["Reconstruction"]["n_SpkTokens"]

        d_hid = config["Model"]["Reconstruction"]["d_contents"]
        d_spk = config["Model"]["Reconstruction"]["d_speaker"]
        
        """ Architecture """
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_hid))

        # 1) Convolution Block
        self.conv_block = EncConvBlock(config)

        # 2) Speaker Attention Extractor
        self.spk_norm = nn.LayerNorm(d_hid)
        self.spk_attn = MultiHeadAttention(d_hid, d_head, decoder=False)
        self.spk_last = nn.Linear(d_hid, d_spk)

        # 3) Vector Quantizer
        self.quan_linear = nn.Linear(d_hid, d_hid)
        self.quantizer = Quantizer(n_codebook, d_hid)


    def forward(self, x):
        batch_size = x.shape[0]

        # 1) Conv Block
        connection = self.conv_block(x)                             # (B, T_down, d_hid)

        # 2) Attn
        query = self.cls_token.repeat(batch_size, 1, 1)
        key = self.spk_norm(connection)

        spk_emb = self.spk_attn(query=query, key=key).squeeze(1)    # (B, d_hid)
        spk_emb = self.spk_last(spk_emb)

        # 3) Vector Quantizer
        vq_emb, _loss, _ = self.quantizer(self.quan_linear(connection))

        return vq_emb, spk_emb, connection, _loss



class DecConvBlock(nn.Module):
    def __init__(self, config, d_hid):
        super().__init__()

        """ Parameter """
        d_spk = config["Model"]["Reconstruction"]["d_speaker"]

        scale_factor = config["Model"]["Reconstruction"]["scale_factor"]
        self.scale_factor = scale_factor
        
        """ Architecture """
        # self.upsample = nn.ConvTranspose1d(d_hid, d_hid, kernel_size=2, stride=scale_factor)
        self.upsample = lambda emb: self._upsample(emb, scale_factor)

        self.conv1 = Conv(config, 2 * d_hid, d_hid, d_cond=2 * d_spk)
        self.conv2 = Conv(config, d_hid, d_hid, d_cond=2 * d_spk)
        self.conv3 = Conv(config, d_hid, d_hid, d_cond=2 * d_spk)
        self.conv4 = Conv(config, d_hid, d_hid, d_cond=2 * d_spk, upsample=True)

    def forward(self, x, style_emb):
        # hid = self.shuffle(self.conv1(x, cond))                           # (B, 2*T, C//2)
        # hid = self.conv2(hid, cond)   # (B, 2*T, C)

        # return res + hid
        hid = self.conv2(self.conv1(x, style_emb), style_emb)
        res = self.upsample(hid)

        # res = hid

        return res + self.conv4(self.conv3(hid, style_emb), style_emb)

    def _upsample(self, emb, scale_factor):
        emb = emb.transpose(1, 2)
        return F.interpolate(emb, scale_factor=scale_factor, mode='nearest').transpose(1, 2)



class DecAttnBlock(nn.Module):
    # ! Reference
    # The Nuts and Bolts of Adopting Transformer in GANs, arxiv:2110.13107
    def __init__(self, config, d_hid):
        super().__init__()

        """ Parameters """
        d_head = config["Model"]["Reconstruction"]["d_head"]
        d_spk = config["Model"]["Reconstruction"]["d_speaker"]

        dropout = config["Model"]["Reconstruction"]["dropout"]

        kernel_size = config["Model"]["Reconstruction"]["kernel_size"]
        padding = (kernel_size - 1) // 2


        """ Architecture """

        self.norm = nn.LayerNorm(d_hid)
        self.attn = MultiHeadAttention(d_hid, d_head, decoder=True)
        self.adanorm1 = AdaIN(d_hid, 2 * d_spk)

        self.adanorm2 = AdaIN(d_hid, 2 * d_spk)
        self.conv_feed = nn.Sequential(
            nn.Conv1d(d_hid, 2*d_hid, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.Conv1d(2*d_hid, d_hid, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond):
        """
        - x: (B, T, C)
        - cond: (B, d_spk + d_emo)
        """

        # Feed
        # hid = self.norm(x)              # LayerNorm
        res = x
        hid = self.adanorm1(x, cond)  # AdaNorm
        hid = self.attn(hid)            # Attn
        hid = hid + res                   # Residual

        res = hid
        hid = self.adanorm2(hid, cond)  # AdaNorm
        hid = self.conv_feed(hid.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)
        out = hid + res

        return out






class Conv(nn.Module):
    def __init__(self, config, d_in, d_out, stride=1, d_cond=None, upsample=False):
        super().__init__()

        self.upsample = upsample

        if upsample:
            d_out *= 2

        dropout = config['Model']['Reconstruction']['dropout']
        kernel_size = config["Model"]['Reconstruction']['kernel_size']
        scale_factor = config['Model']['Reconstruction']['scale_factor']
        padding = (kernel_size - 1) // 2

        """ Architecture """
        self.conv = nn.Conv1d(d_in, d_out, kernel_size, padding=padding, padding_mode='replicate', stride=stride)

        if upsample:
            self.shuffle = PixelShuffle(scale_factor)
            d_out = d_out // 2

        self.dropout = nn.Dropout(dropout)
        self.norm = AdaIN(d_out, d_cond) if d_cond is not None else nn.LayerNorm(d_out)

    def forward(self, x, cond=None):
        out = self.conv(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)

        if self.upsample:
            out = self.shuffle(out) 

        if isinstance(self.norm, AdaIN):
            out = F.gelu(self.norm(out, cond))
        else:
            out = F.gelu(self.norm(out))

        return self.dropout(out)


class BatchNorm1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, hidden):
        out = self.norm(hidden.contiguous().transpose(1, 2))
        return out.contiguous().transpose(1, 2)


