import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from module.vc_utils import *




""" Voice Conversion Blocks """

class EncVCBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Parameter """
        n_codebook = config['Model']['n_codebook']
        
        d_hid = config['Model']['d_encoder_hidden']
        d_quant = config['Model']['d_quantizer_hidden']
        d_vae = config['Model']['d_pitch_hidden']
        
        kernel_size = config['Model']['kernel_size_encoder']
        dropout = config['Model']['dropout_encoder']
    
        """ Architecture """
        # Convolution Module
        self.conv_module = EncConvModule(d_hid, d_vae, kernel_size, dropout, 2)
        
        # Phoneme Quantizer (Pitch & timbre invarient feature)
        self.quant_conv = nn.Linear(d_hid, d_quant, bias=False)
        self.quantize_norm = nn.InstanceNorm1d(d_quant, affine=False)
        self.quantizer = Quantizer(n_codebook, d_quant)
        
    def forward(self, x):
        """ (B, T, C) -> (B, T, C) """
        
        # 1) Conv Module
        out, qz_mean, qz_std = self.conv_module(x)
        
        # 2) Phoneme Quantizer (with instance normalization)
        quant_emb = self.quantize_norm(self.quant_conv(out))
        quant_emb, _loss_quant, _ = self.quantizer(quant_emb)
        
        return out, quant_emb, _loss_quant, [qz_mean, qz_std]
        
        
        
class DecVCBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Model Parameter """
        n_emo = config['Dataset']['num_emotions']
        
        d_quant = config['Model']['d_quantizer_hidden']
        d_spk = config['Model']['d_speaker_hidden']
        d_emo = config['Model']['d_emotion_hidden']
        d_vae = config['Model']['d_pitch_hidden']
        d_hid = config['Model']['d_decoder_hidden']
        
        kernel_size = config['Model']['kernel_size_decoder']
        dropout = config['Model']['dropout_decoder']
        
        """ Architecture """
        # 1x1 conv / quantized embedding
        self.quant_conv = nn.Linear(d_quant, d_hid, bias=False)
        
        # Convolution Module
        self.conv_module = DecConvModule(d_hid, d_vae, d_spk, d_emo, kernel_size, dropout, 2)
        
        # 1x1 conv / statistical latent vector (pitch)
        self.pitch_conv = nn.Linear(d_vae, d_hid, bias=False)
        
        # convolution residual net
        self.conv1 = Conv(d_hid, d_hid, kernel_size, dropout)
        self.conv2 = Conv(d_hid, d_hid, kernel_size, dropout)
        
    def forward(self, x, quant_emb, spk_emb, emo_emb, qz_stats=None):
        """
        ? INPUT
        :input: tensor, (B, T, C)
        :quant_emb: tensor, (B, T, C_quant)
            This feature is invariant for timbre, and pitch, which is quantized from dictionarys.
        :spk_emb: tensor, (B, C_spk)
        :emo_id: int tensor, (B, C_emb)
        :qz_stats: list, [qz_mean, qz_std]
        """
        if qz_stats is not None:
            qz_mean, qz_std = qz_stats
        
        #== add quantized embedding
        hid = self.quant_conv(quant_emb) + x
        
        #== Convolution Module
        hid, pz_mean, pz_std = self.conv_module(hid, spk_emb, emo_emb)
        
        #== add pitch embedding
        if qz_stats is not None:
            dist_posterior = D.Normal(qz_mean + pz_mean, qz_std * pz_std)
        dist_prior = D.Normal(pz_mean, pz_std)
        
        if qz_stats is not None:
            z = dist_posterior.rsample()              # (B, T, C_vae)
            kl = D.kl.kl_divergence(dist_posterior, dist_prior).mean()
        else:
            z = dist_prior.rsample()
            kl = 0.
        
        out = hid + self.pitch_conv(z)
        out = self.conv2(self.conv1(out)) + out
        
        return out, kl




""" Speaker Embedding Module """

class SpeakerModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Parameter """
        n_spk = config['Dataset']['num_speakers']
        
        d_hid = config['Model']['d_decoder_hidden']
        d_spk = config['Model']['d_speaker_hidden']
        
        dropout = config['Model']['dropout_encoder']
        
        """ Architecture"""
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_hid))
        
        self.spk_attn = MultiHeadAttention(d_hid, d_head=4, dropout=dropout)
        self.spk_cls = nn.Linear(d_hid, n_spk, bias=False)
        self.spk_last = nn.Linear(d_hid, 2 * d_spk, bias=False)
        
    def forward(self, x, spk_id):
        batch_size = x.shape[0]
        
        query = self.cls_token.repeat(batch_size, 1, 1)
        spk_emb = self.spk_attn(query=query, key=x).squeeze(1)
        
        spk_cls = self.spk_cls(spk_emb)
        s_mean, s_logvar = self.spk_last(spk_emb).chunk(2, dim=-1)
        
        s_std = s_logvar.exp().pow(0.5)
        s_dist = D.Normal(s_mean, s_std)
        
        z_s = s_dist.rsample()
        kl_loss = self.kl_loss(s_mean, s_logvar)
        cls_loss = self.classify_loss(spk_cls, spk_id)
        
        return z_s, kl_loss, cls_loss, s_mean
    
    def kl_loss(self, mean, log_var):
        KL = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return KL / mean.shape[0]
    
    def classify_loss(self, emb, spk_id):
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        return cross_entropy_loss(emb, spk_id)
           
        


""" Convolution Module """

class EncConvModule(nn.Module):
    def __init__(
        self,
        d_hid,
        d_vae,
        kernel_size,
        dropout,
        down_scale = 2
    ):
        super().__init__()
        
        """ Parameter """
        self.down_scale = down_scale
        
        """ Architecture """
        # Softplus
        self.softplus = SoftPlus()
        
        # convolution net
        if down_scale != 1:
            self.downsample = lambda x: self._downsample(x, down_scale)
            
        self.conv1 = Conv(d_hid, d_hid, kernel_size, dropout=dropout)
        self.conv2 = Conv(d_hid, d_hid, kernel_size, dropout=dropout)
        self.conv3 = Conv(d_hid, 2 * d_hid, kernel_size, dropout=dropout)
        self.conv4 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, stride=down_scale)
        
        # Posterior Net
        self.posterior_conv = nn.Linear(d_hid, 2 * d_vae, bias=False)
        
    def forward(self, x):
        #== Residual & 2 conv
        hid = self.conv2(self.conv1(x))
        hid = hid + x
        
        #== Residual + 2 conv
        out, res = self.conv3(hid).chunk(2, dim=-1)
        out = self.conv4(out)

        if self.down_scale != 1:
            out = out + self.downsample(hid)    # residual
        else:
            out = out + hid
        
        #== Posterior
        delta_mean, delta_std = self.posterior_conv(res).chunk(2, dim=-1)
        delta_std = self.softplus(delta_std)    # softplus (std, posterior)
        
        return out, delta_mean, delta_std
    
    def _downsample(self, x, down_scale):
        return F.avg_pool1d(x.contiguous().transpose(1, 2), kernel_size=down_scale).contiguous().transpose(1, 2)
        
        
            
class DecConvModule(nn.Module):
    def __init__(
        self,
        d_hid,
        d_vae,
        d_style,
        d_global,
        kernel_size,
        dropout,
        up_scale = 2,
    ):
        super().__init__()
        
        """ Parameter """
        self.up_scale = up_scale
        
        """ Architecture """
        self.softplus = SoftPlus()
                
        if up_scale != 1:
            self.upsample = lambda emb: self._upsample(emb, up_scale)

        self.conv1 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, d_cond=d_style)
        self.conv2 = Conv(d_hid, d_hid, kernel_size, dropout=dropout, d_cond=d_style)
        self.conv3 = UpConv(d_hid, d_hid, kernel_size, dropout=dropout, d_cond=d_style, up_scale=2)
        self.conv4 = Conv(d_hid, 2 * d_hid, kernel_size, dropout=dropout, d_cond=d_style)
        
        # prior net
        self.prior_conv = nn.Linear(d_hid, 2 * d_vae, bias=False)
        
        # global conditioning convolution
        self.global_conv = nn.Linear(d_global, d_hid, bias=False)
        
    def forward(self, x, spk_emb, emo_emb):
        """
        ? INPUT
        :input tensor: tensor, (B, T, C)
        :spk_emb: tensor, (B, C_style)
        :emo_emb: tensor, (B, C_emo)
            emo_emb is conditioning on the hidden vectors as making global style.
        """
        
        #== residual & 2 conv
        hid = self.conv2(self.conv1(x, spk_emb), spk_emb)
        hid = x + hid
        
        #== residual (upsample) & 2 conv (with PixelShuffle)
        res = hid + self.global_conv(emo_emb).unsqueeze(1)      # global conditioning
        out, res = self.conv4(self.conv3(res, spk_emb), spk_emb).chunk(2, dim=-1)
        
        if self.up_scale != 1:
            out = out + self.upsample(hid)
        else:
            out = out + hid
            
        #== Prior Net
        pz_mean, pz_std = self.prior_conv(res).chunk(2, dim=-1)
        pz_std = self.softplus(pz_std)
    
        return out, pz_mean, pz_std

    def _upsample(self, emb, up_scale):
        emb = emb.transpose(1, 2)
        return F.interpolate(emb, scale_factor=up_scale, mode='nearest').contiguous().transpose(1, 2)




""" Voice Conversion Layers """

class Conv(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        kernel_size,
        dropout=0.,
        stride=1,
        bias=True,
        d_cond=None,
    ):
        super().__init__()
        
        """ Parameter """
        padding = (kernel_size - 1) // 2
        
        """ Architecture """
        self.conv = nn.Conv1d(
            d_in, d_out, kernel_size, 
            padding=padding, padding_mode='replicate', stride=stride, bias=bias
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = AdaIN(d_out, d_cond) if d_cond is not None else nn.LayerNorm(d_out)
        
    def forward(self, x, cond=None):
        out = self.conv(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)
        
        if isinstance(self.norm, AdaIN):
            out = F.gelu(self.norm(out, cond))
        else:
            out = F.gelu(self.norm(out))

        return self.dropout(out)
    
    
    
class UpConv(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        kernel_size,
        dropout=0.,
        up_scale=2,
        d_cond=None,
    ):
        super().__init__()
        
        """ Parameter """
        padding = (kernel_size - 1) // 2
        
        """ Architecture """
        self.conv = nn.Conv1d(
            d_in, 2 * d_out, kernel_size, 
            padding=padding, padding_mode='replicate', stride=1
        )
        
        self.shuffle = PixelShuffle(up_scale)
        self.dropout = nn.Dropout(dropout)
        self.norm = AdaIN(d_out, d_cond) if d_cond is not None else nn.LayerNorm(d_out)
        
    def forward(self, x, cond=None):
        out = self.conv(x.contiguous().transpose(1, 2))
        out = self.shuffle(out).contiguous().transpose(1, 2)
        
        if isinstance(self.norm, AdaIN):
            out = F.gelu(self.norm(out, cond))
        else:
            out = F.gelu(self.norm(out))

        return self.dropout(out)
        
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_hid, d_head, dropout, return_attn=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_hid, d_head, dropout=dropout)
        self.return_weightAttn = return_attn

    def forward(self, query, key=None):
        """ (B, T, C) -> (B, T, C) """
        if key is None:
            key = query
        
        tot_timeStep = query.shape[1]       # (B, T, C)
        
        query = query.contiguous().transpose(0, 1)
        key = key.contiguous().transpose(0, 1)

        query, weight_Attn = self.attn(query, key, key)

        query = query.contiguous().transpose(0, 1)              # (B, T, C)

        if self.return_weightAttn:
            return query, weight_Attn
        else:
            return query