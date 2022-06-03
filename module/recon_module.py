import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from .recon_layer import *
from .utils import PositionalEncoding


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        """ Parameter """
        n_mel = config['Preprocess']['n_mel']

        d_spk = config['Model']['Reconstruction']['d_speaker']
        d_cnt = config['Model']['Reconstruction']['d_contents']

        dropout = config['Model']['Reconstruction']['dropout']

        n_Block = config['Model']['Reconstruction']['n_EncVCBlock']

        """ Architecture """
        # 1) Pre-Linear Layer
        self.prenet = nn.Sequential(
            nn.Linear(n_mel, d_cnt, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_cnt, d_cnt, bias=False),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2) Encoder-VQ Blocks
        self.Blocks = nn.ModuleList([
            EncVCBlock(config) for _ in range(n_Block)
        ])

        # 3) From AgainVC
        self.last_norm = nn.InstanceNorm1d(d_cnt, affine=False)
        self.varActLayer = VariantActivationLayer(config)

        

    def forward(self, mel):
        # Pre Linear Layer
        hid = self.prenet(mel)

        # Encoder VQ Blocks
        list_SpkEmb = []
        list_VQEmb = []
        VQ_loss = 0

        for block in self.Blocks:
            vq_emb, spk_emb, hid, _loss = block(hid)

            list_SpkEmb.append(spk_emb)
            list_VQEmb.append(vq_emb)
            VQ_loss += _loss

        VQ_loss /= len(self.Blocks)
        return list_VQEmb, list_SpkEmb, VQ_loss, self.varActLayer(self.last_norm(hid))



class VariantActivationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        """ Parameter """
        d_cnt = config['Model']['Reconstruction']['d_contents']

        self.conv1 = nn.Conv1d(d_cnt, 4, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv1d(4, d_cnt, kernel_size=3, padding=1, padding_mode='replicate')
    
    def forward(self, x, alpha=0.1):
        hid = self.conv1(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)
        hid = 1 / (1 + torch.exp(-alpha * hid))     # sigmoid layer
        hid = self.conv2(hid.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)

        return hid



        
class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        """ Parameter """
        n_mel = config['Preprocess']['n_mel']
        n_encBlock = config["Model"]['Reconstruction']['n_EncVCBlock']

        d_spk = config['Model']['Reconstruction']['d_speaker']
        d_hid = config['Model']['Reconstruction']['d_contents'] # * n_encBlock

        dropout = config['Model']['Reconstruction']['dropout']
        self.down_rate = config['Model']['Reconstruction']['downSampling_rate']

        n_Block = config["Model"]['Reconstruction']['n_DecConvBlock']
        n_Attn = config["Model"]['Reconstruction']['n_DecAttnBlock']
        

        """ Architecture """

        self.style_linear = nn.Linear(d_hid * 2, d_hid)

        self.dec_conv = nn.Sequential(*[
            DecConvBlock(config, d_hid) for _ in range(n_Block)
        ])

        self.pos_dec = PositionalEncoding(d_hid)
        self.dec_attn = nn.Sequential(*[
            DecAttnBlock(config, d_hid) for _ in range(n_Attn)
        ])

        self.post_lstm = nn.LSTM(d_hid, d_hid * 2, 3, batch_first=True)
        self.post_linear = nn.Linear(d_hid, n_mel)


        #self.upsampling = nn.ConvTranspose1d(d_spk + d_hid, d_hid, kernel_size=down_rate, stride=down_rate)



    def forward(self, list_VQEmb, cnt_emb, style_emb):
        """
        ? INPUT
        - list_VQEmb: list of (B, T/2, C), (B, T/4, C), ...
        - cnt_emb: (B, T_down, C)
        - style_emb: (B, d_spk + d_emo)

        ? OUTPUT
        - output: (B, T, C)
        """

        # Concatenate contents embedding
        
        # hid = torch.cat(vq_emb, dim=-1)
        # hid = torch.stack(vq_emb).sum(0)


        ### Feed Attention Blocks
        # Positional Encoding
        # hid = self.pos_dec(hid)

        # Decoder
        for i, layer in enumerate(self.dec_conv):
            ind = len(self.dec_conv) - 1 - i

            timesteps = list_VQEmb[ind].shape[1]

            if ind == len(self.dec_conv) - 1:
                hid = torch.cat([list_VQEmb[ind], cnt_emb], dim=-1)
            else:
                hid = torch.cat([list_VQEmb[ind], hid], dim=-1)

            hid = layer(hid, style_emb)  # (B, T_up, C)

        hid = self.pos_dec(hid)
        for layer in self.dec_attn:
            hid = layer(hid, style_emb)

        return self.post_linear(hid)


    def _upsample(self, emb, scale_factor):
        emb = emb.transpose(1, 2)
        return F.interpolate(emb, scale_factor=scale_factor, mode='nearest').transpose(1, 2)




class StyleNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()

        n_emo = config['Dataset']['num_emotion']

        d_spk = config["Model"]["Reconstruction"]["d_speaker"]

        n_vecs = config["Model"]["Reconstruction"]["n_EncVCBlock"]
        n_layer = config["Model"]["Reconstruction"]["n_StyleLayer"]

        # self.spk_net = nn.Sequential(
        #     *[nn.Sequential(nn.Linear(d_spk if i != 0 else d_spk * n_vecs, d_spk), nn.ReLU()) for i in range(4)]
        # )

        self.spk_net = nn.Linear(d_spk * n_vecs, d_spk)

        self.emo_embed = nn.Embedding(n_emo, d_spk)     # where we set d_emo == d_spk

        self.style_net = nn.Sequential(
            *[nn.Sequential(nn.Linear(2 * d_spk, 2 * d_spk), nn.ReLU()) for i in range(n_layer)]
        )

        
    def forward(self, list_SpkEmb, emo_id):
        """
            - list_SpkEmb: list, list of (Batch_size, n_spk)
        """
        spk_emb = torch.cat(list_SpkEmb, dim=-1)
        spk_emb = self.spk_net(spk_emb)                     # (B, d_spk)

        emo_emb = self.emo_embed(emo_id)                    # (B, d_emo)

        emb = torch.cat([spk_emb, emo_emb], dim=-1)         # (B, d_spk + d_emo)
        style_emb = self.style_net(emb)                     # (B, d_spk + d_emo)

        return spk_emb, style_emb



    




class PostNet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        n_mel = config["Preprocess"]["n_mel"]

        self.conv = nn.ModuleList()

        self.conv.append(
            nn.Sequential(
                nn.Conv1d(n_mel, 512, kernel_size=5, stride=1, padding=2, dilation=1),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
                    nn.BatchNorm1d(512))
            )

        self.conv.append(
            nn.Sequential(
                nn.Conv1d(512, n_mel, kernel_size=5, stride=1, padding=2, dilation=1),
                nn.BatchNorm1d(n_mel))
        )

    def forward(self, x):
        out = x.contiguous().transpose(1, 2)
        for i in range(len(self.conv) - 1):
            out = torch.tanh(self.conv[i](out))

        out = self.conv[-1](out).contiguous().transpose(1, 2)

        # Residual Connection
        # ! Comment: 의외라 깜짝 놀랐는데 이 한 줄이 loss 입장에서 상당히 중요함.
        out = out + x

        return out


if __name__ == "__main__":
    import yaml
    config = yaml.load(
        open("./config.yaml", "r"), Loader=yaml.FullLoader
    )

    input = torch.zeros(8, 128, 80)

    model = DisentangleReconstruction(config)

    from torchsummaryX import summary
    summary(model, input)


