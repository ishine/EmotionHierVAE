import torch
import torch.nn as nn
import torch.nn.functional as F

from module.vc_module import *




""" Encoder """

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Parameter """
        n_mel = config['Preprocess']['n_mel']

        d_hid = config['Model']['d_encoder_hidden']

        n_Block = config['Model']['n_EncVCBlock']
        
        dropout = config['Model']['dropout_encoder']
        
        """ Architecture """
        # 1) Pre-Linear Layer
        self.prenet = nn.Sequential(
            nn.Linear(n_mel, d_hid, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid, bias=False),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2) Encoder-VQ Blocks
        self.Blocks = nn.ModuleList([
            EncVCBlock(config) for _ in range(n_Block)
        ])
        
        # 3) Spk Module
        self.speaker_module = SpeakerModule(config)
        
    def forward(self, mel, spk_id):
        #== Pre-Net
        hid = self.prenet(mel)
        
        #== Encoder Blocks
        list_quantEmb = []
        list_postStats = []
        quant_loss = 0
        
        for block in self.Blocks:
            hid, quant_emb, _loss_quant, stats = block(hid)
            
            list_quantEmb.append(quant_emb)
            list_postStats.append(stats)
            quant_loss += _loss_quant
            
        #== Speaker Module
        if self.training:
            spk_emb, kl_spk_loss, classify_loss, _ = self.speaker_module(hid, spk_id)
        else:
            _, kl_spk_loss, classify_loss, spk_emb = self.speaker_module(hid, spk_id)
        return spk_emb, list_quantEmb, list_postStats, \
            quant_loss, kl_spk_loss, classify_loss


        

""" Decoder """

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Parameter """
        n_emo = config['Dataset']['num_emotions']
        n_mel = config['Preprocess']['n_mel']
        mel_length = config['Loader']['mel_length']
        
        d_hid = config['Model']['d_decoder_hidden']
        d_emo = config['Model']['d_emotion_hidden']
        
        n_Block = config['Model']['n_DecVCBlock']
        
        """ Architecture """
        self.init_token = nn.Parameter(torch.randn(1, mel_length//(2**n_Block), d_hid))
        
        # emotion embedding
        self.emo_embedding = nn.Embedding(n_emo, d_emo)
        
        # Decoder-VQ Blocks
        self.Blocks = nn.ModuleList([
            DecVCBlock(config) for _ in range(n_Block)
        ])
        
        # last linear
        self.post_linear = nn.Linear(d_hid, n_mel)
        
    def forward(
        self, 
        z_s, 
        emo_id,
        list_quantEmb, 
        list_postStats=None
    ):
        """
        ? INPUT
        :speaker embedding: (B, C_spk)
        :emotion id: (B,), int
        :list_quantEmb: list
        :list_postStats: list
        """
        
        emo_emb = self.emo_embedding(emo_id)
        
        out = self.init_token.repeat(z_s.shape[0], 1, 1)
        kl_pitch_loss = 0.
        
        if list_postStats is not None:
            for i, (_quant, _stats) in enumerate(zip(list_quantEmb[::-1], list_postStats[::-1])):
                out, _kl = self.Blocks[i](out, _quant, z_s, emo_emb, _stats)    
                kl_pitch_loss += _kl
            
            return self.post_linear(out), kl_pitch_loss
        
        else:
            for i, _quant in enumerate(list_quantEmb[::-1]):
                out, _ = self.Blocks[i](out, _quant, z_s, emo_emb)
                
            return self.post_linear(out)

                
            

""" PostNet """

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




""" Model """

class VoiceConversionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        """ Architecture """
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, mel, spk_id, emo_id):
        z_spk, list_quantEmb, list_postStats, \
            quant_loss, kl_spk_loss, cls_spk_loss = self.encoder(mel, spk_id)
            
        output, kl_pitch_loss = self.decoder(z_spk, emo_id, list_quantEmb, list_postStats)
        
        return output, quant_loss, kl_pitch_loss, kl_spk_loss, cls_spk_loss
    
    def inference(self, mel, spk_id, emo_id):
        spk_emb, list_quantEmb, _, quant_loss, _, cls_loss = self.encoder(mel, spk_id)
        
        output = self.decoder(spk_emb, emo_id, list_quantEmb)
        
        return output, quant_loss, cls_loss




if __name__ == "__main__":
    import yaml
    config = yaml.load(
        open("../config/config.yaml", "r"), Loader=yaml.FullLoader
    )

    test_mel = torch.zeros(8, 128, 80)
    test_spk_id = torch.randint(0, 10, (8,))
    test_emo_id = torch.randint(0, 5, (8,))

    model = VoiceConversionModel(config)

    from torchsummaryX import summary
    summary(model, test_mel, spk_id=test_spk_id, emo_id=test_emo_id)


