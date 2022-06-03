import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, config):
        super(VarianceAdaptor, self).__init__()

        out_size = config["Model"]["Reconstruction"]["d_contents"]

        self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()

        self.pitch_embedding = Conv(1, out_size, kernel_size=9, bias=False, padding=4, padding_mode="replicate")
        self.energy_embedding = Conv(1, out_size, kernel_size=9, bias=False, padding=4, padding_mode="replicate")

    def forward(self, connection, p_target=None, e_target=None):
        pitch_prediction = self.pitch_predictor(connection)
        if self.training:
            # If Training Stage, predict embedding vector from a target pitch
            pitch_embedding = self.pitch_embedding_producer(p_target.unsqueeze(2))
        else:
            # Else if inference stage, predict embedding vector directly from the trunk (pitch prediction).
            pitch_embedding = self.pitch_embedding_producer(pitch_prediction.unsqueeze(2))

        energy_prediction = self.energy_predictor(connection)
        if self.training:
            # If Training Stage, predict embedding vector from a target energy
            energy_embedding = self.energy_embedding_producer(e_target.unsqueeze(2))
        else:
            # Else if inference stage, predict embedding vector directly from the trunk (energy prediction).
            energy_embedding = self.energy_embedding_producer(energy_prediction.unsqueeze(2))

        return pitch_embedding + energy_embedding



class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, config):
        super(VariancePredictor, self).__init__()

        self.input_size = config["Model"]["Reconstruction"]["d_contents"]
        
        self.filter_size = config["Model"]["Predictor"]["d_hid"]
        self.kernel = config["Model"]["Predictor"]["kernel_size"]
        self.conv_output_size = config["Model"]["Predictor"]["d_hid"]
        self.dropout = config["Model"]["Predictor"]["dropout"]

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=(self.kernel-1)//2)),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        
        return out



class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='replicate'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode,
                              dilation=dilation,
                              bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
