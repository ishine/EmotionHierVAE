import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm




""" SoftPlus """

class SoftPlus(nn.Module):
    def forward(self, input_tensor):
        return _softplus.apply(input_tensor)

class _softplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result=torch.log(1+torch.exp(i))
        ctx.save_for_backward(i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*torch.sigmoid(ctx.saved_variables[0])




""" Adaptive Instance Norm """

class AdaIN(nn.Module):
    def __init__(self, d_hid, d_cond):
        super().__init__()

        self.linear = nn.Linear(d_cond, 2 * d_hid, bias=False)

    def forward(self, x, cond):
        """
        ? INPUT
            - x: (B, T, C)
            - cond: (B, 1, C)
        ? OUTPUT
            - (B, T, C), torch
        """
        if len(cond.shape) == 2:
            cond = cond.unsqueeze(1)
        
        scale, bias = self.linear(cond).chunk(2, dim=-1)
        return feature_norm(x, dim=-1) * (1 + scale) + bias
        
def feature_norm(x, dim=1, eps: float = 1e-14):
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = torch.square(x - mean).mean(dim=dim, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)




""" Pixel Shuffle """

class PixelShuffle(nn.Module):
    """ 
        Upsampling along time-axis + Downsampling along channel-axis.
    """
    def __init__(self, scale_factor: int):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """ 
        ? INPUT
        :x: tensor, (batch_size, channels, width)
        
        ? OUTPUT
        :return: tensor, (batch_size, out_channels, width),
            Shuffling pixels in the tensor to re-sample to suitable size,
            - channels = channels // scale_factor
            - width = width * scale_factor
        """ 
        batch_size, channels, in_width = x.size()
        channels = channels // self.scale_factor
        out_width = in_width * self.scale_factor
        
        x = x.contiguous().view(batch_size, channels, self.scale_factor, in_width)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, channels, out_width)
        return x



class InversePixelShuffle(nn.Module):
    """ 
        Downsampling along time-axis + Upsampling along channel-axis.
    """
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """ 
        ? INPUT
        :x: tensor, (batch_size, in_channels, width)
        
        ? OUTPUT
        :return: tensor, (batch_size, out_channels, width),
            Shuffling pixels in the tensor to re-sample to suitable size,
            - out_channels = in_channels * scale_factor
            - width = width // scale_factor
        """ 

        batch_size, in_channels, width = x.size()
        out_channels = in_channels * self.scale_factor
        width = width // self.scale_factor

        x = x.contiguous().view(batch_size, in_channels, width, self.scale_factor)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, out_channels, width)
        return x




""" Quantize"""

class Quantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-12):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


