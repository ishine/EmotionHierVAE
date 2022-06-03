import torch
import torch.nn as nn
import torch.nn.functional as F


def AMC_loss(spk_emb, spk_id, tot_class, m_g = 0.5, eps=1e-14):
    """
    ? INPUT
    - spk_emb: (B, spk_channel)
    - spk_id: (B, )
    """
    if spk_emb.dtype != torch.float64:
        spk_emb = spk_emb.to(torch.float64)
        
    # normalize speaker embedding
    emb = spk_emb / torch.norm(spk_emb, p=2, dim=-1, keepdim=True)

    # matrix S_ij (equal 1 when i == j)
    _one_hot = F.one_hot(spk_id, num_classes = tot_class).float()   # (B, tot_class)
    S = torch.matmul(_one_hot, _one_hot.T).bool()                   # (B, B)

    # calculate loss (attract if S_ij == 1, else S_ij == 0)
    inner_prod = (emb @ emb.T).clamp(min=-1+eps, max=1-eps)               # (B, B)

    _loss_attract = torch.acos(inner_prod) ** 2
    _loss_repulse = torch.clamp(m_g - torch.acos(inner_prod), min=0.) ** 2
    _loss = torch.where(S, _loss_attract, _loss_repulse)

    return torch.sum(_loss) / (spk_emb.shape[0])


if __name__ == "__main__":
    spk_emb = torch.randn(8, 128)
    spk_id = torch.randint(10, (8,))

    print(AMC_loss(spk_emb, spk_id, 10))