import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # shape for Q,K,V: (batch, seq_len, dim)
    batch, seq_len, dim = Q.shape
    score = torch.einsum('btd,bsd->bts',Q,K)
    score = score / math.sqrt(dim)

    probs = F.softmax(score, dim=-1)

    return torch.einsum('bts,bsd->btd',probs,V)
    

    
    