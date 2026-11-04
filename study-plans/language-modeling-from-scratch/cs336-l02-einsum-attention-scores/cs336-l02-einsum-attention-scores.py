import torch
import math
def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    # given q k num of heads

    batch_size, seq_len, d_model = q.shape
    key_len = k.shape[1] # we need to extract hte shape of the 
    # seq_len of the key

    d_head = d_model // num_heads

    q = q.reshape(batch_size, seq_len, num_heads, d_head)
    k = k.reshape(batch_size, key_len, num_heads, d_head)
    score = torch.einsum('bshd,bthd->bhst', q,k)

    return score / math.sqrt(d_head)
    
