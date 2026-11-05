import torch
import math

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    # We are given a query with a new key and value
    # we have the key ac eand value cahce and num of query head and num of KV heads

    # lets go throguh the shapes of each
    # q k and v: (batch, head, seq_len (should be 1), dimension)

    # caches: 
    # shape is (batch, head, seq_len of past token, dim)

    # append new key and new value to the matrix
    key_cache = torch.cat([key_cache, new_key.unsqueeze(2)], dim=2)
    value_cache = torch.cat([value_cache, new_value.unsqueeze(2)], dim=2)

    groups = num_query_heads // num_kv_heads

    batch_size , _ , dim = query.shape

    query = query.reshape(batch_size, num_kv_heads, groups, 1, dim)

    scores = torch.einsum('bhgsd,bhtd->bhgst', query, key_cache)

    scores = scores / (math.sqrt(dim)) 

    probs = torch.softmax(scores, dim=-1)

    attention = torch.einsum('bhgst,bhtd->bhgsd',probs,value_cache)

    return {
    "output": attention.reshape(batch_size, num_query_heads, dim),
    "new_key_cache": key_cache,
    "new_value_cache": value_cache,
}

    

    
    

    
