import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # so we have Q K V each with shape (2, 10, 64) 
    # heads is num_heads
    # Weight matrices are shape (d_model, d_model) so (64, 64)
    batch, seq_len, d_model = Q.shape
    d_head = d_model // num_heads

    # do linear projection
    Q = Q @ W_q 
    K = K @ W_k
    V = V @ W_v

    # reshape to separate heads:
    Q = Q.reshape(batch,seq_len,num_heads,d_head).transpose(0,2,1,3)
    K =K.reshape(batch,seq_len,num_heads,d_head).transpose(0,2,1,3)
    V = V.reshape(batch,seq_len,num_heads,d_head).transpose(0,2,1,3)

    # query key dot product
    score = np.einsum('bhtd,bhsd->bhts',Q,K)

    score = score // np.sqrt(d_head)
    probs = softmax(score, axis=-1)
    atten = np.einsum('bhts,bhsd->bhtd',probs,V)
    atten = atten.transpose(0,2,1,3).reshape(batch, seq_len, d_model)
    return atten @ W_o

    

    
    