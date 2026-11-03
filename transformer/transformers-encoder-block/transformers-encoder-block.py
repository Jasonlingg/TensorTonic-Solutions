import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    return gamma * (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(
    np.var(x, axis=-1, keepdims=True) + eps
) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """
    # get the dimensions
    batch_size, seq_len, d_model = Q.shape
    d_head = d_model // num_heads

    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    Q = Q.reshape(batch_size, seq_len, num_heads, d_head)
    K = K.reshape(batch_size, seq_len, num_heads, d_head)
    V = V.reshape(batch_size, seq_len, num_heads, d_head)

    scores = np.einsum('bshd,bthd->bhst', Q,K) / np.sqrt(d_head)
    scores = softmax(scores, axis=-1)

    attention = np.einsum('bhst,bthd->bshd',scores, V)
    # reshape back 

    return attention.reshape(batch_size,seq_len,d_model) @ W_o
    

    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    # just do relu 

    trans = x @ W1 + b1
    hidden = np.maximum(trans, 0)

    return (hidden @ W2 + b2)
    
    
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """

    # self attention
    self_attention = multi_head_attention(
    x, x, x, W_q, W_k, W_v, W_o, num_heads
    )

    # now we have hte self attention we do the add and norm
    layer1 = layer_norm(x + self_attention, gamma1, beta1)

    ff = feed_forward(layer1, W1, b1, W2, b2)

    layer2 = layer_norm(layer1 + ff, gamma2, beta2)

    return layer2
    
    
