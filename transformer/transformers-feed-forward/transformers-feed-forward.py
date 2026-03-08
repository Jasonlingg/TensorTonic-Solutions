import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    # so inputs shapes:
    # x (2,10,64) with d model 64 and dff 256
    # output sill teh same shape after ffn transformation

    # first hidden layer: Expand
    # the input vecotr is projected to a HIGHER DIMENSION
    # dff is 256, WHY? creates a richer representation of more 
    # room for complex feature interactions
    hidden = np.einsum('btd,df->btf',x,W1) + b1

    # Activate RELU
    # applies non - linearity where negative values are set to 0
    # prevent linear transformations from collapsing into a xW1W2
    # gives expressiveness
    hidden_relu = np.maximum(0,hidden)

    # project back to original dims
    output = np.einsum('btf,fd->btd',hidden_relu,W2) + b2
    return output
    