import torch 
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # compute Y = XW +b
    # return an x d_out list of lsit of floats

    X = torch.tensor(X, dtype=torch.float32)
    W = torch.tensor(W, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    Y= X @ W +b
    return Y.tolist()
    