import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # implement sinusoidal postional encoding that ADDS position information to token embeddings
    # Transformers have no inherent notion of position
    # We add positional encodings TO the input embeddings

    # Requirements 
    # return a PE matrix of shape (seq_length, dmodel)
    # use sine for even indcies 
    # cose ofr odd indices
    # valeus must be in rnage[-1,1]

    # Solution:
    # loop through i in seq length and for each dimension in d_model?

    PE = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for dim in range(d_model):
            angle = pos/(10000 ** (2* dim // d_model))
            if dim % 2 == 0:
                # that entry in the pos row, dim col is that sinsosudal value
                PE[pos,dim]= np.sin(angle)
            else:
                PE[pos,dim] = np.cos(angle)
    return PE
    