def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    """
    # list of matmul and attention_flops
    # flops are set to 0 

    # flops totla is equal to Flops foward + flops backward

    # what is matmuls? 
    #  list of list like this: 
    # [ (B,D,K), (B,D,K) ...]

    # counter for flops so we can do f backward and f total
    flops_forward = 0 
    for entry in matmuls: 
        # extract B D K dimensions
        (B, D, K) = entry
        flops_forward += (2 * int(B) * int(D) * int(K)) 

    flops_forward += + int(attention_flops)

    # extracted flops now we return
    return {"forward_flops": flops_forward,"backward_flops": 2 * flops_forward,"total_flops": 3 * flops_forward}
        

    