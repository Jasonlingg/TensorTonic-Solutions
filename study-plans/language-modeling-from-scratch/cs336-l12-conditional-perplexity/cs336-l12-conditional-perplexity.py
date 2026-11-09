import torch

def conditional_perplexity(
    logits: torch.Tensor, target_ids: torch.Tensor, response_mask: torch.Tensor,
) -> dict:
    """
    Returns a dict: negative_log_likelihood (float), perplexity (float), selected_token_count (int).
    """

    # give logits (batch, seq_len, vocab) with at le
    # targe_ids and response mask are shape (batch, seq)
    # target id are valid vocab incides, maks also may slect no tokens

    # compute the negative log likelilood but need to figure out how many 
    # response tokens

    # since the # jsut equal hwo amny 1s there are in the maks we jsut sum

    num_token = int(response_mask.sum())

    # case theres no tokens
    if num_token == 0:
        return {
        "negative_log_likelihood":0.0,
         "perplexity":1.0,
         "selected_token_count":0
        }

    # in the case that there is tokens
    # we need to get het sum of the masked tokens wit hthe log probs
    # log softmax
    probs = torch.log_softmax(logits , dim=-1)
    #  unsqueeze at hte last dimension
    expand_target_id = target_ids.unsqueeze(-1)

    # find the target log probs using torch.gather
    # and also resqueeze the last column
    target_logprobs = torch.gather(probs, dim=-1, index=expand_target_id).squeeze(-1)

    # apply the mask, need to chagne into bool
    bool_mask = response_mask.bool()
    # use torch.masked_select to apply the mask
    valid_probs = torch.masked_select(target_logprobs, bool_mask)

    NLL =  -valid_probs.mean()

    return {
        "negative_log_likelihood": NLL.item(),
         "perplexity": torch.exp(NLL).item(),
         "selected_token_count": num_token
    }
    

    
    