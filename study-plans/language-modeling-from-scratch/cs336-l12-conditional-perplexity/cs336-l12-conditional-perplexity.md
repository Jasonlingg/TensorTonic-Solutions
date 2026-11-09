# Compute Conditional Perplexity

Perplexity is the standard way to measure how well a language model predicts text, and for chat or instruction data you usually want to score only the response, because the prompt was supplied to the model. Conditional perplexity scores just the selected response tokens. The heart of this problem is averaging the negative log likelihood over exactly the masked tokens and turning that average into perplexity.

## From probabilities to negative log likelihood

At each position a language model outputs logits, one score per vocabulary token, and a softmax turns them into a probability distribution. If $x_i$ is the actual token at position $i$, the model's probability for it is $p_i(x_i)$. A good model gives the actual tokens high probability.

The probability of a whole sequence is the product of its per-token probabilities, which quickly becomes a vanishingly small number. Working with $-\log p_i(x_i)$ instead turns that product into a sum and gives a quantity that is 0 for a perfect prediction and grows as the prediction gets worse. This is the same cross-entropy loss used to train the model.

## Scoring only the response

For a prompt followed by a response, the prompt tokens were given to the model, so predicting them is not part of the task being evaluated. Padding tokens carry no meaning at all. A mask $m_i$ marks the positions to score, 1 for selected response tokens and 0 for everything else. With $M = \sum_i m_i$ selected tokens, the average negative log likelihood is

$$
\text{NLL} = -\frac{1}{M} \sum_i m_i \log p_i(x_i)
$$

The average runs over all selected tokens in the whole batch. Averaging each sequence first and then averaging the sequences would give a short response the same weight as a long one, so every token would no longer count equally.

During training a model reads tokens up to position $t$ and its logits there predict token $t + 1$, so code usually shifts targets left by one. Here the targets were shifted in advance, so shifting again would compare each prediction with the wrong token.

## Perplexity as an effective number of choices

Perplexity is the exponential of the average negative log likelihood:

$$
\text{PPL} = \exp(\text{NLL})
$$

A perplexity of $k$ can be read as the model being, on average, as uncertain as a uniform choice among $k$ tokens at each step. A perfect model has perplexity 1, and a model that guesses uniformly over a vocabulary of size $V$ has perplexity $V$. With no selected tokens there is nothing to average, so the result falls back to an NLL of 0 and a perplexity of 1.

## Why log-softmax beats the log of a softmax

Computing a softmax and then taking its logarithm can underflow to $\log 0$ when one logit is much larger than the others. A log-softmax computes the log probabilities directly by subtracting the largest logit first, which stays accurate for any logit values. Taking the final mean in float64 also means that summing many small values loses far less precision.

## Two sequences of different lengths

Take a batch of two sequences over a vocabulary of two tokens.

Sequence A has a prompt position followed by two response tokens, and at both of them the model gives the target a probability of 0.5. Sequence B has a prompt position, one response token and a padding position. At its response token the logits are $[\log 3, 0]$, so the softmax gives $[0.75, 0.25]$, and the target is token 1, with probability 0.25.

The three selected tokens contribute $\log 2$, $\log 2$ and $\log 4$ to the sum of negative log probabilities, so

$$
\text{NLL} = \frac{2 \log 2 + \log 4}{3} = \tfrac{4}{3} \log 2 \approx 0.924196
$$

and the perplexity is $2^{4/3} \approx 2.519842$.

Averaging per sequence first would give $(\log 2 + \log 4) / 2 \approx 1.040$ instead, because sequence B's single hard token would count as much as both of sequence A's tokens together.

## Common mistakes to avoid

- **Including prompt or padding tokens.** Only positions selected by the mask contribute to the average.
- **Averaging per sequence first.** The mean runs over all selected tokens, so long responses count in proportion to their length.
- **Shifting targets again.** The targets are already aligned with the logits.
- **Taking the log of a softmax.** A log-softmax avoids underflow for large logit gaps.
- **Exponentiating each token's loss.** Averaging per-token exponentials overweights the worst tokens, so the exponential is taken once, of the average.
