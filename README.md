# quietGPT

Implementing Quiet Attention (or softmax1) from the paper, [Attention is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) by training a reproduction of [nanoGPT](https://github.com/karpathy/nanoGPT). The objective is to meaasure the kurtosis of each layer's weights with the regular softmax function and with the new softmax1 function.

# installation
`pip install torch numpy transformers datasets tiktoken wandb tqdm`

# implementation
The following Quiet Attention, or Softmax1 function is implemented in code below.
```math
(softmax_one(x))_i = exp(x_i) / (1 + sum(exp(x_j) for all j))
```
## Note on implementation
To achieve numerical stability in the softmax1 activations, the input vector is shifted by the maximum value. Mathematically, this will result in the following formula to implement.
```math
(softmax_one(x))_i = exp(x_i - max(x_i)) / (1 + \sum_{j=1}^n(exp(x_j)))
al;hf
```
```
def softmax1(x, dim=-1):
    shift = x.max(dim=dim, keepdim=True).values
    x = x- shift
    exp_x = torch.exp(x)
    return exp_x / (torch.exp(-shift) + exp_x.sum(dim=dim, keepdim=True))
```

# to train
## using the regular softmax function
`./train_regular_softmax.sh`

## using quiet attention
`./train_softmax1.sh`

# to do
1. Investigate the effects of shifting/not shifting the zeros in softmax1 implementation Refer to [Issue #1](https://github.com/softmax1/quietGPT/issues/1)
2. Evaluate the perplexity scores of the model with and without quiet attention
3. Evaluate the kurtosis of the activations in addition to the kurtosis of the weights 