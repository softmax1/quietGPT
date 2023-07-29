# quietGPT

Implementing Quiet Attention (or softmax1) from the paper, [Attention is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) by training a reproduction of [nanoGPT](https://github.com/karpathy/nanoGPT). The objective is to meaasure the kurtosis of each layer's weights with the regular softmax function and with the new softmax1 function.

# installation
`pip install torch numpy transformers datasets tiktoken wandb tqdm`

# implementation
The following Quiet Attention, or Softmax1 function is implemented in code below.
```math
(softmax_one(x))_i = exp(x_i) / (1 + sum(exp(x_j) for all j))
```
```
def softmax1(x, dim=None):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))
```

# to train
## using the regular softmax function
`./train_regular_softmax.sh`

## using quiet attention
`./train_softmax1.sh`

