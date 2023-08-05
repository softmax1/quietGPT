# quietGPT

Implementing Quiet Attention (or softmax1) from the article, [Attention is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html) by training a reproduction of [nanoGPT](https://github.com/karpathy/nanoGPT). The objective is to meaasure the kurtosis of each layer's weights with the regular softmax function and with the new softmax1 function.

# installation
`pip install torch numpy transformers datasets tiktoken wandb tqdm`

# implementation
The following Quiet Attention, or Softmax1 function is implemented in code below.
```math
softmax_1(x_i) = {exp(x_i) \over 1 + \sum_{j=1}^nexp(x_j)}
```
## note on implementation
To achieve numerical stability in the softmax1 activations, the input vector, $x_i$ is shifted by the maximum value of the vector, $x$. Mathematically, this will result in the following formula to implement.
```math
softmax_1(x_i) = {exp(x_i) \over 1 + \sum_{j=1}^nexp(x_j)} \times {exp(-max(x)) \over exp(-max(x))}
```
```math
softmax_1(x_i) = {exp(x_i-max(x)) \over  exp(-max(x)) + \sum_{j=1}^nexp(x_j - max(x))}
```
```
def softmax1(x, dim=-1):    
    shift = x.max(dim=dim, keepdim=True).values
    x = x - shift
    exp_x = torch.exp(x)
    return exp_x / (torch.exp(-shift) + exp_x.sum(dim=dim, keepdim=True))
```
## empirical proof of implementation
To proof the correctness of the implementation, we compare results against vanilla softmax, which is implementation as follows
```
def softmax(x):
    shift = x.max(dim=-1, keepdim=True).values
    numerator = torch.exp(x-shift)
    denominator = numerator.sum(dim=-1)
    return numerator / denominator
```
Using a simple 1x5 vector `[1, 2, 3, 4, 5]`, the activations are as follows.
```
inp = torch.tensor([1, 2, 3, 4, 5])
softmax(inp)
Output >>> tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364])

sum(softmax(inp))
Output >>> tensor(1.)
```
Using the above softmax1 implementation, we get
```
inp = torch.tensor([1, 2, 3, 4, 5])
softmax1(inp)
Output >>> tensor([0.0116, 0.0315, 0.0858, 0.2331, 0.6337])

sum(softmax(inp))
Output >>> tensor(0.9957)
```
The activations are fairly close to vanilla softmax. We can observe the additional shift causes the sum of the probabilities to not equate to 1. Now, it is important to note that the shrinkage should be made up for during normalization, as described in Evan Miller's article. The crux of this implementation trick lies in its dealing with extreme negative values, where a model simply cannot make a decision.

From the example below, when more negative values appear, the softmax1 function does not assign additional probabilities to other classes, it instead reduces the overall probability of making a decision. On the other hand, vanilla softmax forces a decision by reassigning probabilities to other classes.
```
inp = torch.tensor([1, 2, -3, -4, -10000])

softmax(inp)
Output >>> tensor([0.2671, 0.7262, 0.0049, 0.0018, 0.0000])

sum(softmax(inp))
Output >>> tensor(1.0000)

softmax1(inp)
Output >>> tensor([0.2432, 0.6612, 0.0045, 0.0016, 0.0000])

sum(softmax1(inp))
Output >>> tensor(0.9105)

### introducing more negative extremes
inp = torch.tensor([1, 2, -32498321749821, -190487129857, -10000])

softmax(inp)
Output >>> tensor([0.2689, 0.7311, 0.0000, 0.0000, 0.0000])

sum(softmax(inp))
Output >>> tensor(1.0000)

softmax1(inp)
Output >>> tensor([0.2447, 0.6652, 0.0000, 0.0000, 0.0000])

sum(softmax1(inp))
Output >>> tensor(0.9100)

### introducing more negative values
inp = torch.tensor([-1, -2, -32498321749821, -190487129857, -10000])

softmax(inp)
Output >>> tensor([0.7311, 0.2689, 0.0000, 0.0000, 0.0000])

sum(softmax(inp))
Output >>> tensor(1.0000)

softmax1(inp)
Output >>> tensor([0.2447, 0.0900, 0.0000, 0.0000, 0.0000])

sum(softmax1(inp))
Output >>> tensor(0.3348)
```
# to train
## using the regular softmax function
`./train_regular_softmax.sh`

## using quiet attention
`./train_softmax1.sh`

# to do
1. Evaluate the perplexity scores of the model with and without quiet attention
2. Evaluate the kurtosis of the activations in addition to the kurtosis of the weights 