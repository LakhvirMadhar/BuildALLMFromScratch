# Lesson 02: Multilayer Perceptron

## What We're Doing

Building a small neural network as a `torch.nn.Module` subclass: two hidden layers and an output layer, chained with `nn.Sequential`.

**Book sections:** Appendix A.5 (pages 265–266, Figure A.9 and Listing A.4).
**Code:** `src/build_llm_from_scratch/chapter_00_learn_pytorch/auto_diff_engine/02_multilayer_perceptron.py`

## Why / Context

Lesson 01 had one weight, one bias, and one input: `z = x1 * w1 + b`. A real model needs many inputs and many outputs, stacked in layers so each layer can build on what the one before it found.

`nn.Module` is how PyTorch packages that. Every model in this book is an `nn.Module` subclass, including the attention class in Chapter 3 and the full `GPTModel` in Chapter 4. The pattern you type here (`__init__` builds the layers, `forward` runs data through them) is the one you'll type for the rest of the book.

## Concepts

### Figure A.9 - the network as a diagram (page 265)

Figure A.9 draws a **multilayer perceptron**: a network where every node in one layer connects to every node in the next. Top to bottom:

- **Inputs** - 10 white circles: the input features (10 numbers describing one example). The grey circle at the end is the **bias unit**, the `b` from lesson 01.
- **1st hidden layer** - 6 nodes, plus its own bias unit.
- **2nd hidden layer** - 4 nodes, plus a bias unit.
- **Output layer** - 3 nodes: the 3 numbers the network produces, one per class.
- **The lines (edges)** - each line is one **weight**. Every input connects to every node below it, which is why it's called **fully connected**.

"Hidden" just means "neither the input nor the output": you never see these values directly, only what comes out at the bottom.

Each node does exactly what lesson 01's whole model did: multiply each input by a weight, add them up, add a bias. The figure is lesson 01's `z = x1 * w1 + b` repeated for every node, with every node's output feeding the next layer.

### "Is that really all the code for a neural network?"

Yes. Listing A.4 is a complete, real neural network. The diagram looks scarier than the code because the diagram draws every single weight, and the code describes them in bulk.

Count the lines in Figure A.9: (10 inputs + 1 bias) × 6 nodes + (6 + 1) × 4 + (4 + 1) × 3 = **109 lines**, one per number the network learns. In code, that whole figure is three lines:

```python
torch.nn.Linear(10, 6)
torch.nn.Linear(6, 4)
torch.nn.Linear(4, 3)
```

The listing's own network (50 inputs, layers of 30 and 20, 3 outputs) has **2,213** learnable numbers. A diagram of it would be a solid black smear of lines. The code is still five lines.

You're right that a lot is going on underneath. PyTorch is doing all of this for you in those few lines:

- **Creating the weights** - a `(30, 50)` weight tensor and a `(30,)` bias tensor for the first layer, filled with sensible random starting values.
- **The maths** - every node's multiply-and-add, done as one matrix multiplication, on the GPU if you ask.
- **Registering the weights** - so `model.parameters()` can find all 2,213 numbers and an optimizer can update them.
- **Tracking for autograd** - recording every operation so `loss.backward()` can compute gradients for every weight (A.4).

#### Does it get more complicated with more layers?

Not really. More layers means more lines in `Sequential`, nothing else. Ten hidden layers is ten more `Linear` + `ReLU` pairs.

What gets more complicated in later chapters is the *kind* of layer, not the number. GPT-2 still follows this exact shape (a `Module` with layers built in `__init__` and run in `forward`), but its layers are smarter: attention in Chapter 3, layer normalization and shortcut connections in Chapter 4. Even so, the whole GPT model you'll build is a few hundred lines. The scale comes from the sizes (124 million weights), not from the amount of code.

### `torch.nn.Linear` - a layer of lesson 01's formula

`Linear(in_features, out_features)` is a layer of nodes, each computing `w · x + b`. It holds:

- `weight` - shape `(out_features, in_features)`. For `Linear(50, 30)`: `(30, 50)`, one row of 50 weights per node.
- `bias` - shape `(out_features,)`. For `Linear(50, 30)`: `(30,)`, one bias per node.

**Plain Python** (a `Linear(3, 2)`: 3 inputs, 2 nodes):
```python
inputs = [1.0, 2.0, 3.0]
weights = [[0.1, 0.2, 0.3],   # node 0's weights
           [0.4, 0.5, 0.6]]   # node 1's weights
biases = [0.0, 1.0]

outputs = [
    sum(w * x for w, x in zip(node_weights, inputs)) + node_bias
    for node_weights, node_bias in zip(weights, biases)
]
# [1.4, 4.2]  -> one output per node
```

**PyTorch:**
```python
layer = torch.nn.Linear(3, 2)
layer(torch.tensor([1.0, 2.0, 3.0]))   # shape (2,), one output per node
```

**Template:**
```python
layer = torch.nn.Linear(num_inputs, num_outputs)
outputs = layer(inputs)   # (..., num_inputs) -> (..., num_outputs)
```

The PyTorch version gives different numbers from the plain-Python one because its weights start random rather than being the ones typed above. The calculation is identical: each output is one node's `w · x + b`.

**Try it:** Create a `torch.nn.Linear(4, 2)` and print `.weight.shape` and `.bias.shape`. Predict both before running. Then call it on `torch.rand(4)` and predict the output shape.

### `torch.nn.ReLU` - the nonlinearity between layers

ReLU replaces every negative number with 0 and leaves positive numbers alone.

**Plain Python:**
```python
values = [-2.0, 0.0, 3.0]
[max(0.0, v) for v in values]   # [0.0, 0.0, 3.0]
```

**PyTorch:**
```python
torch.nn.ReLU()(torch.tensor([-2.0, 0.0, 3.0]))   # tensor([0., 0., 3.])
```

**Template:**
```python
activated = torch.nn.ReLU()(values)
```

It sits between the `Linear` layers because stacking `Linear` layers with nothing between them is pointless. A `Linear` is "multiply and add," and multiply-and-add followed by another multiply-and-add is still just one multiply-and-add. Three of them in a row could be squashed into a single `Linear`, so the extra layers would add nothing.

ReLU breaks that. Zeroing the negatives is a bend in an otherwise straight line, and stacking bends lets the network fit curved, complicated patterns. It plays the same role sigmoid did in lesson 01, just a simpler and faster shape.

#### Where the Name Comes From

**ReLU** stands for **Re**ctified **L**inear **U**nit. *Rectify* comes from electronics: a rectifier is a circuit that lets current flow in one direction only, blocking the negative half. ReLU does the same to numbers, letting positives through and blocking negatives. "Linear unit" because for positive inputs it's a plain straight line.

**Perceptron** was coined by Frank Rosenblatt in 1957 for an early machine that learned to recognize simple images: *percept* (something perceived) + *-tron*, a suffix used for devices (like *cyclotron*). A **multilayer** perceptron is several layers of them stacked.

### `torch.nn.Sequential` - run the layers in order

`Sequential` takes a list of layers and runs them one after another, feeding each one's output into the next.

**Plain Python:**
```python
def run_in_order(layers, x):
    for layer in layers:
        x = layer(x)
    return x
```

**PyTorch:**
```python
self.layers = torch.nn.Sequential(layer_1, activation_1, layer_2, activation_2, output_layer)
self.layers(x)
```

**Template:**
```python
self.layers = torch.nn.Sequential(first, second, third)
output = self.layers(inputs)
```

The rule the listing's annotations point out: **each layer's output size must match the next layer's input size.** `Linear(num_inputs, 30)` outputs 30, so the next is `Linear(30, 20)`, which outputs 20, so the last is `Linear(20, num_outputs)`. ReLU doesn't change the size, so it doesn't affect the chain.

Following one example through the listing's network with `model = NeuralNetwork(50, 3)`:

| After | Shape |
|---|---|
| input | `(1, 50)` - 1 example, 50 features |
| `Linear(50, 30)` + `ReLU` | `(1, 30)` |
| `Linear(30, 20)` + `ReLU` | `(1, 20)` |
| `Linear(20, 3)` | `(1, 3)` - 1 example, 3 scores |

Pass a batch of 8 examples, shape `(8, 50)`, and you get `(8, 3)`. The first dimension (batch) passes through untouched.

Page 266 ends mid-sentence explaining why `Sequential` is convenient ("...we just have to"); it continues on page 267.

**Try it:** Without running anything, write down the shape after each layer for a batch of 4 examples through `NeuralNetwork(50, 3)`. Then build `torch.rand(4, 50)`, pass it through `model`, and check the final shape.

### Listing A.4 - the `nn.Module` subclass (page 266)

Decoding each part:

- `class NeuralNetwork(torch.nn.Module)` - your model inherits from `Module`, which supplies parameter tracking, `.to(device)`, `print(model)`, saving and loading.
- `__init__(self, num_inputs, num_outputs)` - builds the layers. Taking the sizes as arguments means the same class works for any dataset.
- `super().__init__()` - runs `Module`'s own setup first. Without it, `Module` hasn't prepared its internal bookkeeping, and assigning `self.layers` fails. It must come before you assign any layers.
- `self.layers = torch.nn.Sequential(...)` - assigning a layer to `self` is what registers it. `Module` notices the assignment and adds its weights to `model.parameters()`.
- `forward(self, x)` - describes how data flows through the layers. You never call `forward` directly; you call `model(x)`, and `Module` calls `forward` for you (plus some hooks it needs).
- `logits` - the name for the raw output scores of the last layer, before anything like sigmoid or softmax turns them into probabilities.

**Plain Python** (the same shape of class, no PyTorch):
```python
class Doubler:
    def __call__(self, x: float) -> float:
        return self.forward(x)

    def forward(self, x: float) -> float:
        return x * 2

Doubler()(5)   # 10 - calling the object runs forward
```

`nn.Module` works like `Doubler`: its `__call__` runs your `forward`. That's why `model(x)` works.

**Template:**
```python
class ModelName(torch.nn.Module):
    def __init__(self, sizes: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.layers(x)
        return output
```

#### Where the Name Comes From

**Logit** is a contraction of "**log**istic un**it**," coined by statistician Joseph Berkson in 1944, by analogy with *probit* ("**prob**ability un**it**"). In statistics, the logit is the function `log(p / (1 - p))`, which stretches a probability in 0–1 out to any number. Run backwards, that's sigmoid from lesson 01. Deep learning borrowed the word loosely: logits are the raw scores sitting on the "before the squash" side of sigmoid or softmax. In lesson 01, `z` was a logit.

### `print(model)` - reading the summary

```
NeuralNetwork(
  (layers): Sequential(
    (0): Linear(in_features=50, out_features=30, bias=True)
    (1): ReLU()
    ...
```

- `(layers)` - the attribute name you assigned in `__init__`.
- `(0)`, `(1)`, ... - each layer's position inside `Sequential`. You can index it: `model.layers[0]` is the first `Linear`.
- `in_features` / `out_features` - the two arguments you passed to `Linear`.
- `bias=True` - the layer has a bias (the default). `Linear(50, 30, bias=False)` would leave it out.

**Try it:** Print `model.layers[0].weight.shape` and predict it first.

### `y = mx + b` in ML notation

From school algebra, `y = mx + b` is a straight line: `m` is the slope, `b` is where the line crosses the y-axis. One input `x`, one output `y`, two numbers that set the line's shape.

ML uses the same formula with different letters and more inputs:

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

- `m` becomes `w` - a **weight**. Same job as the slope: how much this input pushes the output up or down.
- `b` stays `b` - the **bias**. Same job as the intercept: a starting offset added no matter what the inputs are.
- `x` becomes `x1 ... xn` - one example has many input features, and each gets its own weight.
- `y` is often called `z` - the raw output before any activation, as in lesson 01.

Lesson 01's `z = x1 * w1 + b` was exactly `y = mx + b`, renamed: one input, one weight, one bias.

A `Linear(in_features, out_features)` layer is `out_features` of these formulas side by side, one per node. Each node has its own `in_features` weights and its own bias. Written as matrices, that's `y = Wx + b`: `W` holds every node's weights, shape `(out_features, in_features)`, and `b` holds every node's bias, shape `(out_features,)`.

#### Building up from one `m` to many, with numbers

The jump from `y = mx + b` to `y = Wx + b` happens in two steps. Each step adds one thing.

**Step 1 - one input, one output.** Plain `y = mx + b`.

Predicting a house price `y` from its size `x`, with `m = 3` and `b = 10`:

```
x = 2
y = 3*2 + 10 = 16
```

Parameters: `m` and `b`, so **2**. As a layer, this is `Linear(1, 1)`.

**Step 2 - more inputs, still one output.** Now the house has two features: size `x1` and number of rooms `x2`. Each input gets its own `m`, but there's still one `b`, because there's still one output:

```
x1 = 2, x2 = 4
w1 = 3, w2 = 0.5, b = 10

y = w1*x1 + w2*x2 + b
  = 3*2 + 0.5*4 + 10
  = 6 + 2 + 10
  = 18
```

Each input's contribution gets added up, then the bias is added once. That "multiply matching pairs, then add" is a **dot product**, `w · x`, which you met in A.2.

Parameters: 2 weights + 1 bias = **3**. As a layer, this is `Linear(2, 1)`: one node.

**Step 3 - more inputs, more outputs.** Now you want three outputs from the same two inputs. Say three different scores, like the 3 output nodes in Figure A.9.

Each output is its own Step 2 formula. Each one has its own pair of weights and its own bias:

```
y1 = 3*x1   + 0.5*x2 + 10
y2 = -1*x1  + 2*x2   + 0
y3 = 0.2*x1 + 1*x2   + -4
```

Stack those weights into a grid, one row per output and one column per input, and you get a matrix:

```
         x1    x2
W = [[  3,   0.5 ],   <- y1's weights
     [ -1,   2   ],   <- y2's weights
     [ 0.2,  1   ]]   <- y3's weights

b = [10, 0, -4]       <- one bias per output
```

That grid is `W`, shape `(3, 2)`, which is `(out_features, in_features)`. `y = Wx + b` is shorthand for "run all three Step 2 formulas at once." Each row of `W` dotted with `x`, plus that row's bias.

Parameters: 3 × 2 weights + 3 biases = **9**. As a layer, this is `Linear(2, 3)`, and `(in × out) + out = 2×3 + 3 = 9`.

So the counting formula falls straight out of the grid: **one weight per cell (`in × out`), plus one bias per row (`out`).**

**Plain Python** (Step 3 with lists):
```python
x = [2.0, 4.0]
W = [[3.0, 0.5], [-1.0, 2.0], [0.2, 1.0]]
b = [10.0, 0.0, -4.0]

y = [sum(w * xi for w, xi in zip(row, x)) + bias for row, bias in zip(W, b)]
# [18.0, 6.0, 0.4]
```

**PyTorch:**
```python
x = torch.tensor([2.0, 4.0])
W = torch.tensor([[3.0, 0.5], [-1.0, 2.0], [0.2, 1.0]])
b = torch.tensor([10.0, 0.0, -4.0])

y = W @ x + b   # shape (3,)
```

`torch.nn.Linear(2, 3)` stores a `W` of shape `(3, 2)` and a `b` of shape `(3,)`, exactly like these. It just fills them with random starting values instead of the ones typed here, and training adjusts them.

**Template:**
```python
y = W @ x + b   # W: (out_features, in_features), x: (in_features,), b: (out_features,)
```

**Try it:** Work out `y2` and `y3` from Step 3 by hand, using `x1 = 2, x2 = 4`. Check them against the plain-Python result. Then change Step 3 to **four** outputs by adding a row of your own, and predict the new shapes of `W`, `b`, and `y` before running the PyTorch version.

### Counting parameters (Exercise A.3, page 267)

A **parameter** is one learnable number: one weight or one bias. Training adjusts every one of them. Counting them tells you how big a model is. "GPT-2 has 124 million parameters" means exactly this count.

Only `Linear` layers have parameters here. `ReLU` is a fixed rule (zero the negatives) with nothing to learn.

Per `Linear(in_features, out_features)` layer:

```
params = (in_features × out_features) + out_features
          weights                        biases
```

The total is that, summed over every `Linear` in the model.

**Plain Python** (counting by hand from each layer's sizes):
```python
layer_sizes = [(50, 30), (30, 20), (20, 3)]   # (in_features, out_features) per Linear

total = 0
for in_features, out_features in layer_sizes:
    total += in_features * out_features + out_features
```

**PyTorch** (asking the model to count its own tensors, from page 267):
```python
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Decoding it piece by piece:

- `model.parameters()` - yields every weight and bias tensor the model registered. For this network: 3 `Linear` layers × (1 weight + 1 bias) = 6 tensors.
- `for p in ...` - `p` is one of those tensors, e.g. the first layer's weight, shape `(30, 50)`.
- `p.numel()` - "**num**ber of **el**ements": how many numbers are in the tensor. For shape `(30, 50)`, that's `30 × 50 = 1500`. For a bias of shape `(30,)`, it's `30`.
- `if p.requires_grad` - keep only tensors training will update. In this model that's all of them; it matters later when layers get **frozen** (Chapter 6) and should be left out of the "trainable" count.
- `sum(... for ...)` - a generator expression, the same as the plain-Python loop above: add up every tensor's count.

The plain-Python version reasons from the layer sizes; the PyTorch version counts the tensors those sizes created. They give the same number because `numel()` of a `(out, in)` weight is `in × out`, and `numel()` of a `(out,)` bias is `out`.

**Template:**
```python
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**Try it:** Exercise A.3 asks for the parameter count of the network in Listing A.9 (the one in your `01_data_loader.py`). By hand first:

1. Write down each `Linear` layer's `in_features` and `out_features`, plugging in the sizes that listing passes.
2. Write the expanded sum, one `(in × out) + out` term per layer.
3. Do the arithmetic.
4. Only then run the `num_params` line on that model and check your answer.

### Notes on your `02_multilayer_perceptron.py`

- **`super.__init__()` should be `super().__init__()`.** Without the parentheses, `super` is Python's built-in `super` type rather than your parent class, so the call fails with `TypeError: descriptor '__init__' of 'super' object needs an argument`.
- **Drop the `# type: ignore` comments.** That ignore is what hid the bug above: mypy would have flagged the bad call before you ran anything. Add type hints instead: `num_inputs: int, num_outputs: int` and `-> None` on `__init__`, and `x: torch.Tensor` with `-> torch.Tensor` on `forward`.
- **mypy gotcha in `forward`:** calling a module (`self.layers(x)`) is typed as returning `Any`, so strict mypy rejects returning it from a function declared `-> torch.Tensor`. Annotate the variable, `logits: torch.Tensor = self.layers(x)`, and it passes.

## Your Task

- [ ] Fix `super().__init__()` and replace the `# type: ignore` comments with type hints, so mypy and ruff pass
- [ ] Run the file and compare your `print(model)` output to page 266
- [ ] Check `Linear` weight and bias shapes against your predictions
- [ ] Trace a batch of 4 examples through the network, shape by shape
- [ ] Count the parameters and match 2,213
- [ ] Exercise A.3: count Listing A.9's parameters by hand, then check with `num_params`
- [ ] Next: page 267, the rest of A.5
