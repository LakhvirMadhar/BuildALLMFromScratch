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

**Try it:** Print `model.layers[0].weight.shape` and predict it first. Then count every learnable number with `sum(p.numel() for p in model.parameters())` and check it against this: `50 × 30 + 30 + 30 × 20 + 20 + 20 × 3 + 3 = 2,213`.

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
- [ ] Next: page 267, the rest of A.5
