# Lesson 01: Logistic Regression Forward Pass

## What We're Doing

Running one input through a tiny model (a weight, a bias, and a sigmoid) and measuring how wrong its prediction is with a loss function.

**Book sections:** Appendix A.3 (page 262), leading into A.4 (page 263).
**Code:** `src/build_llm_from_scratch/chapter_00_learn_pytorch/auto_diff_engine/01_logistic_regression_forward_pass.py`

## Why / Context

Training any model, from this one-weight toy to GPT-2, follows the same loop:

```
input → (w·x + b) → sigmoid → prediction → compare to truth → loss → gradients → update weights
```

This lesson covers the first half, up to the loss. That part is called the **forward pass**: data flows forward through the model and comes out as a prediction and a score for how wrong it was.

The second half (gradients and weight updates) is what autograd does, and it's the next step in the book. It all hinges on the loss: autograd's whole job is to work out how to change each weight to make that one number smaller. That's why this file lives in `auto_diff_engine/`.

An LLM runs this exact pipeline. It just has billions of weights instead of one, and predicts one of 50,257 tokens instead of a yes/no.

## Concepts

### `z = x1 * w1 + b` - the net input

This is `y = mx + b` from school, with the pieces renamed:

- `m` (slope) → `w1`, the **weight**. It controls how much the input matters. A big weight means a small change in `x1` makes a big change in `z`.
- `b` (intercept) → `b`, the **bias**. It shifts the result up or down regardless of the input.
- `x` → `x1`, the input **feature**: one measurable fact about the example (e.g. "how many times the word 'free' appears in this email").

ML says "weight" instead of "slope" because once there are many inputs (`w1*x1 + w2*x2 + w3*x3 + ...`), there's no single slope anymore. Each weight says how heavily its own input counts toward the result.

With the values in the code:

```python
z = 1.1 * 2.2 + 0.0   # = 2.42
```

`z` is a raw score. It can be any number from negative to positive infinity, so it isn't a prediction yet.

All four tensors have shape `(1,)`: a 1D tensor holding one number. `z` comes out as `tensor([2.4200])`, also shape `(1,)`.

**Try it:** Before running anything, change `x1` to `-1.0` and predict `z`. Then predict whether the sigmoid of that `z` will land above or below 0.5.

### `a = torch.sigmoid(z)` - the prediction

#### The problem sigmoid solves

We want the model to answer a yes/no question ("is this email spam?") with a probability: a number between 0 and 1. But `z = w1 * x1 + b` can be anything. Here it's 2.42, but with other inputs it could be -37 or 500. "500% likely to be spam" means nothing.

The obvious fix is a hard cutoff:

```python
def step(z: float) -> float:
    return 1.0 if z > 0 else 0.0
```

That does give an answer between 0 and 1, but it has two problems:

- **It throws away confidence.** `z = 0.01` and `z = 500` both become 1.0. The model can't say "barely yes" vs "definitely yes," and the loss can't tell a near-miss from a confident mistake.
- **It can't be learned from.** Training works by nudging `w1` a little and asking "did the output get better or worse?" With a step function, a small nudge almost never changes the output: it stays 1.0 until `z` crosses 0, then jumps. A flat output gives autograd no signal about which way to move. This is the bigger problem, and it's why sigmoid exists.

Sigmoid is the smooth version of that cutoff. It still goes from 0 to 1 with the switch around `z = 0`, but it curves gradually instead of jumping. Every small change in `z` makes a small change in the output, so there's always a direction to nudge.

#### How the formula works

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Decoding the symbols:

- `e` - Euler's number, about 2.718. It's the standard base for growth and decay in maths.
- `e^(-z)` - e raised to the power of negative z. This is the only moving part. When `z` is big, `-z` is very negative and `e^(-z)` is tiny. When `z` is very negative, `-z` is big and `e^(-z)` is huge.
- `1 / (1 + ...)` - one divided by "one plus something positive." Since that something is never negative, the bottom is always at least 1, so the result is always at most 1, and it's always above 0.

Now trace three inputs through it:

| `z` | `e^(-z)` | `1 / (1 + e^(-z))` | Meaning |
|---|---|---|---|
| 5 | 0.0067 (tiny) | 1 / 1.0067 ≈ **0.993** | big positive → close to 1 |
| 0 | 1 | 1 / 2 = **0.5** | zero → exactly on the fence |
| -5 | 148.4 (huge) | 1 / 149.4 ≈ **0.0067** | big negative → close to 0 |

And the values in between, so you can see the S shape:

| `z` | -5 | -2 | -1 | 0 | 1 | 2 | 2.42 | 5 |
|---|---|---|---|---|---|---|---|---|
| `sigmoid(z)` | 0.007 | 0.119 | 0.269 | 0.5 | 0.731 | 0.881 | 0.918 | 0.993 |

Read left to right: flat near 0, a steep climb through the middle (between about -2 and 2), then flat near 1. It gets very close to 0 and 1 but never actually reaches either.

It's also symmetric: `sigmoid(-z) = 1 - sigmoid(z)`. `sigmoid(1)` is 0.731 and `sigmoid(-1)` is 0.269, which add up to 1.

**See it move:** open [sigmoid_explorer.html](sigmoid_explorer.html) in a browser. Drag `x1`, `w1`, and `b` to move `z` along the curve, tick "Show hard cutoff" to compare against the step function, and watch the slope line flatten as `z` goes far from 0.

All three sliders do the same thing on the plot: they slide the dot along one fixed curve. That's correct, because sigmoid only ever sees `z`. It has no idea whether `z = 2.42` came from `1.1 * 2.2 + 0` or `0 * 5 + 2.42`. The curve's x-axis is `z`, so the curve never changes shape; only the dot's position does.

Where the sliders differ is *how much* they move `z`:

- `b` - moves `z` by exactly as much as you move it. +1 on `b` is +1 on `z`.
- `w1` - moves `z` by `x1` times as much. With `x1 = 1.1`, +1 on `w1` is +1.1 on `z`. Set `x1` to 0 and `w1` does nothing at all.
- `x1` - moves `z` by `w1` times as much, and in reverse if `w1` is negative.

That second point matters for training. The weight's effect on the output depends on the input it multiplies, so the gradient for `w1` will involve `x1`. You'll see exactly that when autograd computes it in A.4.

#### Reading the output

`sigmoid(2.42) ≈ 0.918`, so the model is saying "I'm 91.8% sure this example is class 1." The sign of `z` tells you which side of 0.5 you land on, and the size of `z` tells you how confident the model is.

LLMs don't use sigmoid for their output, because they choose between 50,257 tokens instead of yes/no. They use **softmax**, its many-choice cousin, which you'll meet in Chapter 3. Same idea: turn raw scores into probabilities.

**Plain Python:**
```python
import math

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

sigmoid(2.42)  # 0.9183
```

**PyTorch:**
```python
torch.sigmoid(torch.tensor([2.42]))  # tensor([0.9183]), shape (1,)
```

**Template:**
```python
probability = torch.sigmoid(raw_score)
```

**Try it:** Type the plain-Python `sigmoid` yourself, then call `torch.sigmoid` on `torch.tensor([-3.0, 0.0, 3.0])` (shape `(3,)`). Before running, predict the middle value exactly, and use the symmetry rule to predict how the first and last values relate.

### `loss = F.binary_cross_entropy(a, y)` - how wrong was it?

The **loss** is a single number that answers one question: how wrong was that prediction?

- `y = 1.0` is the true answer (the label).
- `a = 0.918` is the model's guess.
- The loss compares the two. 0 means a perfect prediction, and bigger means worse.

**Binary cross-entropy** is one specific way to measure "how wrong," used when the answer is yes/no (binary). When the true label is 1, it reduces to:

```python
loss = -math.log(a)   # -log(0.918) ≈ 0.085
```

Here's how that behaves when the truth is 1:

| Model's guess `a` | Loss |
|---|---|
| 0.99 | 0.01 |
| 0.918 (this code) | 0.085 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.01 | 4.61 |

A guess that's close gives a tiny loss. A guess that's confident and wrong gives a huge loss.

The `log` is what does that. As `a` heads toward 0, `log(a)` heads toward negative infinity, so `-log(a)` blows up. Being 99% sure of the wrong answer costs far more than being unsure.

#### The full formula

The label can also be 0, so the full formula has a term for each case:

**Plain Python:**
```python
import math

def binary_cross_entropy(a: float, y: float) -> float:
    return -(y * math.log(a) + (1 - y) * math.log(1 - a))

binary_cross_entropy(0.918, 1.0)  # 0.085
binary_cross_entropy(0.918, 0.0)  # 2.505
```

**PyTorch:**
```python
import torch.nn.functional as F

F.binary_cross_entropy(torch.tensor([0.918]), torch.tensor([1.0]))  # tensor(0.0852)
F.binary_cross_entropy(torch.tensor([0.918]), torch.tensor([0.0]))  # tensor(2.5052)
```

**Template:**
```python
loss = F.binary_cross_entropy(predicted_probabilities, true_labels)
```

The `y` and `(1 - y)` act as switches:

- When `y = 1`, the second term is multiplied by 0 and disappears, leaving `-log(a)`. That punishes guesses close to 0.
- When `y = 0`, the first term disappears, leaving `-log(1 - a)`. That punishes guesses close to 1.

#### What "cross-entropy" actually is

> **Optional - skip on first read.** For now, you only need four facts: the loss is one number saying how wrong the prediction was, lower is better, confident wrong answers are punished hardest, and training changes the weights to push it down. This section explains where the formula comes from. Come back to it in Chapter 5, where cross-entropy returns as the LLM's training loss alongside perplexity.

The formula above can look like it was invented specially for this. It wasn't. It's one case of a general idea from information theory, built in three steps.

**Step 1: surprise.** If you said something had probability `p` and then it happened, your surprise is `-log(p)`:

| You gave it probability `p` | Surprise `-log(p)` |
|---|---|
| 1.0 (certain) | 0 |
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |

This matches everyday surprise. Something you were sure of happening isn't surprising at all, and something you gave 1% is very surprising. The `-` sign is there because the log of a number between 0 and 1 is negative, and flipping it makes surprise a positive number.

**Step 2: entropy = average surprise when you know the true odds.** Take a coin that lands heads 90% of the time. If you predict with the true odds (90/10), your average surprise over many flips is:

```python
0.9 * -math.log(0.9) + 0.1 * -math.log(0.1)   # = 0.325
```

That's the coin's **entropy**: how unpredictable it is, even when you know everything about it. A fair 50/50 coin has entropy 0.693, higher because it's harder to predict. Entropy is the lowest average surprise anyone can achieve on that coin.

**Step 3: cross-entropy = average surprise when you predict with the *wrong* odds.** The coin still lands heads 90% of the time, but now you believe it's 50/50:

```python
0.9 * -math.log(0.5) + 0.1 * -math.log(0.5)   # = 0.693
```

Your average surprise went up from 0.325 to 0.693, because your beliefs don't match reality. That's **cross-entropy**. It "crosses" two distributions: the **true** probabilities decide how often each outcome happens (the weights outside the log), and **your predicted** probabilities decide how surprised you are by each one (inside the log).

Cross-entropy is never lower than entropy, and the two are equal only when your predictions exactly match the truth. So **making cross-entropy smaller means making your predictions closer to the truth**. That's exactly what we want a loss function to do.

**Plain Python:**
```python
import math

def cross_entropy(true_probs: list[float], predicted_probs: list[float]) -> float:
    return -sum(p * math.log(q) for p, q in zip(true_probs, predicted_probs))

cross_entropy([0.9, 0.1], [0.9, 0.1])  # 0.325 - predictions match the truth (this is the entropy)
cross_entropy([0.9, 0.1], [0.5, 0.5])  # 0.693 - predictions are off, so it's higher
```

**Template:**
```python
loss = -sum(true_p * math.log(predicted_p) for true_p, predicted_p in zip(true_probs, predicted_probs))
```

#### How that becomes binary cross-entropy

In training, each example's truth isn't a 90/10 coin; we *know* the answer. If the label is `y = 1`, the true distribution over (class 1, class 0) is `[1.0, 0.0]`: class 1 happened with certainty.

The model's predicted distribution is `[a, 1 - a]`: `a` for class 1, and whatever's left for class 0.

Plug both into the general formula:

```python
cross_entropy([1.0, 0.0], [0.9183, 0.0817])
# = -(1.0 * log(0.9183) + 0.0 * log(0.0817))
# = -log(0.9183)
# = 0.0852   <- the same number F.binary_cross_entropy gave
```

So `-(y * log(a) + (1 - y) * log(1 - a))` is just the general cross-entropy with:

- true distribution `[y, 1 - y]`
- predicted distribution `[a, 1 - a]`

The `y` and `(1 - y)` weights that act as "switches" are the true probabilities. "**Binary**" only means there are two outcomes.

In plain words, the loss is how surprised the model was by the right answer. Training makes the model less and less surprised by the truth.

This idea comes back in Chapter 5, scaled up. An LLM predicts a distribution over 50,257 tokens. The truth is "the actual next token, with certainty," so the loss is `-log(probability the model gave the correct next token)`. It's the same formula with more outcomes.

**Try it:** Type the plain-Python `cross_entropy`. Before running each line, predict whether the result is bigger or smaller than 0.325: `cross_entropy([0.9, 0.1], [0.8, 0.2])` and `cross_entropy([0.9, 0.1], [0.1, 0.9])`. Then call it with `[1.0, 0.0]` and `[0.9183, 0.0817]`, and check you get the same number as `F.binary_cross_entropy` on your file's values.

Notice the output shape. `a` and `y` are shape `(1,)`, but the loss prints as `tensor(0.0852)` with no brackets: a 0D scalar. Loss functions average over every example and return one number, because training needs a single thing to minimize.

`F` is the conventional alias for `torch.nn.functional`, a module of plain functions (sigmoid, losses, activations) that take tensors in and give tensors out, with no weights stored inside.

**Try it:** Change `y` to `0.0` and predict whether the loss goes up or down before running it. Then set `w1` to `-2.2` (keeping `y = 0.0`) and predict again. Use the table above to sanity-check your guesses.

### Why the loss matters

Training means changing `w1` and `b` so the loss gets smaller. The open question is which direction to change each one, and by how much.

That's what **autograd** answers. It computes how the loss changes when you nudge each weight (the **gradient**). The next step in the book (A.4, Figure A.8 on page 263) marks `w1` and `b` as trainable and calls `.backward()` on exactly this loss.

#### Where the Name Comes From

**Cross-entropy** comes from information theory. Claude Shannon borrowed *entropy* from physics in 1948, where it measured disorder (from the Greek *entropia*, "a turning toward"). In information theory, entropy is the average "surprise" of outcomes, and `-log(p)` is the surprise of seeing something you gave probability `p`. Something you predicted with 99% confidence is barely surprising; something you gave 1% is very surprising.

*Cross*-entropy measures the surprise when the truth comes from one distribution (the real labels) and your predictions come from another (the model). Training minimizes how surprised the model is by the truth. **Binary** just means two classes.

**Sigmoid** means "S-shaped," from the Greek letter sigma (σ) in its curved form, plus *-oid*, "resembling." Plot it and you get a stretched-out S.

## Your Task

- [ ] Type the forward pass: label, feature, weight, bias, net input, sigmoid, loss
- [ ] Predict `z` and the side of 0.5 the sigmoid lands on for `x1 = -1.0`
- [ ] Type a plain-Python sigmoid and check it against `torch.sigmoid`
- [ ] Predict how the loss changes for `y = 0.0`, then for `y = 0.0` with `w1 = -2.2`
- [ ] Next: A.4, computing gradients with autograd
