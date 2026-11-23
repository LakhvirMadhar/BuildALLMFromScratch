# Lesson 02_2: Tensor Data Types

## What We're Doing

Learning what kind of number a tensor holds (its `dtype`), what 32-bit and 64-bit mean, and how to convert between types.

**Book sections:** Appendix A.2.2 (pages 259–260).
**Code:** `src/build_llm_from_scratch/chapter_00_learn_pytorch/02_2_tensor_data_types.py`

## Why / Context

A tensor's `dtype` decides how much memory it takes and how fast the GPU can work with it. Every model weight in the book is a `float32`, every token ID is an `int64`, and mixing them up is a common source of errors. Later, in post-training work, choosing smaller types like `bfloat16` is how large models fit on a GPU at all.

## Concepts

### Tensor data types: `dtype` (A.2.2, page 259)

Every tensor has a **data type**, or `dtype`: what kind of number every element is, and how much memory each one takes. Unlike a Python list, a tensor can't mix types — all its elements share one `dtype`. That's part of what makes tensors fast: the computer knows exactly how big each number is and can pack them side by side in one block of memory.

You read it with the `.dtype` attribute:

**Plain Python:**
```python
type(1)    # <class 'int'>
type(1.0)  # <class 'float'>
```

**PyTorch:**
```python
torch.tensor([1, 2, 3]).dtype        # torch.int64
torch.tensor([1.0, 2.0, 3.0]).dtype  # torch.float32
```

**Template:**
```python
tensor.dtype  # the type every element in the tensor shares
```

PyTorch picks the `dtype` from what you give it: whole numbers become `int64`, numbers with a decimal point become `float32`.

#### What the "bit" part means

A **bit** is the smallest unit a computer stores: a single 0 or 1. Numbers are stored as a fixed-length row of bits, and the length is the number in the name:

- **32-bit** - each number is stored in 32 zeros-and-ones, which is **4 bytes** (a byte is 8 bits)
- **64-bit** - each number is stored in 64 zeros-and-ones, which is **8 bytes**

More bits means more possible patterns, so more numbers can be represented. Each extra bit doubles the count: 32 bits give 2³² (about 4.3 billion) patterns, 64 bits give 2⁶⁴ (about 18 quintillion). What those patterns *mean* depends on whether the type is an integer or a float.

**For integers (`int32`, `int64`), more bits means a bigger range.** Every pattern is one whole number, exactly:

- `int32` - from about −2.1 billion to +2.1 billion
- `int64` - from about −9.2 quintillion to +9.2 quintillion

Within that range, integers are always exact. The bits only limit how *big* a number can get.

**For floats (`float32`, `float64`), more bits means more precision.** Floats can't store most decimals exactly — they store the closest value they can, and more bits means a closer match:

- `float32` - about **7 significant digits** of accuracy
- `float64` - about **15–16 significant digits** of accuracy

You can see the difference by storing `0.1`, which no float can hold exactly:

```python
torch.tensor(0.1, dtype=torch.float32).item()  # 0.10000000149011612
torch.tensor(0.1, dtype=torch.float64).item()  # 0.1
```

The `float32` version is off from the eighth digit onward; the `float64` version is off too, but so far down (around the 17th digit) that Python prints it as `0.1`. This is the same reason `0.1 + 0.2` prints `0.30000000000000004` in plain Python — Python's own `float` is a 64-bit float.

**In plain Python terms:** Python's `int` has no bit limit at all — it grows as big as memory allows, so you've never had to think about this. Python's `float` is always 64-bit. Tensors make you choose because they store millions of numbers, and the choice changes how much memory they take.

#### Why PyTorch defaults to `float32`

Deep learning doesn't need 16 digits of precision. A model's weights are nudged by tiny amounts millions of times during training; whether a weight is `0.1234567` or `0.123456789012345` makes no practical difference to what the model learns. So the extra precision of `float64` buys nothing, and it costs two things:

- **Memory** - `float64` takes twice the space. GPT-2's smallest model has about 124 million weights: about 500 MB in `float32`, about 1 GB in `float64`. That doubling applies to everything on your GPU's 12 GB.
- **Speed** - GPUs are built to be fastest at 32-bit (and smaller) maths. Consumer cards like your RTX 4070 run `float64` dramatically slower than `float32`.

Integers stay `int64` because they're used for exact things — counting, indexing, token IDs — where there's no precision to trade away, and the larger range never hurts.

Later in the book and in post-training work you'll see even smaller floats, like `bfloat16` (16 bits), used for the same reason taken further: half the memory again, and still accurate enough for training.

### Changing a tensor's `dtype` with `.to()`

`.to()` returns a copy of the tensor converted to another `dtype`. The original is unchanged.

**Plain Python:**
```python
values = [1, 2, 3]
as_floats = [float(value) for value in values]  # [1.0, 2.0, 3.0]
```

**PyTorch:**
```python
tensor1d = torch.tensor([1, 2, 3])        # int64
floatvec = tensor1d.to(torch.float32)     # float32: tensor([1., 2., 3.])
```

- `tensor1d` - the tensor to convert; it stays `int64`
- `.to(...)` - "give me a version of this tensor converted to..." The same method is also used to move tensors to the GPU (`.to("cuda")`), which comes in A.9
- `torch.float32` - the target `dtype`

**Template:**
```python
converted = tensor.to(target_dtype)
```

Converting matters in practice because PyTorch won't always mix `dtype`s for you: many operations (like multiplying by a model's `float32` weights) need the input to be a float already.

**Try it:** Create an integer tensor and a float tensor and print each `dtype`. Convert the integer one to `float32` with `.to()` and print both `dtype`s again to confirm the original didn't change. Then print `.element_size()` on a `float32` tensor and a `float64` tensor — it returns how many bytes each element uses. Predict the two numbers before you run it.

#### Where the Name Comes From

`Bit` is a contraction of "**bi**nary dig**it**," coined by the statistician John Tukey at Bell Labs around 1947 and popularized by Claude Shannon in 1948. *Binary* comes from the Latin *bini*, "two at a time" — a binary digit can only be one of two values, 0 or 1.

`Floating-point` describes how the number is stored: like scientific notation, a float keeps a fixed number of significant digits plus a separate exponent that says where the decimal point goes. The point "floats" — it can sit anywhere relative to the digits — so the same 32 bits can hold both `0.000001234` and `1234000000.0`. The alternative, *fixed-point*, pins the decimal point in one place, which limits how big or small the number can be.

`dtype` is just short for "**d**ata **type**." PyTorch borrowed the name from NumPy.

## Your Task

- [ ] Print the `dtype` of an integer tensor and a float tensor
- [ ] Convert an integer tensor to `float32` with `.to()` and confirm the original is unchanged
- [ ] Predict, then check with `.element_size()`, the bytes per element for `float32` and `float64`
