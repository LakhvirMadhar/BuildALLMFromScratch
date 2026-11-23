# Lesson 02_1: Tensors

## What We're Doing

Creating tensors with different numbers of dimensions, and learning to tell them apart by their brackets and their shape.

**Book sections:** Appendix A.2.1 (pages 258–259).
**Code:** `src/build_llm_from_scratch/chapter_00_learn_pytorch/02_1_tensors.py`

## Why / Context

Every piece of data in the book is a tensor. Text becomes a tensor of token IDs in Chapter 2, then a tensor of shape `(batch, tokens, embedding_size)` that flows through every layer of the GPT model. Reading a tensor's shape and knowing what each dimension means is the single most useful skill for not getting lost later.

## Concepts


### Dimensions: counting the brackets

A tensor is a grid of numbers with any number of dimensions. The number of dimensions is how many levels of brackets are nested. Count the `[` at the very start:

**Plain Python:**
```python
scalar = 5                                    # no brackets  -> 0 levels
vector = [1, 2, 3]                            # [            -> 1 level
matrix = [[1, 2], [3, 4]]                     # [[           -> 2 levels
stack = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # [[[          -> 3 levels
```

**PyTorch:**
```python
torch.tensor(5)                                    # 0D, shape ()
torch.tensor([1, 2, 3])                            # 1D, shape (3,)
torch.tensor([[1, 2], [3, 4]])                     # 2D, shape (2, 2)
torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # 3D, shape (2, 2, 2)
```

**Template:**
```python
tensor = torch.tensor(nested_list)
tensor.shape  # one number per level of nesting
```

### 2D vs 3D

- **2D** - a table: rows and columns, like one spreadsheet. Shape `(rows, columns)`.
- **3D** - a stack of tables, like several spreadsheet tabs. Shape `(tables, rows, columns)`.

Reading `(2, 2, 2)` from left to right: 2 tables, each with 2 rows, each row holding 2 numbers. The leftmost number is the outermost level of brackets; the rightmost is the innermost.

In plain Python, a 2D tensor is a list of lists, and a 3D tensor is a list of lists of lists. In the book, the 3D case shows up constantly: a batch of sentences, where each sentence is a table of tokens and each token is a row of numbers.

#### Saying it out loud

| Dims | Name | `.shape` prints | Said aloud |
|---|---|---|---|
| 0 | scalar | `torch.Size([])` | "a scalar" (the empty shape means no dimensions at all) |
| 1 | vector | `torch.Size([3])` | "a vector of length 3" |
| 2 | matrix | `torch.Size([2, 2])` | "a 2 by 2 matrix" |
| 3 | 3D tensor | `torch.Size([2, 2, 2])` | "a 2 by 2 by 2 tensor," or "a 3D tensor" |

There's no special everyday name past matrix. A 3D tensor is not a "3D vector" — a 3D vector means a 1D tensor with 3 numbers, like `[1, 2, 3]` (a point in 3D space), which is a different thing. From three dimensions up, people just say "3D tensor," "4D tensor," and so on. You'll also hear **rank**: "rank-3 tensor" means the same as "3D tensor" (rank = number of dimensions).

In the book, people usually describe a 3D tensor by what its dimensions mean rather than its name: "a batch of 8 sequences of 4 tokens, each a 768-dimensional embedding" for shape `(8, 4, 768)`. Naming each dimension is the habit worth building.

#### The easy mistake: same numbers, different brackets

These two hold the same eight numbers but have different dimensions:

```python
torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])      # 2D: one table, 4 rows -> shape (4, 2)
torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D: two tables of 2 rows -> shape (2, 2, 2)
```

The only difference is the extra bracket pair grouping the rows into tables. Laying rows out across several lines doesn't change anything — only the brackets count. When in doubt, print `.shape`: it's the fastest way to check how many dimensions a tensor really has.

**Try it:** Check your `tensor3d` in `02_1_tensors.py` by printing its `.shape`. If it shows two numbers, it's 2D — add the brackets that group its rows into two tables, and confirm the shape becomes three numbers. Then predict the shape of a tensor holding 3 tables of 2 rows with 4 numbers each, build it, and check your prediction. Compare against `tensor3d` in Listing A.1 on page 259: count its opening brackets.

## Your Task

- [ ] Create 0D, 1D, 2D, and 3D tensors and print each one's `.shape`
- [ ] Fix `tensor3d` so it's genuinely 3D
- [ ] Predict, then build and check, a `(3, 2, 4)` tensor
- [ ] Next: `lesson_02_2_tensor_data_types.md` (A.2.2)
