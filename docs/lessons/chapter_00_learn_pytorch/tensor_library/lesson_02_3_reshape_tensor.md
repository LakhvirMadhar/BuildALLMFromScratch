# Lesson 02_3: Reshaping, Transposing, and Multiplying Tensors

## What We're Doing

Four everyday tensor operations: reading a tensor's shape, rearranging it into a new shape (`.reshape` / `.view`), flipping it across its diagonal (`.T`), and multiplying two matrices (`.matmul` / `@`).

**Book sections:** Appendix A.2.3 (pages 260–261).
**Code:** `src/build_llm_from_scratch/chapter_00_learn_pytorch/tensor_library/02_3_reshape_tensor.py`

## Why / Context

Lessons 02_1 and 02_2 built tensors and set their dtypes. This lesson is about *moving the numbers around* and *combining* tensors.

These four operations show up on almost every page of Chapter 3. Attention scores are `queries @ keys.T`. Multi-head attention splits one big tensor into several heads with `.view(...)` and swaps dimensions with `.transpose(...)`. If these four feel automatic, Chapter 3's code reads like prose. If they don't, it reads like noise.

## Concepts

### `.shape` - how big is each dimension?

Every tensor knows how many entries it has along each dimension. `.shape` reports it.

```python
tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
tensor2d.shape   # torch.Size([2, 3])
```

`torch.Size([2, 3])` means 2 rows, 3 columns. `torch.Size` is a subclass of Python's `tuple`, so you can index it (`tensor2d.shape[0]` is `2`) and unpack it (`rows, cols = tensor2d.shape`).

**Plain Python:**
```python
nested = [[1, 2, 3],
          [4, 5, 6]]
(len(nested), len(nested[0]))   # (2, 3)
```

**PyTorch:**
```python
tensor2d.shape   # torch.Size([2, 3])
```

**Template:**
```python
num_rows, num_cols = some_matrix.shape
```

The plain-Python version only works because every row happens to be the same length. A tensor *guarantees* that (it's a rectangle by construction), which is why `.shape` can be one tuple instead of a length per row.

**Try it:** Type a tensor with 4 rows and 2 columns. Predict `.shape`, then unpack it into `num_rows, num_cols` and print both.

### `.reshape` - same numbers, new shape

Reshaping keeps every number and their order, and changes how they're grouped into rows. Think of reading all six numbers left-to-right, top-to-bottom into one long line, then cutting that line into rows of a new length.

```python
tensor2d.reshape(3, 2)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
```

Shape goes `(2, 3)` → `(3, 2)`. The line `1 2 3 4 5 6` is the same; it's just been cut every 2 numbers instead of every 3.

**Plain Python:**
```python
nested = [[1, 2, 3],
          [4, 5, 6]]
flat = [value for row in nested for value in row]   # [1, 2, 3, 4, 5, 6]
new_cols = 2
[flat[i:i + new_cols] for i in range(0, len(flat), new_cols)]
# [[1, 2], [3, 4], [5, 6]]
```

**PyTorch:**
```python
tensor2d.reshape(3, 2)
```

**Template:**
```python
reshaped = some_tensor.reshape(new_dim_0, new_dim_1)
```

The one rule: the total number of entries can't change. `(2, 3)` has 6, so `(3, 2)`, `(6, 1)`, `(1, 6)`, and `(6,)` all work. `(4, 2)` needs 8 and raises an error.

**Try it:** Reshape `tensor2d` to `(6, 1)` and to `(1, 6)`. Before running each, write out what you expect it to print. Then try `(4, 2)` and read the error message.

### `.view` - reshape without copying

Page 260 says `.view` is "the more common command" and gives the same result:

```python
tensor2d.view(3, 2)   # same output as reshape(3, 2)
```

Page 261 explains the difference. A tensor's numbers live in one flat block of memory, and the shape is a label telling PyTorch how to read that block as rows and columns.

- **`.view`** writes a new label for the *same* memory block. No copy, so it's fast, but it only works when the numbers are laid out in memory in the order the new shape needs. That layout is called **contiguous**. If it isn't, `.view` raises an error.
- **`.reshape`** tries to do a `.view`. If the memory isn't contiguous, it quietly copies the numbers into a new block in the right order and reshapes that.

When does memory stop being contiguous? The main case is right below: `.T` (transpose) doesn't move any numbers, it just changes the label to read the block in a different order. So `tensor2d.T.view(6)` fails, while `tensor2d.T.reshape(6)` works.

For this book, both work in almost every place you'll use them. The book (and Chapter 3) uses `.view`; when it needs one after a transpose, it calls `.contiguous()` first, which makes the copy explicitly.

#### Where the Name Comes From

**Contiguous** is from the Latin *contiguus*, "touching," from *contingere*, "to touch." Contiguous memory is memory where each next number sits directly touching the last one, in the order you'll read them.

**Try it:** Type `tensor2d.T.view(6)` and read the error. Then fix it two ways: with `.reshape(6)`, and with `.contiguous().view(6)`. Predict the printed order of the six numbers before running. (It's `1, 4, 2, 5, 3, 6`, not `1, 2, 3, 4, 5, 6`. Work out why from the transpose section below.)

### `.T` - transpose: flip across the diagonal

Transposing turns rows into columns. Row 0 becomes column 0, row 1 becomes column 1.

```python
tensor2d.T
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
```

Shape goes `(2, 3)` → `(3, 2)`, the same shape `.reshape(3, 2)` produced, but different contents. That's page 261's point: reshape re-cuts the same line of numbers, transpose rotates the grid. Compare:

| `reshape(3, 2)` | `.T` |
|---|---|
| `[[1, 2], [3, 4], [5, 6]]` | `[[1, 4], [2, 5], [3, 6]]` |

The "diagonal" is the line of cells where row index equals column index (`[0, 0]`, `[1, 1]`, ...). Transposing mirrors the grid across that line: the value at `[i, j]` moves to `[j, i]`.

**Plain Python:**
```python
nested = [[1, 2, 3],
          [4, 5, 6]]
[list(column) for column in zip(*nested)]
# [[1, 4], [2, 5], [3, 6]]
```

`zip(*nested)` unpacks the rows as separate arguments to `zip`, which pairs up their first items, their second items, and so on. That's exactly reading the grid column by column.

**PyTorch:**
```python
tensor2d.T
```

**Template:**
```python
flipped = some_matrix.T   # (rows, cols) -> (cols, rows)
```

#### Where the Name Comes From

**Transpose** is from the Latin *trans* ("across") + *ponere* ("to place"): to place across. It's the same word musicians use for moving a song to a different key. In maths it means placing each value across the diagonal from where it started.

**Try it:** Take your `(4, 2)` tensor from the first Try-it. Predict its `.T` (shape and contents) on paper, then run it. Then check that `.T.T` gives back the original.

### `.matmul` and `@` - matrix multiplication

Matrix multiplication fills each cell `[i, j]` of the result with the **dot product** of row `i` of the left matrix and column `j` of the right matrix. A dot product multiplies matching positions and adds them up.

The book multiplies `tensor2d` by its own transpose:

```python
tensor2d.matmul(tensor2d.T)
# tensor([[14, 32],
#         [32, 77]])
```

Working out each cell by hand. `tensor2d`'s rows are `[1, 2, 3]` and `[4, 5, 6]`. The columns of `tensor2d.T` are those same two rows, standing up:

- `[0, 0]` - row 0 · column 0 = `1·1 + 2·2 + 3·3 = 14`
- `[0, 1]` - row 0 · column 1 = `1·4 + 2·5 + 3·6 = 32`
- `[1, 0]` - row 1 · column 0 = `4·1 + 5·2 + 6·3 = 32`
- `[1, 1]` - row 1 · column 1 = `4·4 + 5·5 + 6·6 = 77`

Shapes: `(2, 3) @ (3, 2)` → `(2, 2)`. The rule is that the **inner** numbers must match (the `3`s: the length of each row on the left equals the length of each column on the right, so the dot products line up), and the **outer** numbers become the result's shape.

```
(2, 3) @ (3, 2)
    ^     ^         inner: must match
 ^           ^      outer: result shape (2, 2)
```

**Plain Python:**
```python
left = [[1, 2, 3],
        [4, 5, 6]]
right = [[1, 4],
         [2, 5],
         [3, 6]]
right_columns = list(zip(*right))

[[sum(a * b for a, b in zip(row, column)) for column in right_columns]
 for row in left]
# [[14, 32], [32, 77]]
```

**PyTorch:**
```python
tensor2d.matmul(tensor2d.T)
tensor2d @ tensor2d.T        # identical
```

`@` is Python's own matrix-multiplication operator, added in Python 3.5 (PEP 465) for exactly this. Python itself doesn't implement it for lists; it calls the left object's `__matmul__` method, and PyTorch tensors define that to mean `.matmul`. So `@` and `.matmul` are the same call.

**Template:**
```python
result = left_matrix @ right_matrix   # (m, n) @ (n, p) -> (m, p)
```

This exact line, `x @ x.T`, is the first piece of code in Chapter 3. There it scores every token against every other token: cell `[i, j]` says how similar token `i` is to token `j`. Notice that here, cell `[0, 1]` and cell `[1, 0]` are both 32. A matrix times its own transpose always comes out symmetric like that, because "row 0 · row 1" and "row 1 · row 0" are the same dot product.

**Try it:** Without running anything, predict the shape *and* all the values of `tensor2d.T @ tensor2d` (transpose on the left this time). Then run it. Then try `tensor2d @ tensor2d` and explain the error using the inner-numbers rule.

### Notes on your `02_3_reshape_tensor.py`

- **`tensor2d.view()` with no arguments crashes.** `.view` needs the new shape, the same as `.reshape`: `tensor2d.view(3, 2)`. Everything after that line never ran.
- **The `RASKA_NOTE` string.** Explanations like this belong in this doc rather than in a string printed at runtime, and it's now covered in the `.view` section above. If you keep a note in the code, a short comment works better than a variable, and the spelling is *Raschka*.

## Your Task

- [ ] Fix the `.view()` call and run the whole file
- [ ] Predict and check `.shape` for a `(4, 2)` tensor
- [ ] Reshape to `(6, 1)`, `(1, 6)`, and `(4, 2)`, and read the error on the last one
- [ ] Break `.view` on a transposed tensor, then fix it with `.reshape` and with `.contiguous().view`
- [ ] Transpose your `(4, 2)` tensor by hand, then check it
- [ ] Work out `tensor2d.T @ tensor2d` by hand, then check it, and explain why `tensor2d @ tensor2d` fails
