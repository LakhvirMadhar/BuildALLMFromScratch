# Lesson 01: Check PyTorch

## What We're Doing

A first look at PyTorch: what it is, how it's organized, and where it sits in the AI / machine learning / deep learning picture. Then installing it and checking it can see the GPU, so tensors in A.2 can be typed and run.

**Book sections:** Appendix A.1 (pages 251–257).
**Code:** `src/build_llm_from_scratch/chapter_00_learn_pytorch/01_check_pytorch.py`

## Why / Context

Every chapter of the book builds its LLM out of PyTorch. The tokenizer's output becomes a PyTorch tensor in Chapter 2, attention in Chapter 3 is PyTorch matrix maths, the GPT model in Chapter 4 is a PyTorch `nn.Module`, and training in Chapters 5–7 is a PyTorch training loop.

Page 251 says you can skip Appendix A if you already know deep learning. You haven't used PyTorch before, so this appendix is the foundation the rest of the book assumes.

## How to Go Through Appendix A

Appendix A runs from page 251 to 288, but not every page matters equally for this book.

| Section | Pages | What to do | Why |
|---|---|---|---|
| A.1 What is PyTorch | 251–254 | Read (short) | The overview: what PyTorch is made of |
| A.1.3 Installing PyTorch | 254–257 | Skim | The book uses `pip`; this project uses Poetry. See "Installing PyTorch" below |
| A.2 Understanding tensors | 258–261 | Read closely, type the code | Tensors are in every line of the book |
| A.3 Computation graphs | 261–263 | Read closely | How PyTorch keeps track of the steps it computes |
| A.4 Automatic differentiation | 263–265 | Read closely | How a model learns, through `.backward()` |
| A.5 Multilayer neural networks | 265–270 | Read closely, type the code | `nn.Module` is how every model in the book is built |
| A.6 Data loaders | 270–274 | Read closely, type the code | Chapter 2 builds one for text |
| A.7 A typical training loop | 274–278 | Read closely, type the code | Chapters 5–7 are this loop, scaled up |
| A.8 Saving and loading | 278–279 | Read (short) | Used in Chapter 5 |
| A.9.1–A.9.2 GPUs | 279–282 | Read | `.to(device)` shows up all through the book |
| A.9.3 Multiple GPUs | 282–288 | Skip | Training across several GPUs at once. The book is designed to run on one machine |

A.2 to A.7 are the core: tensors, autograd, `nn.Module`, data loaders, and the training loop. Chapters 2–7 assume all of them.

Each topic gets its own numbered lesson in `chapter_00_learn_pytorch/`, mirroring the numbered files in `src/`. This lesson is A.1; `lesson_02_tensors.md` is A.2.

#### Where the Name Comes From

`Tensor` comes from the Latin *tendere*, "to stretch." The physicist Woldemar Voigt used it in 1898 to describe the stresses stretching a crystal. Mathematicians later generalized it to mean a grid of numbers with any number of dimensions. In PyTorch the physics meaning is gone; "tensor" just means "an N-dimensional array."

### Installing PyTorch (A.1.3, pages 254–257)

The book installs PyTorch with `pip install torch`. This project uses Poetry, and there's a Windows-specific catch: the normal package index (PyPI) only has the **CPU-only** build of PyTorch for Windows. The GPU builds live on PyTorch's own server.

You have an RTX 4070, so install the GPU build. It makes the training chapters (5–7) much faster.

**Try it:** Turn the step-3 check into a reusable script at `01_check_pytorch.py`, so you can run it any time instead of retyping the one-liner. Things to get right:

- It should print the PyTorch version and whether CUDA is available.
- `torch.cuda.get_device_name(0)` raises an error when there's no GPU, so only ask for the device name once you know CUDA is available, and print a clear message otherwise.
- It has to pass the project's checks: `poetry run ruff check src` and `poetry run mypy src`. Strict mypy wants a return type on every function, including `main`.
- Use the `if __name__ == "__main__":` guard, so importing the file doesn't run it.

#### Where page 255 doesn't apply to this project

- **"`pip install torch` will automatically install the GPU version"** - true on Linux, not on Windows. On Windows, the default package is the CPU-only build, which is why step 1 below adds PyTorch's own GPU package source.
- **`pip` itself** - this project uses Poetry, which also records the dependency in `pyproject.toml` and `poetry.lock`. Running `pip install` directly would install into the environment without Poetry knowing about it.
- **Figure A.4's selector (Stable 2.0.1, CUDA 11.7/11.8)** - those were the versions when the book was written. This project pins PyTorch 2.4.0 with CUDA 12.4, below.
- **`torchvision` and `torchaudio`** - the book says they're optional, and nothing in it needs them. Install only `torch`.

This project pins **PyTorch 2.4.0**. That version predates CUDA 13, so it comes from the `cu124` (CUDA 12.4) build — the newest CUDA build of 2.4.0 for Python 3.12 on Windows.

**1. Add PyTorch's GPU package source**
```powershell
poetry source add --priority=explicit pytorch-cu124 https://download.pytorch.org/whl/cu124
```

- `pytorch-cu124` - the name Poetry gives this source
- `cu124` - "built for CUDA 12.4." CUDA is NVIDIA's software for running calculations on the graphics card
- `--priority=explicit` - Poetry only uses this source for packages you specifically point at it; everything else still comes from PyPI

**2. Install PyTorch from that source**
```powershell
poetry add torch==2.4.0 --source pytorch-cu124
```

This is a download of a few GB.

**3. Check it can see the graphics card**
```powershell
poetry run python src/build_llm_from_scratch/chapter_00_learn_pytorch/01_check_pytorch.py
```

Expect the version `2.4.0+cu124`, `True` for CUDA, and your RTX 4070's name. If CUDA shows `False`, PyTorch is running on the CPU only. Ask for help debugging it.

Why this works on your card: your NVIDIA driver (610.88) supports CUDA 13, and a driver can always run builds made for an older CUDA, so 12.4 is fine.

`pyproject.toml` still has an unused `pytorch-cu132` source from an earlier attempt at PyTorch 2.15. Nothing installs from it; remove it with `poetry source remove pytorch-cu132` to tidy up.

#### Troubleshooting: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"

**What it means:** PyTorch's core is compiled C++, not Python. When it was built, the builders linked it against a specific version of NumPy (the standard Python library for number arrays), so tensors can convert to and from NumPy arrays. The PyTorch 2.4.0 Windows build was linked against **NumPy 1.x**. NumPy 2.0 changed its internal C interface, so a module built for 1.x can't talk to 2.x. That's the `_ARRAY_API not found` line.

**Why NumPy 2 is installed at all:** this project never asked for NumPy. `pymupdf` (the library behind `tools/book_page.py`) depends on it with no upper limit, so Poetry installed the newest one, 2.5.5. PyTorch 2.4.0 doesn't declare which NumPy it needs, so Poetry had no reason to pick an older one.

**Is it broken?** Partly. It's a warning, not a crash, and the script still printed its tensors, because plain tensor maths doesn't touch NumPy. Anything that converts between tensors and NumPy arrays (`tensor.numpy()`, `torch.from_numpy(...)`) would fail, and the book uses those later.

**The fix:** keep NumPy on 1.x, which PyTorch 2.4.0 was built for. `pymupdf` accepts any NumPy, so it's happy with 1.x too:

```powershell
poetry add "numpy<2"
```

Then rerun the script. The warning should be gone. (The other fix would be a newer PyTorch, built against NumPy 2, but this project pins 2.4.0.)

#### "Which CUDA am I on?" - you aren't on one

There's no single "CUDA version" installed on your machine that you have to match. Two different numbers are involved:

- **Your driver's maximum** - `nvidia-smi` shows "CUDA Version: X.Y" in its top-right corner. That isn't something installed; it's the *newest* CUDA your graphics driver can run. Updating the driver raises it.
- **The CUDA that PyTorch is built with** - the `cu124` in the URL. PyTorch's GPU package ships with its own copy of the CUDA libraries inside it, so you don't install CUDA separately.

The only rule: **PyTorch's CUDA version must be at or below your driver's maximum.** Your driver supports CUDA 13, so `cu124` (12.4) fits with room to spare. If `nvidia-smi` ever showed a number below 12.4, you'd update the driver or pick an older build.

Between two builds that both fit, use whichever [pytorch.org](https://pytorch.org/get-started/locally/)'s selector currently shows as the default for your PyTorch version. For learning, small CUDA version differences don't matter: the book's code runs the same.

#### Where the Name Comes From

`nvidia-smi` stands for **NVIDIA System Management Interface**. "System management" is the job: reporting and controlling the state of the graphics card (temperature, memory used, which programs are running on it, power draw). "Interface" because it's a command-line front end to NVIDIA's management library underneath, the NVIDIA Management Library (NVML). It's the command to reach for whenever you want to know what your GPU is doing, for example during training in Chapter 5, to see how much of its memory your model is using.

`CUDA` was originally short for **Compute Unified Device Architecture**. NVIDIA introduced it in 2007, when graphics cards were built only for drawing graphics. The idea was "unified": one general-purpose way to run any computation on the card's thousands of small cores, not just graphics. NVIDIA has since dropped the expansion and treats CUDA as a plain name.

#### Verifying the setup yourself

Don't take the commands above on trust. Each part can be checked against an official source.

**Before installing:**

- **The URL is PyTorch's official one.** Go to [pytorch.org](https://pytorch.org/get-started/locally/) and pick Stable, Windows, Pip, Python, and a CUDA version. The command it shows ends in `--index-url https://download.pytorch.org/whl/cu1XX`. That's the same server the `[[tool.poetry.source]]` block points at.
- **The `explicit` pattern is Poetry's documented approach.** Poetry's docs on [package sources](https://python-poetry.org/docs/repositories/#package-sources) describe `priority = "explicit"`: the source is only used for packages that name it.
- **Your driver supports that CUDA version.** Run `nvidia-smi` in a terminal. The top-right corner shows "CUDA Version: X.Y", the highest CUDA your driver can run. It must be at least the version in the URL (12.4 for `cu124`).
- **The build exists for your setup.** Open `https://download.pytorch.org/whl/cu124/torch/` in a browser. Look for `torch-2.4.0+cu124-cp312-cp312-win_amd64.whl`: `cp312` means Python 3.12, `win_amd64` means 64-bit Windows.

**After installing:**

- **Poetry got torch from the right place.** Search `poetry.lock` for `name = "torch"`. Its entry should show the `pytorch-cu124` source and the version `2.4.0+cu124`.
- **It's the GPU build and it sees your card.** Run the check in step 3. The version should end in `+cu124` (a CPU-only build ends in `+cpu`), CUDA should be `True`, and your RTX 4070 should be named.
- **The GPU actually computes.** A final test that puts real work on the card:
  ```powershell
  poetry run python -c "import torch; x = torch.ones(3, device='cuda'); print(x * 2)"
  ```
  Expect `tensor([2., 2., 2.], device='cuda:0')`. The `device='cuda:0'` part confirms the maths ran on the GPU.

## Your Task

- [ ] Read A.1 (pages 251–254) and skim A.1.3 (pages 254–257)
- [ ] Write a rule-based `is_spam` and list emails it would miss
- [ ] Write the plain-Python grid addition with loops, and keep it for A.2
- [ ] Install PyTorch with the three steps above and confirm `torch.cuda.is_available()` prints `True`
- [ ] Write `01_check_pytorch.py` so the GPU check is one command
- [ ] Next: `lesson_02_tensors.md` (A.2)
