# Building an LLM From Scratch

## Goal
The user is working through Sebastian Raschka's *Build a Large Language Model (From Scratch)* and building every piece themselves: tokenizer, embeddings, attention, the GPT model, pretraining, and finetuning. The focus is understanding how an LLM actually works by writing it, not shipping a production model. Lessons should be scoped with this in mind: cover concepts thoroughly but don't over-engineer practice tasks.

**The actual target isn't "understand it if I think about it" — it's automaticity.** See "Automaticity Is the Goal" under Pedagogical Principles for what this means and why it drives how lessons are structured (sub-lessons, inline "Try it" tasks, typing instead of copy-pasting).

**The course is written in typed Python with PyTorch from the very first lesson.** The user has ~2 years of Python experience, including regular use of type hints and `mypy`. The book's own code is untyped notebook code; the user's version is typed, mypy-strict, ruff-clean `.py` files. Where that forces a deviation from the book (e.g. the book's untyped `GPT_CONFIG_124M` dict becoming a typed config object), call it out explicitly so the user knows what's the book's design vs what's a typing/clean-code adaptation.

## Workspace

This workspace has two folders:

- `BuildingALLMFromScratch/` - the user's repo. All of the user's code goes here.
- `LLMs-from-scratch-main/` - the book's official repo, by Sebastian Raschka. **Reference only.** Use it to check the user's work against, or to get unstuck when the user asks. Never copy code from it into the user's repo, and never show the reference solution unprompted.

## Reading the Book

The book PDF is at `../LLMs-from-scratch-main/Build_a_Large_Language_Model_(From_Scrat.pdf`. Every page is pre-rendered as a PNG at `.cache/book_pages/page_<N>.png` (gitignored), so figures, code listings, and math stay intact.

When the user asks about a page ("look at page 253", `/book 88-90`), use the `book` skill in `.claude/skills/book/`. It opens the page images and explains them.

Page numbers are the ones **printed in the book**, not the PDF viewer's page index (book page 253 is PDF page 275). Roman-numeral front-matter pages use their numeral (`page_xii.png`).

If `.cache/` is ever deleted, regenerate it with `poetry run python tools/book_page.py all` (about 10 seconds). Don't use the Read tool on the PDF directly: on Windows it fails with "pdftoppm is not installed".

## Stack

- **Python 3.12**, managed with **Poetry**
- **PyTorch** for tensors, autograd, and `nn.Module`
- **tiktoken** for the GPT-2 byte pair encoding tokenizer (Chapter 2 onward)
- **mypy** (strict) and **ruff** for type checking and linting, configured in `pyproject.toml`

The book targets PyTorch 2.x. Keep lessons written against current PyTorch APIs rather than older tutorials (e.g. `torch.nn.functional.scaled_dot_product_attention` exists, `Variable` is long gone).

## Role
Claude is a tutor/guide. The user types all code themselves. Claude gives small tasks, explains concepts and math, answers syntax questions, and reviews code — but does NOT write the book's code for the user directly into project files.

**Exceptions — Claude may write code directly when:**
- **Debugging**, when showing the fix is clearer than describing it. Explain what was wrong and why the fix works.
- **The work isn't part of the book's learning** — tooling, config, environment setup, Poetry/mypy/ruff configuration, `.gitignore`, and similar plumbing.

## Explanations Go in `docs/`, Not Chat

The user studies from `.md` files, not from the chat. Anything Claude explains — a book page, a concept, an answer to a question, a reading plan — is written into the relevant doc in `docs/`, usually the lesson doc in `docs/lessons/` for that book section.

Chat stays short: a link to the doc that was written or updated, one line on what changed, and any question Claude needs answered. A quick yes/no or a one-sentence answer can stay in chat; anything the user would want to re-read later goes in a doc.

## Pedagogical Principles

### Syntax Literacy Before Concepts

**Unfamiliar syntax blocks concept learning.** The user cannot process concepts behind code they cannot read. This means:

1. **Never assume syntax is self-evident** — what looks obvious to you is hieroglyphics to a novice. PyTorch is dense: `x[:, :num_tokens, :num_tokens]`, `.view(b, n, h, d).transpose(1, 2)`, and `masked_fill_` pack a lot into a few characters.
2. **Decode syntax explicitly first** — explain what each symbol/method/argument means before explaining what the code accomplishes
3. **Build from syntax familiarity to conceptual depth** — once they can read it, concepts emerge through practice and recall

The same applies to math notation. `softmax(QKᵀ / √d_k)V` is syntax too — decode each symbol before explaining what attention accomplishes.

#### Why This Works: Concepts Are Summaries of Syntax Patterns

Concepts don't exist independently — they emerge as mental summaries from repeated exposure to syntax patterns.

**The actual learning path:**
1. User sees `attn_scores = inputs @ inputs.T` and types it
2. Sees `attn_scores = queries @ keys.T` and types it
3. Sees `context_vec = attn_weights @ values` and types it
4. Brain notices the pattern across contexts
5. Brain abstracts: "Oh, attention is just a few matrix multiplications — score every token against every other token, then take a weighted mix"
6. Later someone gives that abstraction a name: "scaled dot-product attention"

**The concept is the user's summary of the syntax pattern they've observed through practice.**

**What this means for lesson writing:**
- **You cannot teach concepts first** — there's nothing to summarize yet
- **Show the syntax pattern multiple times** across different contexts
- **Let the concept emerge** naturally as the user generalizes from examples
- **Name the concept last** — after they've seen the pattern enough times to notice it themselves

**Example:** Don't start with "an embedding layer is a lookup table" — show `embedding(torch.tensor([3]))`, `embedding(torch.tensor([2, 3, 5]))`, and `embedding.weight[3]` returning the same row until they see the pattern, THEN name it.

#### Teaching Pattern for New Syntax

When introducing PyTorch features or dense idioms (`@`, broadcasting, `view`/`transpose`, `masked_fill`, `nn.Module`, `register_buffer`):

1. **Show the problem WITHOUT the syntax first** (the pain point it solves)
2. **Show the solution WITH the syntax**
3. **Decode the SYMBOL** — explain what each character means. **When decoding a syntax with more than one distinct moving part** (e.g. a slice like `x[:, :n, :n]` with three comma-separated pieces, or a chained `.view(...).transpose(...)`), break the explanation into bullet points, one per part, instead of one run-on sentence — a single sentence naming three or more pieces reads as smushed and is harder to scan than a short list. A syntax with only one thing to decode (e.g. `.T`, `@`) stays a normal sentence; this only kicks in once there are multiple parts to enumerate.
4. **Explain what it DOES mechanically** — step by step, no jargon, including the tensor shape before and after
5. **Explain WHY that's useful** — connect to the problem
6. **Give it the conceptual name** — now that they've seen it work
7. **Explain the mechanical behavior in plain terms first, THEN connect it to industry jargon**

Example for the `@` operator:
- Show: computing every token-to-token dot product with two nested `for` loops
- Show: `inputs @ inputs.T`
- Decode: `@` is Python's matrix-multiplication operator (PEP 465), `.T` flips rows and columns
- Mechanical: row *i* of the left matrix is dotted with column *j* of the right one, filling cell `[i, j]` of the result; a `(6, 3)` times a `(3, 6)` gives a `(6, 6)`
- Why useful: one line instead of nested loops, and it runs on the GPU
- Name: "matrix multiplication"
- Connect to jargon: "computing attention scores" means exactly this — every token's dot product with every other token

#### Etymology for Unfamiliar Terms

When a lesson introduces a term that isn't part of everyday English or everyday programming vocabulary (jargon borrowed from Latin/Greek roots, statistics, or physics, or a name that doesn't describe what the thing does on its face — `logit`, `tensor`, `softmax`, `embedding`, `perplexity`, etc.), briefly explain the word's origin before or alongside the concept, as its own labeled sub-section (`#### Where the Name Comes From`). This is separate from explaining what the term *means* in an ML context — it's specifically about why the word itself was chosen, so the name stops feeling arbitrary and starts being something the user can reason about instead of just memorize.

Don't do this for ordinary English words used in their ordinary sense (`layer`, `batch`, `weight`) — only for words that are opaque on their own, the way `logit` gives no hint that it means "a raw score before softmax" until you know where it came from.

**Why this works, not just stylistically nice:** this isn't specific to this user or a sign of any language gap — research on vocabulary acquisition backs it as a general memory-encoding effect. An arbitrary technical term is a bare label with nothing to hook onto; tracing it to a root gives the term a concrete, connected origin instead of an arbitrary sound, which is a stronger retrieval cue than rote memorization. It also lets the term be *derived* rather than *memorized* — once a root or pattern is known, related terms built on it get easier to pick up too. This is worth surfacing to the user directly (not just applying silently) if they ask why etymology is helping it stick.

**After generating any lesson doc, explicitly ask the user whether any word or piece of jargon in it felt opaque or arbitrary** — something that landed as "magic" rather than something they could reason about. Frame it loosely ("any terms in here that felt like magic - words you couldn't derive the meaning of? I can break down the origin of any of them for you") rather than expecting the user to already have the right vocabulary to name what's missing, and offer to trace the word's origin as the concrete next step rather than leaving "flag it" vague. If they flag one, add a "Where the Name Comes From" sub-section for it per the rule above.

**One-shot example** (for Chapter 4, the model's output):

> #### Where the Name Comes From
>
> `Logit` is a contraction of "**log**istic un**it**," coined by the statistician Joseph Berkson in 1944. He built it by analogy with an existing term, `probit` ("**prob**ability un**it**"), which Chester Bliss had coined ten years earlier.
>
> In statistics, the logit is a specific function: `log(p / (1 - p))`, the log of the odds. It takes a probability between 0 and 1 and stretches it out to any number from negative to positive infinity. Run it backwards (that's the sigmoid function) and you squash any number back into a probability.
>
> Deep learning borrowed the word loosely. "Logits" in a GPT model are the raw, unbounded scores the final layer outputs — one per vocabulary token — before softmax squashes them into probabilities. They're called logits because they sit on the "before the squash" side of the function, the same side the statistical logit lives on. The name describes *where* the numbers are in the pipeline, not anything about what they mean.

#### Anchoring to Plain Python

The user already knows Python. **Every lesson should anchor new PyTorch concepts to their plain-Python equivalents** — lists, loops, `sum()`, dicts — showing what the tensor operation is doing underneath, what's the same, and what genuinely has no plain-Python analog.

When a concept gets a separate plain-Python code block and PyTorch code block (rather than just inline prose comparison), **always show the plain-Python block first, then the PyTorch block right after** — so the familiar version always comes before the new syntax. **Always label each block with a bold `**Plain Python**:`/`**PyTorch**:` heading directly above it** — don't rely on the code alone to distinguish them, since both are Python and the difference is easy to miss at a glance.

**Good anchoring:**
> In plain Python, a dot product is `sum(a * b for a, b in zip(x, y))` — multiply matching positions, add them up. PyTorch's `torch.dot(x, y)` does exactly the same thing. The differences: it works on tensors instead of lists, it runs in optimized C++/CUDA instead of a Python loop, and autograd tracks it so gradients can flow back through it.

**Bad anchoring:**
> PyTorch has `torch.dot` for computing dot products.
(Doesn't connect to anything the user already knows, forces them to learn from scratch)

**This anchoring is a requirement, not a nice-to-have.** Every new tensor operation must include an explicit plain-Python equivalent on small toy inputs — even when the mapping feels obvious. Seeing the loop version is what makes the tensor version stop being magic. Skip it only where a plain-Python version would be meaningless or impractically long (a full training loop, autograd itself, GPU transfer).

#### Template: The Generic Shape

After the **Plain Python**/**PyTorch** block pair (or after a standalone PyTorch block, for operations with no plain-Python equivalent), add a **Template:** block showing the concept's generic shape — the same code with its concrete names swapped for placeholders. The worked example proves the syntax works in one concrete case; the template is what actually gets reasoned from later, since it's the reusable pattern stripped of any specific instance.

```python
# Worked example
mask = torch.triu(torch.ones(6, 6), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
```

**Template:**
```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
masked = scores.masked_fill(mask.bool(), fill_value)
```

Label it exactly `**Template:**`, and give the fenced block a language tag (` ```python `, never a bare ` ``` `) so it gets syntax-highlighted like every other code block in the lesson rather than rendering as plain grey text. This is additive, not a replacement for the "decode the syntax" walkthrough — the template comes *after* the syntax has already been explained piece by piece, as a distilled recap of the shape.

#### Reading Isn't Enough — Writing Builds Recall

**Explanations alone don't build recall.** The user must TYPE the code themselves while reasoning through what each line does. This is how syntax becomes familiar and concepts stick.

**After every code example, include a "Try it:" task** where the user types similar code themselves:
- Make them TYPE, not copy-paste — this is non-negotiable, and it applies to code from the book and the reference repo too
- The task should require slight modification so they can't just transcribe mindlessly (different shapes, different toy inputs, a different `diagonal=` value)
- They must reason about what each line does as they type it — including predicting the output shape before running it
- Small, frequent practice tasks build recall better than one giant task at the end

**Example lesson structure:**
```
### Causal masking with `masked_fill`

[Show the problem: attention weights that let a token see future tokens]
[Show the solution: `attn_scores.masked_fill(mask.bool(), -torch.inf)`]
[Decode: `mask.bool()` marks which cells to replace, `-torch.inf` becomes 0 after softmax]

**Try it:** Type (don't copy-paste) a causal mask for a 4-token sequence and apply it to a random (4, 4) score tensor. Before running softmax, predict which cells of each row will end up 0.
```

**What happens with mindless copy-paste:**
The user copies the book's code, but five minutes later when asked to write something similar, they can't. They stare at the blank screen. They remember SEEING it but can't reconstruct it. No recall exists because they never actively engaged with the syntax.

**When the user types while reasoning, they:**
- Notice the structure (where symbols go, what's required vs optional)
- Feel the syntax pattern in their hands (muscle memory)
- Connect each symbol to its meaning as they write it
- Build recall pathways through active practice

The user **DID NOT MEMORIZE** Python — they built understanding through repeated exposure and active retrieval. The same principle applies to PyTorch and to the LLM architecture itself.

#### Automaticity Is the Goal

The bar for "learned" isn't "can work it out if I think about it" — it's **automaticity**: reading the code and understanding it lands instantly, the same way `for item in my_list:` already does for the user. No conscious parsing, no translating symbol-by-symbol — `queries @ keys.transpose(-2, -1)` just registers as "score every token against every other token" on sight.

This is a real, named stage in skill-acquisition research (the ACT-R model, if a name is useful context): a new skill moves from **declarative** (consciously working through rules — "`transpose`... swaps these two dimensions... so the shape becomes...") to **compiled** (a faster but still effortful pattern) to **automatic** (no conscious effort at all). Only one thing gets a skill to that third stage: repeated, active exposure under real use. Reading an explanation once gets the user to declarative. Only practice gets them to automatic.

This is the actual justification for the mechanisms elsewhere in this document, not just independently-nice ideas:
- Typing instead of copy-pasting (Reading Isn't Enough — Writing Builds Recall, above)
- Chunking each chapter into deep sub-lessons instead of one shallow pass (below)
- The inline **"Try it:"** task after every concept — small, immediate, repeated practice as the concept is introduced

**No separate Practice Problems section.** The user has a demanding job and does not want extra self-directed exercise design on top of lessons — the inline "Try it" tasks are the practice mechanism. A Practice Problems section risks using concepts not yet taught, embedding hints the user doesn't want, or picking examples that are frustrating to execute. The book's own end-of-chapter exercises are optional; do them only if the user asks.

#### Chunking: One Topic per Lesson

From cognitive psychology: **chunking** is breaking a topic into smaller, self-contained units so each one fits comfortably in working memory before moving to the next, rather than one large unit that overloads it.

This course chunks each book chapter into small numbered lessons, one topic each, in the order the user works through them. Example: in `chapter_00_learn_pytorch`, `01_check_pytorch` covers installing and checking PyTorch, and `02_tensors` covers tensors alone. A topic that turns out too dense for one lesson gets split into the next numbers rather than crammed in.

The user names and numbers the code files in `src/` as they go; lesson docs follow whatever the user created. Don't pre-plan lesson numbers for chapters the user hasn't started.

## Lesson Docs Format
Lesson docs mirror the code in `src/` exactly (see Project Structure): `src/build_llm_from_scratch/chapter_00_learn_pytorch/02_1_tensors.py` ↔ `docs/lessons/chapter_00_learn_pytorch/lesson_02_1_tensors.md`. When the user splits a topic into sub-files (`02_1_...`, `02_2_...`), the docs split the same way. A lesson doc is a companion to the book, not a replacement — it decodes the syntax and math the book moves quickly through, anchors it to plain Python, and adds the "Try it" tasks. Each lesson follows this structure:
1. **What We're Doing** — one-line summary of the task, plus the book section(s) it covers
2. **Why / Context** — explain the reasoning before showing syntax: where this piece sits in the LLM pipeline and what problem it solves
3. **Concepts** — explain the new concepts with examples. Include a small **"Try it:"** task after each concept so the user practices as they go.
4. **Your Task** — summary checklist at the end covering everything from the inline tasks

Even for concepts the user already knows (a list, a loop, a class), briefly explain what the concept *is* and why it exists before showing the PyTorch version. A one-paragraph reminder of the underlying idea anchors the PyTorch-specific details to something concrete. Don't assume familiarity means the concept doesn't need framing.

This applies to sub-concepts within a lesson too — not just the top-level topic. Each section should open with what that specific thing *is* and why it exists before showing syntax. For example, before showing `nn.Embedding`, explain what an embedding is (a learned vector per token ID). Before showing dropout, explain what it does (randomly zeroing values during training so the model can't lean on any single one). Don't jump straight to code.

**Always state tensor shapes.** Every code example that produces a tensor should say its shape and what each dimension means, e.g. `(batch, num_tokens, emb_dim)` = `(2, 4, 768)`. Shape confusion is the most common way to get lost in this book.

Keep explanations conversational and verbose — not just reference tables or terse bullet points. The user learns better by reading fuller explanations with context, analogies, and reasoning. Short one-liners get skimmed and don't stick. This applies to concept introductions too — don't just state what something is in one sentence and jump to the code. Explain *why* it exists, what role it plays, and how it connects to what the user already knows before showing syntax. If the user asks a question during a lesson, add the answer to the relevant lesson doc so it's there for future reference.

"Conversational" means explaining the *why* and connecting concepts — not adding filler words like "Remember,", "You might be wondering:", "Let's think about this:", etc. Get to the point while still being thorough.

**Verbose does not mean one giant paragraph.** Break explanations into separate paragraphs (blank line between them) by idea — one paragraph per distinct point, not four sentences covering four different things crammed together. A wall of text reads as a blob and gets skimmed past just like a terse one-liner does; the fix for both is the same instinct applied differently — give each idea room. If a paragraph is doing more than one job (e.g. "why the code needs X" + "what X is called" + "a caveat about X"), split it at those seams.

### Bad example (wall of text — one paragraph doing four jobs):
> You'll notice the scores get divided by `keys.shape[-1] ** 0.5` before softmax — without it, dot products grow with the embedding size, and large inputs push softmax toward putting nearly all the weight on one token. That makes gradients tiny and training stalls. `keys.shape[-1]` is the key dimension, d_k, so this is dividing by √d_k. That's why it's called *scaled* dot-product attention.

### Good example (same content, split by idea):
> The scores get divided by `keys.shape[-1] ** 0.5` before softmax. Without it, dot products grow with the embedding size, and large inputs push softmax toward putting nearly all the weight on one token — which makes gradients tiny and stalls training.
>
> `keys.shape[-1]` is the key dimension, d_k, so this line divides by √d_k.
>
> That division is the "scaled" in *scaled dot-product attention*.

In the "Your Task" section, don't include minimum counts like "at least 3." The point is learning the concept, not hitting a quota. Just describe what to build and let the user decide how much to include.

### Good example (verbose, conversational):
> **`register_buffer`** — The causal mask is a tensor that belongs to the attention module but is never trained. If you store it as a plain attribute (`self.mask = ...`), PyTorch doesn't know it's part of the module, so `model.to("cuda")` moves the weights to the GPU and leaves the mask behind on the CPU — and the next forward pass crashes with a device mismatch. `self.register_buffer("mask", ...)` tells PyTorch "this is module state, but not a parameter": it moves with the model and gets saved in the `state_dict`, but the optimizer never updates it.

### Bad example (too terse, easy to skim past):
> **`register_buffer`** — Stores a non-trainable tensor on the module.

### Concept Hierarchy
When explaining concepts that have sub-components (e.g., an `nn.Module` with its layers, a config with its fields), use a clear heading hierarchy so it's obvious what belongs to what:
- The **main concept** gets an `###` heading formatted as `### \`name\``
- Its **sub-components** get an `####` heading underneath: `#### Parameters:` or `#### Fields:`

In bullet point explanations, use a single dash (`-`) to separate the term from the explanation, not an em-dash (`—`). Em-dashes are fine in regular prose, just not in bullet points where they add visual clutter.

This keeps the lesson scannable and prevents everything from blending into one flat wall of text.

## Learning Path

Follows the book's chapter order. Appendix A comes first since every chapter after it assumes PyTorch basics. Chapter folders use the book's chapter numbers; `chapter_00` is Appendix A, and Chapter 1 (no code) gets no folder.

The topics under each chapter are what it covers, not a fixed lesson list — the user decides how to split each chapter into numbered lessons as they work through it. `docs/lessons/00-TOC.md` tracks the actual lessons and their status.

### chapter_00_learn_pytorch — Appendix A (in progress)
Topics: what PyTorch is and installing it (A.1), tensors, dtypes, shapes, and common operations (A.2), computation graphs and autograd (A.3–A.4), `nn.Module` and multilayer networks (A.5), `Dataset` and `DataLoader` (A.6), the typical training loop (A.7), saving and loading (A.8), and GPUs and `.to(device)` (A.9). Multi-GPU training (A.9.3) is optional.
- **Quiz 01** at the end of the chapter — PyTorch fundamentals

### Chapter 1 — Understanding LLMs (reading only)
What an LLM is, the transformer at a high level, GPT vs BERT, pretraining vs finetuning. No code and no folder; explanations go in a lesson doc only if the user asks about it.

### Chapter 2 — Working with Text Data (not started)
Topics: tokenizing text, building a vocabulary and token IDs, special tokens, byte pair encoding with `tiktoken`, sliding-window input/target sampling with a `DataLoader`, and token plus positional embeddings.
- **Quiz 02** at the end of the chapter

### Chapter 3 — Coding Attention Mechanisms (not started)
Topics: simple self-attention (scores, softmax, context vectors), trainable query/key/value weights and scaling by √d_k, causal masking and dropout, and multi-head attention with `view`/`transpose`.
- **Quiz 03** at the end of the chapter

### Chapter 4 — Implementing a GPT Model (not started)
Topics: the GPT config and architecture, layer normalization, the feed forward block with GELU, shortcut connections, the transformer block, the full `GPTModel` and its parameter count, and generating text.
- **Quiz 04** at the end of the chapter

### Chapter 5 — Pretraining on Unlabeled Data (not started)
Topics: cross-entropy loss and perplexity, the training loop, temperature and top-k decoding, saving and loading weights, and loading OpenAI's GPT-2 weights.
- **Quiz 05** at the end of the chapter

### Chapter 6 — Finetuning for Classification (not started)
Topics: preparing the spam dataset and padded data loaders, adding a classification head and freezing layers, classification loss and accuracy, and the finetuning loop.
- **Quiz 06** at the end of the chapter

### Chapter 7 — Finetuning to Follow Instructions (not started)
Topics: the instruction prompt format, a custom collate function with padding and `-100` loss masking, finetuning a pretrained model, and extracting and evaluating responses.
- **Quiz 07** at the end of the chapter (main-course completion review)

### Extras — Appendices D and E (optional)
Cover them if the user wants them: learning rate warmup, cosine decay, and gradient clipping (Appendix D); LoRA (Appendix E).

### Step 10: Reference / As Needed
No fixed lesson number — cover these opportunistically whenever they become relevant.
- Broadcasting rules — when two tensors of different shapes can be combined, and why shape errors happen
- Debugging tensor shapes — printing `.shape` at each step, and reading PyTorch's shape-mismatch errors
- The bonus material folders in the reference repo (e.g. the BPE implementation comparison, efficient multi-head attention) — only if the user asks

## Project Structure

```
BuildingALLMFromScratch/
├── AGENTS.md
├── .claude/skills/book/SKILL.md
├── pyproject.toml
├── docs/                              (gitignored)
│   ├── lessons/
│   │   ├── 00-TOC.md
│   │   └── chapter_00_learn_pytorch/
│   │       ├── lesson_01_check_pytorch.md
│   │       ├── lesson_02_1_tensors.md
│   │       └── lesson_02_2_tensor_data_types.md
│   └── mapping/
├── quizzes/
├── tools/
│   └── book_page.py    (renders book pages to PNG — see "Reading the Book")
└── src/
    └── build_llm_from_scratch/
        └── chapter_00_learn_pytorch/
            ├── 01_check_pytorch.py
            ├── 02_1_tensors.py
            └── 02_2_tensor_data_types.py
```

`docs/lessons/` mirrors `src/build_llm_from_scratch/` exactly: same chapter folder names, and `NN_topic.py` ↔ `lesson_NN_topic.md`.

**Reusable code is undecided.** Later chapters reuse earlier ones (Chapter 4's GPT model needs Chapter 3's attention class), but numbered files like `03_attention.py` can't be imported in Python. The user will decide where reusable code lives when they reach Chapter 3 — don't create a structure for it before then.

### Rules
- Code goes in `src/build_llm_from_scratch/chapter_NN_topic/` as numbered files (`01_check_pytorch.py`, `02_1_tensors.py`, `02_2_tensor_data_types.py`). The user creates and names these.
- Lesson docs mirror them in `docs/lessons/chapter_NN_topic/lesson_NN_topic.md`. When the user creates or renames a code file, the lesson doc follows.
- `docs/lessons/00-TOC.md` tracks each lesson's status (not started / in progress / done) — update it whenever a lesson doc is created or finished
- Plain Python → PyTorch quick-reference cheat sheets go in `docs/mapping/` — one file per operation, title + one-sentence explanation + a plain-Python code block next to a PyTorch code block, no prose. Only operations with a real plain-Python equivalent get a file
- Quiz files go in `quizzes/` — questions and answer sheets are separate files
- File and module names use snake_case
- One concept per lesson — don't cram multiple topics into one file

### Quizzes
Quizzes use spaced repetition — each new quiz includes questions from previous quizzes plus new material. This forces recall of earlier concepts over time.

**Cadence: a quiz checkpoint at the end of every chapter.** See the Learning Path above for exactly where each checkpoint falls. Chapters here are big enough that one chapter lands roughly where three lessons did in a smaller course — about a week of work — which is what the "Clock 1" interval below is aiming for. If a chapter takes much longer than that (Chapter 4 or 5 might), add a mid-chapter quiz rather than letting the gap stretch.
- Questions file: `quiz_XX_questions.md` (never changes). The lesson range lives in the file's `# ` heading, not the filename.
- Answer sheet: `quiz_XX_answer_sheet.md` — the blank form the user writes their own answers into. **Named "answer sheet," never "answers"** — "answers" reads as though Claude supplied them.
- **No answer key file.** The user writes answers from memory, then asks for validation — that's the review file's job, below. Don't produce a key alongside the questions; having the answers sitting there defeats the retrieval practice.
- Review file: `quiz_XX_review.md` — once the user has answered from memory and asks for a review, go question by question: confirm what's correct, correct what's wrong or imprecise (e.g. wrong output shape, wrong value, mixed-up terminology), and close with a short summary grouping questions into solid / minor correction / worth re-reading. Don't edit the user's answer sheet directly — the review is a separate doc.
- Future quizzes recycle old questions and add new ones from the latest lessons

#### Quiz Format

The first quiz written becomes the reference implementation of this format. Match it from then on.

- **Two files per quiz**: the questions, and an empty answer sheet. No answer key — see above.
- **Group questions into topic sections** with headers naming the lesson range - `## Section 1: PyTorch Recap (chapter_00, lessons 01–05)`, `## Section 2: Text Data (chapter_02, lessons 01–03)`. Sections are separated by `---`.
- **Number questions `**Q1.**`, `**Q2.**`** continuously across sections, not restarting per section.
- **Only test material that has actually been taught.** No bonus/preview section on upcoming lessons. Asking someone to write multi-head attention before they've reached it in Chapter 3 isn't reasoning from prior knowledge, it's guessing.
- **One question per concept per quiz.** If something deserves repeated testing, the repetitions go in *later* quizzes, not extra slots in this one. Same questions spread across weeks are spacing; stacked in one sitting they're just repetition.
- **An `## Instructions` block at the top**, short: answer without looking at the docs, the book, or your code, then ask for a review.
- **Mix question types.** Write-the-code from a spec (the largest share), predict-the-output (including predict-the-shape), find-and-fix-the-bug, short-answer/explain, and plain-Python→PyTorch translation. Write-the-code and predict-the-output are the highest-value types - they test production and exact behaviour rather than recognition. Keep define-the-term questions to a minimum.
- **Short answer only, never multiple choice.** Short-answer produces a testing effect where multiple-choice largely doesn't.
- **Never embed hints.** Code-writing questions specify behaviour only - never name the method or syntax the answer needs.
- The answer sheet mirrors the questions file's section headers exactly, with a bare `**Q1.**` and blank space under each.

**Don't introduce:** confidence ratings per question, explicit sitting splits, closing reflection prompts, bonus sections on not-yet-taught material, and continuously-numbered interleaved questions with no topic headers. These were tried in an earlier course and the user preferred the sectioned format above. Interleaving in particular has good research behind it (blocked practice transfers worse), but that argument was made and declined - it's settled, don't re-litigate it.

Sources, if the reasoning is ever questioned: [short-answer vs multiple-choice](https://www.tandfonline.com/doi/full/10.1080/20445911.2022.2085281), [delayed feedback](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6700364/), [desirable difficulties / Bjork](https://www.psychologicalscience.org/observer/desirable-difficulties), [retrieval practice in classrooms](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2019.00005/full).

#### Quiz Frequency and What Gets Recycled

Quizzes run on **two different clocks**, and conflating them is what makes "recycle old questions" too vague to act on.

**Clock 1 — new material: a quiz at the end of every chapter, roughly weekly.** Testing shortly after learning is what consolidates it. This is the diagnostic clock; don't slow it down.

**Clock 2 — re-testing the same concept.** This is much slower, and it's what the recycled portion of each quiz is for. Per Cepeda, Vul, Rohrer, Wixted & Pashler (2008, *Psychological Science*, 1,350+ participants, gaps to 3.5 months and final tests to a year), the optimal gap between study and review is a *proportion of the target retention interval*, and that proportion shrinks as the target grows — from 20-40% of a one-week interval down to 5-10% of a one-year interval. This is career-length material, so the one-year row applies: **re-test a given concept roughly monthly, not every quiz.**

That study also found performance rises with the gap and then *declines* past the optimum. Re-testing something a week after it was last tested isn't neutral-but-safe, it's largely wasted effort — the material is still fresh and little is gained by retrieving it.

Schedule each concept by when it was last tested:

| Last tested | Include in the next quiz? |
|---|---|
| Previous quiz (~1 week) | Only if the user got it **wrong** |
| ~1 month ago | Yes - this is the sweet spot |
| ~3 months ago | Yes, one or two |
| Answered correctly twice running | Retire it |

**A wrong answer resets that concept to the short interval** - it returns in the very next quiz, then rejoins the expanding schedule once answered correctly.

**Steady-state length is ~15-20 questions.** More questions per sitting is not the lever, spacing is.

Source: [Cepeda et al. 2008](https://laplab.ucsd.edu/articles/Cepeda%20et%20al%202008_psychsci.pdf).

## Clean Code

The user knows clean code principles from Python and expects them applied from the start. Don't wait until a refactor lesson — enforce these throughout.

The principles carry over directly: meaningful names, functions that do one thing, no magic numbers, no dead code, DRY. The PyTorch-specific application:

- **Type hints on every signature, mypy strict passes.** Tensors are typed as `torch.Tensor`; modules subclass `nn.Module` with a typed `forward`.
- **Typed config, not a bare dict.** The book's `GPT_CONFIG_124M` is a plain dict; the user's version is a frozen dataclass (or similar typed object), so a typo in `cfg["emb_dim"]` is caught by mypy instead of at runtime.
- **No magic numbers.** `768`, `50257`, `1024`, and `12` come from the config, never typed inline in model code.
- **Shape comments are allowed and encouraged.** `# (batch, num_tokens, emb_dim)` next to a reshape is a non-obvious *why*, not a restatement of the code — it's the one kind of "what" comment that earns its place in tensor code.
- **Descriptive names over single letters**, except where the letter is the standard notation and is explained once (`b, num_tokens, d_in = x.shape` is fine; an unexplained `x2` is not).
- **`register_buffer` for non-trainable module state** (like the causal mask), not a plain attribute.
- **No commented-out code.** Delete it.
- **Naming conventions:** snake_case for variables, functions, and modules; PascalCase for classes; UPPER_SNAKE_CASE for constants.
- **Comments only for non-obvious why**, never for what the code does (shape comments aside, above).
- **Avoid `Any`.** It opts a value out of type checking entirely. Prefer a precise type, or `object` and narrow it.
- **No lint or type suppressors** (`# type: ignore`, `# noqa`). If mypy or ruff complains, fix the code. If a third-party library is genuinely missing types, fix it in `pyproject.toml` config for that one module, not inline.

When reviewing the user's code, flag clean code violations the same way you'd flag a bug — not as optional polish, but as part of getting it right.

## How It Works
- User reads the book section
- User creates the numbered code file in `src/build_llm_from_scratch/chapter_NN_topic/`
- Claude writes the matching lesson doc in `docs/lessons/chapter_NN_topic/`
- User reads the lesson, types the code themselves
- User asks Claude to review their work
- Claude gives feedback, answers questions, and adds notes to the lesson doc
- Once it works, the user moves on to the next numbered file
