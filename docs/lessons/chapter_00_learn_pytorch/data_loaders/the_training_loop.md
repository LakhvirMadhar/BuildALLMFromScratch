# The Training Loop

Every model in PyTorch is trained with the same three steps and the same loop. This doc is the big picture: what the pieces are, where each one shows up in the book, and how the same loop stretches from a toy classifier all the way to post-training methods like DPO and GRPO.

## The Pipeline

```
data → Dataset → DataLoader → training loop → trained model
```

| Piece | Job | Engineering analogy |
|---|---|---|
| Data as tensors | Turn raw data (text, labels) into numbers | Parsing input |
| `Dataset` | "How do I get example `i`?" | A repository with a `get(id)` |
| `DataLoader` | Shuffle, group into batches, hand them out one at a time | A paginated iterator over that repository |
| Training loop | For each batch: predict, measure the error, adjust the weights | The worker that consumes the queue |

The `DataLoader` doesn't train anything. It only hands out batches. The training loop is what actually changes the model's weights.

## PyTorch in Three Steps

1. **Define the dataset** - what one example looks like, and how to batch them.
2. **Define the model** - an `nn.Module` with layers in `__init__` and the data flow in `forward`.
3. **Define the training loop** - predict, compute the loss, backpropagate, update.

### Where each step is in Appendix A

| Step | Book section |
|---|---|
| 1. Define the dataset | A.6, `Dataset` + `DataLoader` |
| 2. Define the model | A.5, `NeuralNetwork(nn.Module)` |
| 3. Define the training loop | A.7 (loss, `backward()`, optimizer step) |

A.7 is the first time all three run together on toy data, and you can watch the loss go down.

### The rest of the book repeats them at larger scale

| Chapter | Dataset | Model | Loop |
|---|---|---|---|
| Ch 2 | text → token chunks | | |
| Ch 3–4 | | the GPT model | |
| Ch 5 | reuses Ch 2 | reuses Ch 4 | pretraining loop |
| Ch 6 | spam / not-spam texts | GPT + classification head | finetuning loop |
| Ch 7 | instruction / response pairs | GPT | instruction finetuning loop |

By Chapter 7 the three steps have been written four times.

## The Loop

```python
for epoch in range(num_epochs):
    for x, y in train_loader:              # DataLoader hands out a batch
        logits = model(x)                  # forward pass: model predicts
        loss = loss_fn(logits, y)          # how wrong was it?
        optimizer.zero_grad()              # clear the previous batch's gradients
        loss.backward()                    # autograd: which way to nudge each weight (A.4)
        optimizer.step()                   # nudge them
```

- **epoch** - one full pass over the training data. Training runs many.
- **`model(x)`** - your `nn.Module` from A.5. Swap in any model and the loop doesn't care, because every `nn.Module` looks the same from the outside.
- **`loss.backward()`** - autograd from A.4.

This loop trains essentially any neural network in PyTorch: a toy classifier, an image model, GPT, a reasoning model. The exception is classic machine learning (random forests, XGBoost), which trains with a single `.fit()` call and isn't part of post-training.

## The Same Loop, Five Ways

These are previews, not code to type yet: `log_softmax`, `gather`, and the DPO formula come later. Read them for what changes (marked `# ←`) and what stays the same.

### 1. Toy classifier (Appendix A.7)

```python
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(num_epochs):
    for x, y in train_loader:            # x: (batch, 2) features, y: (batch,) labels
        logits = model(x)                # (batch, 2) one score per class
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 2. GPT pretraining (Chapter 5)

```python
model = GPTModel(cfg)                                  # ← different model
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

for epoch in range(num_epochs):
    for x, y in train_loader:            # x: (batch, seq_len) token IDs
                                         # y: the same tokens shifted one to the left
        logits = model(x)                # (batch, seq_len, vocab_size)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())   # ← predict every next token
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Only the model and the reshaping in the loss line changed. "Predict the next token" is still classification, with 50,257 classes.

### 3. Instruction finetuning / SFT (Chapter 7)

```python
model = GPTModel(cfg)
model.load_state_dict(torch.load("gpt2_pretrained.pth"))   # ← start from trained weights
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    for x, y in train_loader:            # x: prompt + response tokens
                                         # y: shifted, with prompt positions set to -100
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())   # -100 positions are ignored
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

This is loop 2 with pretrained weights and a different dataset. The `-100` labels mean "don't grade the model on the prompt, only on its answer."

### A helper for loops 4 and 5

Both remaining loops need to know how likely the model thinks a whole answer is:

```python
def sequence_logprob(model, ids, response_mask):
    logits = model(ids[:, :-1])
    token_logprobs = logits.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * response_mask[:, 1:]).sum(-1)   # (batch,) one score per answer
```

### 4. DPO preference tuning (RLHF book)

```python
policy = load_sft_model()                          # the model being trained
reference = load_sft_model().eval()                # ← a frozen copy, never updated
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)

for epoch in range(num_epochs):
    for chosen, rejected, chosen_mask, rejected_mask in train_loader:   # ← pairs: good answer, bad answer
        with torch.no_grad():
            ref_chosen = sequence_logprob(reference, chosen, chosen_mask)
            ref_rejected = sequence_logprob(reference, rejected, rejected_mask)
        pol_chosen = sequence_logprob(policy, chosen, chosen_mask)
        pol_rejected = sequence_logprob(policy, rejected, rejected_mask)

        margin = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
        loss = -F.logsigmoid(beta * margin).mean()  # ← push chosen up, rejected down

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

The data is now pairs and the loss is a new formula. The last three lines haven't changed.

### 5. GRPO-style RL (reasoning book), simplified

```python
for step in range(num_steps):
    for prompts in prompt_loader:                          # ← only prompts, no answers
        with torch.no_grad():
            completions, masks = generate(policy, prompts, num_samples=8)  # ← model writes its own data
        rewards = reward_fn(prompts, completions)          # e.g. 1.0 if the math answer is right
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # better or worse than average?

        logprobs = sequence_logprob(policy, completions, masks)
        loss = -(advantages * logprobs).mean()             # ← make good answers likelier, bad ones less

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Real GRPO adds clipping and a KL penalty, left out here, but the shape is right: generate, score, then the usual three lines.

## What Never Changes

Every version ends with:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

What changes from method to method:

- **the model** - toy network, GPT, GPT with pretrained weights
- **the data** - labeled examples, token chunks, instruction pairs, preference pairs, the model's own generations
- **how the loss is computed** - cross-entropy, the DPO margin, reward-weighted log-probabilities

Post-training methods arrive about once a year (RLHF for LLMs in 2022, DPO in 2023, RLVR/GRPO in 2024–2025), and each one is a new answer to "what goes in `loss = ...`" inside this same loop. The new methods add to the old ones rather than replacing them.

The loop is worth knowing cold, well enough to write it from a blank file. Model architectures aren't: nobody writes Llama from memory, people read the code or the paper.
