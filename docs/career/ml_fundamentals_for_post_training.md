# What Skipping "Basic ML" Means for Post-Training

My mentor recommended going straight into LLMs (this book, then the reasoning book, then the RLHF book) instead of a traditional ML course, because the target role is post-training. This doc covers what that skips, and what it doesn't.

What's skipped splits in two: one part is mostly safe to ignore, and the other part has to be picked up along the way.

## Mostly Safe to Skip: Classical ML Algorithms

- **Tabular data** (spreadsheets, database rows): linear and logistic regression, decision trees, random forests, gradient boosting (XGBoost, LightGBM).
- **Other classics:** SVMs, k-means clustering, PCA, feature engineering.

### Not everything is done in PyTorch

For tabular data, gradient-boosted trees usually beat neural networks, and people use scikit-learn and XGBoost for it, not PyTorch. That's generalist MLE work. Post-training never touches it.

## Don't Skip: The Fundamentals Post-Training Uses

| Fundamental | Why it matters for post-training | Where it's picked up |
|---|---|---|
| Overfitting, bias vs. variance, train/val/test | Every finetuning run can overfit | A.7, Chapters 5–7 (train vs. val loss curves) |
| Evaluation metrics: precision, recall, F1, calibration | Evals work and the job postings | Evals background |
| Statistics: variance, confidence intervals, sample size, "is this difference just noise" | Ramp, Harvey, and Profound all list it | Evals background, plus a review before interviews |
| Probability: softmax, log-likelihood, entropy, **KL divergence** | KL divergence is central in RLHF, PPO, and DPO | Chapter 5, then the RLHF book |
| Optimization: gradient descent, Adam, learning-rate schedules | Every training run | A.7, Chapter 5, Appendix D |
| RL basics: policy, reward, advantage | GRPO and PPO | The reasoning book |

So this path doesn't skip fundamentals. It skips the classical algorithm catalog, and picks up the fundamentals in the context where they're used.

## The One Real Gap: Interviews

Some ML interviews ask general questions:

- "Explain the bias–variance tradeoff."
- "What does L2 regularization do?"
- "Precision vs. recall?"

Post-training interviews ask fewer of these, but they do come up. A few evenings of review before interviewing covers it. A separate course isn't needed.

## Worth Learning Early

**KL divergence** is the one concept worth getting ahead on before the RLHF book. It's basically cross-entropy's sibling, and Chapter 5's cross-entropy section gets you most of the way there.
