# Decision Tree Optimization on the German Credit Dataset

## Problem

Most credit-risk projects default to a multi-model comparison — try five algorithms,
report the one with the highest accuracy, move on. This project deliberately does
the opposite: it holds the algorithm fixed (a single decision tree) and asks a
narrower, harder question — **how far can one tree be pushed, and what does the
tuning process itself reveal about the model and the data?**

The dataset is the [German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
(1,000 loan applicants, 700 labeled "good" credit risk, 300 labeled "bad"). The
task is binary classification: predict whether an applicant is a good or bad
credit risk.

## Why accuracy was the wrong metric to optimize

With a 70/30 class split, a model that predicts "good" for every applicant already
scores 70% accuracy while being financially useless. More importantly, the two
error types aren't equally costly to a lender:

- **False negative** (a bad borrower approved as good) → a loan that likely
  defaults — a real financial loss.
- **False positive** (a good borrower rejected as bad) → a lost customer — a
  smaller, opportunity-cost loss.

This project uses a weighted cost function (`5 x false negatives + 1 x false
positives`, a standard convention for this dataset) as the primary evaluation
metric throughout, rather than accuracy. Every tuning decision — criterion
choice, depth, leaf size, pruning strength, and decision threshold — was made
against this cost function, not against ROC-AUC or accuracy alone.

## Approach

1. **Baseline.** An unconstrained tree overfits severely (train ROC-AUC 1.00,
   validation ROC-AUC ≈ 0.62) — establishing that tuning is necessary, not optional.
2. **Grid search vs. cost.** `GridSearchCV`, optimizing ROC-AUC, selected
   `max_depth=7, min_samples_leaf=12`. This was *not* the cost-minimizing tree —
   a shallower configuration scored slightly lower on ROC-AUC but achieved a
   lower real cost. Optimizing a generic ranking metric and optimizing the
   actual business objective produced different answers.
3. **Validation curves.** Sweeping `max_depth` and `min_samples_leaf`
   independently across cross-validated ROC-AUC showed the two parameters
   behave very differently: `max_depth` has a narrow, high-stakes optimum
   (~4–5); `min_samples_leaf` has a wide, forgiving plateau past ~6–10.
4. **Threshold optimization.** The default 0.5 decision threshold assumes
   symmetric error costs, which isn't true here. Searching the threshold space
   against the cost function reduced cost by ~16% on the reference tree with
   no change to the model itself.
5. **Cost-complexity pruning.** Rather than applying uniform depth/leaf
   constraints, `ccp_alpha` pruning removes individual branches based on
   their actual contribution to the model. This outperformed every manually
   tuned depth/leaf combination, converging on a surprisingly shallow final
   tree (depth 2, 4 leaves).
6. **Stability check.** Feature importances were recomputed across 20 random
   seeds to separate genuine signal from noise-driven splits — one feature
   in the original tree turned out to rest on a leaf of only 10 training
   samples, and the stability check confirmed it was not reliable.

## Results

| Stage | Configuration | Cost | ROC-AUC |
|---|---|---|---|
| Baseline (unconstrained, default threshold) | gini, no constraints | 219 | 0.720 |
| Grid search choice (default threshold) | gini, depth=7, leaf=12 | 219 | 0.720 |
| Grid search choice (tuned threshold) | gini, depth=7, leaf=12 | 183 | 0.720 |
| Depth/leaf sweep, best found | gini, depth=4, leaf=10 | 176 | 0.7075 |
| **Final: cost-complexity pruning** | **gini, ccp_alpha=0.0078** | **168** | **0.7072** |

Final model: recall on bad-risk applicants = 0.87, precision on bad-risk = 0.42
— a deliberate tradeoff, catching the large majority of actual risk at the
cost of over-flagging some good applicants, consistent with the 5:1 cost
asymmetry defined above.

## Key finding

The final tree uses only three features in a stable, interpretable two-level
structure: **checking account status**, then **loan duration** or
**credit-to-age ratio**. Every other feature that appeared in an unstabilized
run turned out to be noise once checked across multiple random seeds — a
reminder that a single decision tree's feature importances should not be
trusted without a stability check.

Validation ROC-AUC plateaued around 0.70–0.72 regardless of tuning strategy,
suggesting this is close to the structural ceiling for a single tree on this
feature set — the natural next step would be ensemble methods, which is
intentionally out of scope here.

## Repository contents

- `German_Credit.ipynb` — full analysis, in order: EDA → encoding →
  baseline trees → grid search → cost function and threshold optimization →
  validation curves → cost-complexity pruning → final tree interpretation
  and stability analysis.

## Tools

Python, pandas, scikit-learn, matplotlib, seaborn.
