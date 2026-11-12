# Pricing Confounding Lab

**Question:** can a demand model fit historical sales yet recommend bad prices
because the historical pricing policy was confounded?

The second grouped OR prototype, connecting predictive analytics and prescriptive
decisions. Themes align with
[Georgia Perakis's pricing/ML/optimization research](https://mitsloan.mit.edu/faculty/directory/georgia-perakis),
[Dimitris Bertsimas](https://dbertsim.mit.edu/),
[Retsef Levi](https://mitsloan.mit.edu/faculty/directory/retsef-levi),
[David Simchi-Levi](https://slevi1.mit.edu/), and
[Garud Iyengar](https://ieor.columbia.edu/garud-iyengar).
These are broad connections, not affiliations, faculty-approved projects, or
claims to implement a specific paper. The Iyengar page was inaccessible during
source verification; no detailed current-research attribution is made.

## Reproduce

Python **3.12**, CPU only. Create a virtual environment with
`python -m venv .venv`; activate with `.\.venv\Scripts\Activate.ps1` on PowerShell
or `source .venv/bin/activate` on Unix.

```text
python -m pip install -r requirements.txt
python -m pytest -q
python benchmark.py --seeds 0 1 2 --output examples/three-seeds.json
```

NumPy and pytest are pinned, not every transitive dependency. No API keys or
retail datasets are required. [Committed results](examples/three-seeds.json)
contain actual seeded runs and fitted coefficients. Hosted CI is not configured.

## Explicit causal assumptions

Synthetic structural mean demand is `max(90 + 24*x - 3*price, 0)`.
Historical prices follow `clip(12 + 5*x + noise, 4, 24)`, so strong demand
contexts get higher prices. Independent Gaussian outcome noise can make observed
sales negative; these are continuous noisy measurements, not integer counts.

The context is observed; adjustment suffices **in this simulator only**, because
there is no unobserved price-demand confounder once conditioning on context.
Historical price noise supplies variation. Recovering a negative coefficient
without these assumptions is not a causal-identification guarantee.

Compare:

- Naive least squares on price alone.
- Context-adjusted least squares.
- Context-adjusted pessimistic pricing using
  `predicted_mean - 2*sqrt(features @ covariance @ features)`, clipped at zero.
- Fixed price 15.
- Oracle maximizing true expected revenue over the **same grid** `[4,24]` in 0.5
  increments. The oracle is an evaluation reference, not a deployable policy.

The pessimistic rule uses ordinary least-squares parameter covariance; it is a
sensitivity heuristic, **not** a simultaneous confidence band, robust causal
certificate, or distributionally robust method. Extrapolation can invalidate it.

Train on 500 observations. Evaluate independently on 400 IID contexts `[-1,1]`
and 400 shifted contexts `[1,2]`, with unchanged structural demand law.
There is no hyperparameter tuning on test outcomes. Metrics are exact expected
revenue, nonnegative grid-oracle regret and average price. All five policies
share the same contexts.

## Why the negative control matters

The confounded regression can learn a *positive* price coefficient even though
every structural demand curve slopes down. Good optimization of a bad model
does not repair that model. Tests demonstrate this reversal, recover known
coefficients under the correct specification, and verify exact oracle behavior.

## Limits and extensions

No strategic customers, price competition, inventory, censored sales, or costs.
Noise is homoscedastic; ordinary least-squares covariance is not robust to
arbitrary misspecification. Compare against randomized pricing data, overlap
diagnostics, nonlinear structural demand, and hidden-confounder sensitivity
before drawing operational conclusions. Three seeds do not establish superiority.

MIT licensed. Contributions should make identification assumptions and
evaluation information available to each policy explicit.
