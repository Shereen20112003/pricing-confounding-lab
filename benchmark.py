"""Offline pricing under confounding: predictive fit is not policy value."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np


def demand_mean(context, price):
    return np.maximum(90 + 24 * context - 3 * price, 0)


def historical_data(rng, n):
    if n < 3:
        raise ValueError("n must be at least three")
    context = rng.uniform(-1, 1, n)
    price = np.clip(12 + 5 * context + rng.normal(0, 1.3, n), 4, 24)
    demand = demand_mean(context, price) + rng.normal(0, 3, n)
    return context, price, demand


def design(context, price, adjusted):
    context, price = np.broadcast_arrays(context, price)
    columns = [np.ones(context.size), price.ravel()]
    if adjusted:
        columns.append(context.ravel())
    return np.column_stack(columns)


def fit(context, price, demand, adjusted=True):
    x = design(context, price, adjusted)
    y = np.asarray(demand, float)
    if y.shape != (len(x),) or len(y) <= x.shape[1]:
        raise ValueError(
            "need matching observations and positive residual degrees of freedom"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("training values must be finite")
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("pricing design is rank deficient")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    variance = np.sum((y - x @ beta) ** 2) / (len(x) - x.shape[1])
    covariance = variance * np.linalg.inv(x.T @ x)
    return beta, covariance


def policy(context, grid, beta, covariance, adjusted=True, radius=0):
    context, grid = np.asarray(context, float), np.asarray(grid, float)
    if (
        context.ndim != 1
        or grid.ndim != 1
        or len(grid) == 0
        or not np.isfinite(context).all()
        or not np.isfinite(grid).all()
        or np.any(grid <= 0)
        or not np.isfinite(radius)
        or radius < 0
    ):
        raise ValueError("invalid contexts, price grid or radius")
    x = design(context[:, None], grid[None, :], adjusted)
    mean = x @ beta
    uncertainty = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", x, covariance, x), 0))
    lower_demand = np.maximum(mean - radius * uncertainty, 0)
    revenue = lower_demand.reshape(len(context), len(grid)) * grid
    return grid[np.argmax(revenue, axis=1)]


def evaluate(context, prices, grid):
    revenue = prices * demand_mean(context, prices)
    optimal = np.max(
        grid[None, :] * demand_mean(context[:, None], grid[None, :]), axis=1
    )
    return {
        "expected_revenue": float(revenue.mean()),
        "oracle_regret": float((optimal - revenue).mean()),
        "mean_price": float(prices.mean()),
    }


def run(seed, n_train=500, n_test=400):
    if min(n_train, n_test) < 5:
        raise ValueError("split sizes must be at least five")
    streams = np.random.SeedSequence(seed).spawn(3)
    train = historical_data(np.random.default_rng(streams[0]), n_train)
    grid = np.arange(4.0, 24.01, 0.5)
    tests = {
        "iid": np.random.default_rng(streams[1]).uniform(-1, 1, n_test),
        "context_shift": np.random.default_rng(streams[2]).uniform(1, 2, n_test),
    }
    models = {"naive": fit(*train, adjusted=False), "adjusted": fit(*train)}
    results = {}
    for name in ("naive", "adjusted", "adjusted_pessimistic", "fixed", "oracle"):
        results[name] = {}
        for scenario, context in tests.items():
            if name == "fixed":
                prices = np.full(n_test, 15.0)
            elif name == "oracle":
                revenue = grid * demand_mean(context[:, None], grid)
                prices = grid[revenue.argmax(axis=1)]
            else:
                adjusted = name != "naive"
                beta, covariance = models["adjusted" if adjusted else "naive"]
                prices = policy(
                    context,
                    grid,
                    beta,
                    covariance,
                    adjusted,
                    2 if name.endswith("pessimistic") else 0,
                )
            results[name][scenario] = evaluate(context, prices, grid)
    return {
        "seed": seed,
        "coefficients": {k: v[0].tolist() for k, v in models.items()},
        "policies": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct and nonnegative")
    try:
        runs = [run(s, args.n_train, args.n_test) for s in args.seeds]
    except ValueError as exc:
        parser.error(str(exc))
    report = {
        "schema_version": 1,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "config": {
            "seeds": args.seeds,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "price_grid": np.arange(4.0, 24.01, 0.5).tolist(),
            "radius": 2,
        },
        "data": "synthetic; no real retail transactions",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
