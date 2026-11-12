import json
import subprocess
import sys

import numpy as np
import pytest

from benchmark import demand_mean, evaluate, fit, historical_data, policy, run


def test_recover_known_coefficients():
    rng = np.random.default_rng(42)
    context, price = rng.uniform(-1, 1, 100), rng.uniform(5, 15, 100)
    beta, covariance = fit(context, price, demand_mean(context, price))
    np.testing.assert_allclose(beta, [90, -3, 24], atol=1e-10)
    np.testing.assert_allclose(covariance, 0, atol=1e-20)


def test_oracle_price():
    context, grid = np.array([0.0]), np.arange(4.0, 24.01, 0.5)
    q = policy(context, grid, np.array([90.0, -3.0, 24.0]), np.zeros((3, 3)))
    assert q == pytest.approx([15])
    assert evaluate(context, q, grid)["oracle_regret"] == 0


def test_confounding_reverses_slope():
    data = historical_data(np.random.default_rng(1), 2000)
    naive, _ = fit(*data, adjusted=False)
    adjusted, _ = fit(*data)
    assert naive[1] > 0
    assert adjusted[1] == pytest.approx(-3, abs=0.2)


def test_invalid_rank_and_sizes():
    with pytest.raises(ValueError, match="rank"):
        fit(np.ones(10), np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        fit(np.arange(3), np.arange(3), np.ones(3))


def test_seeded_regret_and_policies():
    result = run(1, 50, 40)
    assert result == run(1, 50, 40)
    for name, scenarios in result["policies"].items():
        for metrics in scenarios.values():
            assert metrics["oracle_regret"] >= -1e-9
            assert 4 <= metrics["mean_price"] <= 24
            if name == "oracle":
                assert metrics["oracle_regret"] == 0


def test_cli(tmp_path):
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "benchmark.py",
            "--seeds",
            "0",
            "--n-train",
            "30",
            "--n-test",
            "20",
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert len(report["runs"][0]["policies"]) == 5
