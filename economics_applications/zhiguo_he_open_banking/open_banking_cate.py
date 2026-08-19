"""Open banking: heterogeneous effects on borrower outcomes.

Inspired by Zhiguo He's work on open banking, information portability,
competition, and information externalities. This synthetic application asks
who gains and loses from data portability using CausalML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from causalml.inference.meta import BaseXRegressor
from sklearn.ensemble import GradientBoostingRegressor


def simulate_open_banking(n: int = 20_000, seed: int = 21):
    rng = np.random.default_rng(seed)

    credit_quality = rng.normal(size=n)
    opacity = rng.beta(2.0, 3.0, n)
    incumbent_relationship = rng.beta(2.0, 2.0, n)
    fintech_access = rng.uniform(size=n)
    local_competition = rng.uniform(size=n)
    privacy_cost = rng.beta(2.0, 4.0, n)
    income = rng.lognormal(mean=10.5, sigma=0.45, size=n)

    # Adoption/data-sharing selection.
    index = (
        -0.4
        + 0.55 * fintech_access
        + 0.35 * opacity
        - 0.45 * privacy_cost
        - 0.25 * incumbent_relationship
        + 0.15 * credit_quality
    )
    p = 1 / (1 + np.exp(-index))
    treatment = rng.binomial(1, p)

    # Lower spread is better. Benefits from portability are larger for opaque
    # borrowers with fintech access, but strategic competition/inference can
    # attenuate gains in thin markets or for borrowers with strong incumbent ties.
    true_tau = (
        -35 * opacity * fintech_access
        - 12 * local_competition
        + 16 * incumbent_relationship
        + 10 * privacy_cost
        - 5 * credit_quality
    )

    baseline = (
        245
        - 40 * credit_quality
        + 60 * opacity
        - 16 * local_competition
        - 14 * incumbent_relationship
        - 0.0008 * income
    )
    spread = baseline + treatment * true_tau + rng.normal(0, 30, n)

    X = pd.DataFrame(
        {
            "credit_quality": credit_quality,
            "opacity": opacity,
            "incumbent_relationship": incumbent_relationship,
            "fintech_access": fintech_access,
            "local_competition": local_competition,
            "privacy_cost": privacy_cost,
            "log_income": np.log(income),
        }
    )
    return X, treatment, spread, true_tau


def estimate_open_banking_cate(seed: int = 21) -> pd.DataFrame:
    X, treatment, y, true_tau = simulate_open_banking(seed=seed)

    learner = BaseXRegressor(
        learner=GradientBoostingRegressor(
            n_estimators=180, max_depth=3, learning_rate=0.04, random_state=seed
        )
    )
    cate = learner.fit_predict(X=X.values, treatment=treatment, y=y).reshape(-1)

    out = X.copy()
    out["treatment"] = treatment
    out["true_cate_bps"] = true_tau
    out["estimated_cate_bps"] = cate
    out["opacity_quartile"] = pd.qcut(
        out["opacity"], 4, labels=["Q1 transparent", "Q2", "Q3", "Q4 opaque"]
    )
    return out


if __name__ == "__main__":
    result = estimate_open_banking_cate()
    summary = (
        result.groupby("opacity_quartile", observed=True)
        .agg(
            true_effect_bps=("true_cate_bps", "mean"),
            estimated_effect_bps=("estimated_cate_bps", "mean"),
            adoption_rate=("treatment", "mean"),
            n=("treatment", "size"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(
        "\nNegative effects mean lower borrowing spreads. The distribution, not "
        "only the average, is the object of interest."
    )
