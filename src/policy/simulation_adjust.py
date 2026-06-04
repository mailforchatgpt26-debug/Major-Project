"""
Sign-consistent adjustments for GNN policy counterfactuals.

The causal GNN is not guaranteed to move monotonically with every shock
(e.g. a tariff hike can still raise the residual head's score). For the
policy engine we blend the model's magnitude with gravity-style elasticities
and enforce the economically expected sign of the change.
"""
from __future__ import annotations

import numpy as np


def policy_expected_pct_impact(feature: str, change_percent: float) -> float:
    """
    Expected bilateral export change (%) for a given shock.

    Signs follow standard trade intuition:
      - Partner GDP / sentiment up  -> exports up
      - Trade barriers up (tariff +) -> exports down
    """
    if change_percent == 0:
        return 0.0

    f = feature.lower()
    if "tariff" in f or "trade_cost" in f:
        # Positive change_percent = barriers rise -> trade falls
        return -0.9 * change_percent
    if "sentiment" in f:
        return 0.4 * change_percent
    if "gdp" in f:
        return 1.05 * change_percent
    if "pop" in f or "population" in f:
        return 0.35 * change_percent
    return 0.5 * change_percent


def align_policy_simulation(
    gnn_pct_impact: float,
    feature: str,
    change_percent: float,
    baseline_usd: float,
    *,
    table_baseline_usd: float | None = None,
    gnn_blend: float = 0.25,
) -> tuple[float, float, float, float]:
    """
    Return (pct_impact, baseline_usd, counterfactual_usd, delta_usd).

    Uses the predictions-table baseline when provided so API numbers match the UI.
    """
    expected = policy_expected_pct_impact(feature, change_percent)
    base = float(table_baseline_usd) if table_baseline_usd and table_baseline_usd > 0 else float(baseline_usd)

    if change_percent == 0 or base <= 0:
        return 0.0, base, base, 0.0

    sign_ok = gnn_pct_impact == 0 or np.sign(gnn_pct_impact) == np.sign(expected)
    if sign_ok:
        pct = float(gnn_blend * gnn_pct_impact + (1.0 - gnn_blend) * expected)
    else:
        gnn_mag = min(abs(gnn_pct_impact), 25.0)
        pct = float(np.sign(expected) * max(abs(expected), 0.35 * gnn_mag))

    counter = base * (1.0 + pct / 100.0)
    delta = counter - base
    return pct, base, counter, delta
