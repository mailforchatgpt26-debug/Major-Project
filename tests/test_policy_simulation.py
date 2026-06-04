"""Unit tests for policy simulation sign alignment."""
import pytest

from src.policy.simulation_adjust import (
    align_policy_simulation,
    policy_expected_pct_impact,
)


@pytest.mark.parametrize(
    "feature,change,expected_sign",
    [
        ("gdp", -20, -1),
        ("gdp", 20, 1),
        ("sentiment", -30, -1),
        ("sentiment", 15, 1),
        ("tariff", 25, -1),
        ("tariff", -10, 1),
        ("population", -15, -1),
    ],
)
def test_expected_pct_sign(feature, change, expected_sign):
    pct = policy_expected_pct_impact(feature, change)
    assert pct != 0
    assert (pct > 0) == (expected_sign > 0)


def test_align_corrects_wrong_gnn_sign():
    pct, base, counter, delta = align_policy_simulation(
        gnn_pct_impact=8.5,
        feature="sentiment",
        change_percent=-20.0,
        baseline_usd=100.0,
        table_baseline_usd=200.0,
    )
    assert pct < 0
    assert base == 200.0
    assert counter < base
    assert delta < 0


def test_align_preserves_correct_gnn_sign():
    pct, base, counter, delta = align_policy_simulation(
        gnn_pct_impact=-6.0,
        feature="gdp",
        change_percent=-10.0,
        baseline_usd=50.0,
    )
    assert pct < 0
    assert counter < base
    assert delta < 0


def test_tariff_hike_reduces_exports():
    pct, _, counter, _ = align_policy_simulation(
        gnn_pct_impact=12.0,
        feature="tariff",
        change_percent=20.0,
        baseline_usd=80.0,
    )
    assert pct < 0
    assert counter < 80.0
