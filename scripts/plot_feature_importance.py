"""
Generate feature-importance bar charts for the IEEE paper (gradient-based, same as /api/explainability).
Outputs:
  results/feature_importance_USA.png  — single top partner
  results/feature_importance_avg_top10.png — mean normalized importance, top-10 partners by reference actual
  results/feature_importance.json — raw values for captions/tables
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import GraphDataLoader
from src.models.gnn import GravityTradeGNN

OUT_DIR = PROJECT_ROOT / "results"
FEATURE_NAMES = [
    "GDP",
    "Population",
    "Trade History",
    "Sentiment",
    "Distance",
    "Trade Agreement",
    "Shared Language",
]


def _load_gov_reference() -> dict[str, float]:
    text = (PROJECT_ROOT / "src/api/main.py").read_text(encoding="utf-8")
    gov: dict[str, float] = {}
    in_dict = False
    for line in text.splitlines():
        if "GOVT_PHARMA_EXPORT_ACTUAL_2025_USD_M" in line:
            in_dict = True
            continue
        if in_dict:
            m = re.match(r'\s+"([A-Z]{3})":\s*([\d.]+)', line)
            if m:
                gov[m.group(1)] = float(m.group(2))
            if line.strip() == "}":
                break
    return gov


def _importance_for_partner(
    model: GravityTradeGNN,
    g,
    loader: GraphDataLoader,
    partner: str,
) -> list[tuple[str, float]]:
    india_id = loader.node_mapping["IND"]
    partner_id = loader.node_mapping[partner]
    ei = g.edge_index
    ea = g.edge_attr.clone()

    row, col = ei
    exp_candidates = ((row == india_id) & (col == partner_id)).nonzero(as_tuple=True)[0]
    if len(exp_candidates) == 0:
        return []
    edge_idx = int(exp_candidates[0])

    model.eval()
    x_req = g.x.clone().requires_grad_(True)
    ea_req = ea.clone().requires_grad_(True)
    pred = model(x_req, ei, ea_req)
    model.zero_grad(set_to_none=True)
    pred[edge_idx].backward()

    ng = x_req.grad[partner_id].abs().detach()
    eg = ea_req.grad[edge_idx].abs().detach()

    raw = [
        ("GDP", float(ng[0])),
        ("Population", float(ng[1])),
        ("Trade History", float(eg[7])),
        ("Sentiment", float(eg[0])),
        ("Distance", float(eg[2])),
        ("Trade Agreement", float(eg[5])),
        ("Shared Language", float(eg[3])),
    ]
    max_imp = max(v for _, v in raw) or 1.0
    return [(name, v / max_imp) for name, v in raw]


def _plot_horizontal(
    items: list[tuple[str, float]],
    title: str,
    out_path: Path,
) -> None:
    items = sorted(items, key=lambda x: x[1])
    names = [x[0] for x in items]
    vals = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(names)))
    ax.barh(names, vals, color=colors, edgecolor="#1e3a5f", linewidth=0.4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Normalized importance (0–1)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 0.98), i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = PROJECT_ROOT / "models/gravity_gnn_working.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    loader = GraphDataLoader(PROJECT_ROOT / "data/processed")
    graphs = loader.create_temporal_graphs()
    g = None
    for graph in reversed(graphs):
        if str(getattr(graph, "time_key", "")).startswith("2024-12"):
            g = graph
            break
    if g is None:
        g = graphs[-1]

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = GravityTradeGNN(
        num_node_features=cfg["num_node_features"],
        num_edge_features=cfg["num_edge_features"],
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout", 0.3),
        heads=cfg.get("heads", 4),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    gov = _load_gov_reference()
    top_partners = sorted(gov.keys(), key=lambda p: -gov[p])[:10]

    # --- Single partner: USA (largest reference market) ---
    usa_items = _importance_for_partner(model, g, loader, "USA")
    if usa_items:
        _plot_horizontal(
            usa_items,
            "Feature importance — India → USA (pharmaceutical export)",
            OUT_DIR / "feature_importance_USA.png",
        )

    # --- Average over top-10 partners ---
    accum: dict[str, list[float]] = {n: [] for n in FEATURE_NAMES}
    per_partner: dict[str, list[dict[str, float]]] = {}
    for partner in top_partners:
        if partner not in loader.node_mapping:
            continue
        items = _importance_for_partner(model, g, loader, partner)
        if not items:
            continue
        per_partner[partner] = [{"feature": n, "importance": round(v, 4)} for n, v in items]
        for n, v in items:
            accum[n].append(v)

    avg_items = [
        (n, float(np.mean(accum[n]))) for n in FEATURE_NAMES if accum[n]
    ]
    # re-normalize mean profile to [0,1]
    m = max(v for _, v in avg_items) or 1.0
    avg_items = [(n, v / m) for n, v in avg_items]

    _plot_horizontal(
        avg_items,
        "Mean feature importance — top 10 partners (by reference trade volume)",
        OUT_DIR / "feature_importance_avg_top10.png",
    )

    payload = {
        "graph_time_key": str(getattr(g, "time_key", "")),
        "method": "Gradient magnitude w.r.t. node/edge inputs on Hybrid Gravity-GAT (same as /api/explainability)",
        "features": FEATURE_NAMES,
        "usa": [{"feature": n, "importance": round(v, 4)} for n, v in usa_items],
        "avg_top10": [{"feature": n, "importance": round(v, 4)} for n, v in avg_items],
        "per_partner_top10": per_partner,
    }
    out_json = OUT_DIR / "feature_importance.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_DIR / 'feature_importance_USA.png'}")
    print(f"Wrote {OUT_DIR / 'feature_importance_avg_top10.png'}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
