#!/usr/bin/env python3
"""
Train and compare baseline models vs hybrid (GAT + sentiment + structural gravity).

When **--pretrained-hybrid** is set, baselines default to the **same schedule as
`scripts/train_gravity_gnn.py`**: **200 epochs**, **patience 50**, Adam lr=1e-3,
hidden=128, layers=3, dropout=0.3, heads=4 (override with `--epochs` / `--patience`).

Metrics:
  - **log_*** : MSE, RMSE, MAE, R² on log1p USD (training target).
  - **usd_*** : same on expm1(log) ≈ USD scale (heavy-tailed; R² usually more modest).
  - **mape_pct, mpe_pct** : on USD scale (clipped for stability).

Usage:
  PYTHONPATH=. python scripts/compare_baseline_models.py --pretrained-hybrid
  PYTHONPATH=. python scripts/compare_baseline_models.py --pretrained-hybrid --epochs 40 --patience 12   # shorter dev run
  PYTHONPATH=. python scripts/compare_baseline_models.py --quick
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import GraphDataLoader, TemporalDataset  # noqa: E402
from src.models.gnn import GravityTradeGNN, TradeGCN, TradeGNN  # noqa: E402

# Matches scripts/train_gravity_gnn.py when comparing to pretrained hybrid.
GRAVITY_SCRIPT_EPOCHS = 200
GRAVITY_SCRIPT_PATIENCE = 50


def _mask_sentiment(edge_attr: torch.Tensor) -> torch.Tensor:
    out = edge_attr.clone()
    out[:, 0:2] = 0.0
    return out


def _subsample(graphs: List, every: int) -> List:
    if every <= 1:
        return graphs
    return graphs[::every]


def _metrics_dict(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """Both log-scale and USD-scale regression metrics."""
    y_true_log = np.asarray(y_true_log, dtype=np.float64).ravel()
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64).ravel()
    y_pred_log = np.clip(y_pred_log, -10.0, 30.0)

    mse_log = float(mean_squared_error(y_true_log, y_pred_log))
    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    r2_log = float(r2_score(y_true_log, y_pred_log))
    rmse_log = float(np.sqrt(mse_log))

    yt = np.expm1(np.clip(y_true_log, -10, 30))
    yp = np.expm1(np.clip(y_pred_log, -10, 30))
    # Avoid rare exp blowups dominating USD R² / MSE
    cap = max(float(np.percentile(yt, 99.9)) * 20.0, 1e6)
    yp = np.clip(yp, 0.0, cap)
    yt = np.maximum(yt, 1.0)
    yp = np.maximum(yp, 1.0)

    mse_usd = float(mean_squared_error(yt, yp))
    mae_usd = float(mean_absolute_error(yt, yp))
    rmse_usd = float(np.sqrt(mse_usd))
    try:
        r2_usd = float(r2_score(yt, yp))
    except ValueError:
        r2_usd = float("nan")

    pct_err = (yp - yt) / yt
    mape = float(np.mean(np.abs(np.clip(pct_err, -10.0, 10.0))) * 100.0)
    mpe = float(np.mean(np.clip(pct_err, -10.0, 10.0)) * 100.0)

    return {
        "log_mse": mse_log,
        "log_rmse": rmse_log,
        "log_mae": mae_log,
        "log_r2": r2_log,
        "usd_mse": mse_usd,
        "usd_rmse": rmse_usd,
        "usd_mae": mae_usd,
        "usd_r2": r2_usd,
        "mape_pct": mape,
        "mpe_pct": mpe,
    }


class LagLSTMRegressor(nn.Module):
    """Uses only trade lags (edge_attr indices 7,8,9) as a 3-step univariate series."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr):
        seq = edge_attr[:, 7:10].unsqueeze(-1).float()
        out, _ = self.lstm(seq)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class TabularEdgeMLP(nn.Module):
    """Concatenates endpoint node features and edge attributes (no graph conv)."""

    def __init__(self, num_node_features: int, num_edge_features: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        in_dim = num_node_features * 2 + num_edge_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        z = torch.cat([x[row], x[col], edge_attr.float()], dim=1)
        return self.net(z).squeeze(-1)


def _graphs_to_device(graphs: List, device: torch.device):
    return [g.to(device) for g in graphs]


@torch.no_grad()
def collect_preds(
    model: Optional[nn.Module],
    graphs: List,
    device: torch.device,
    edge_attr_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    india_node_id: Optional[int] = None,
    persistence_lag1: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """If persistence_lag1, model is ignored and pred = edge_attr[:, 7] (log lag-1)."""
    preds, labels = [], []
    for g in graphs:
        g = g.to(device)
        ei = g.edge_index
        y = g.y
        if india_node_id is not None:
            m = ei[0] == india_node_id
            if m.sum() == 0:
                continue
            ei = ei[:, m]
            y = y[m]
            ea = g.edge_attr[m]
        else:
            ea = g.edge_attr

        ea_use = edge_attr_fn(ea) if edge_attr_fn is not None else ea

        if persistence_lag1:
            out = ea_use[:, 7].float()
        else:
            assert model is not None
            model.eval()
            out = model(g.x, ei, ea_use)

        preds.extend(out.detach().cpu().numpy().tolist())
        labels.extend(y.detach().cpu().numpy().tolist())
    return np.array(labels), np.array(preds)


def train_one(
    model: nn.Module,
    train: List,
    val: List,
    test: List,
    device: torch.device,
    epochs: int,
    patience: int,
    edge_attr_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    use_equilibrium: bool,
    india_node_id: Optional[int],
) -> Dict[str, float]:
    model = model.to(device)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(3, patience // 3))

    train = _graphs_to_device(train, device)
    val = _graphs_to_device(val, device)
    test = _graphs_to_device(test, device)

    def forward_batch(g):
        ea = edge_attr_fn(g.edge_attr) if edge_attr_fn is not None else g.edge_attr
        ei, y = g.edge_index, g.y
        if india_node_id is not None:
            m = ei[0] == india_node_id
            if m.sum() == 0:
                return None, None, None
            ei, y, ea = ei[:, m], y[m], ea[m]
        out = model(g.x, ei, ea)
        return out, y, ei

    best_val = float("inf")
    best_state = None
    bad = 0

    for _epoch in range(epochs):
        model.train()
        val_losses = []
        for g in train:
            opt.zero_grad()
            out, y, ei = forward_batch(g)
            if out is None:
                continue
            loss = crit(out, y)
            if use_equilibrium and hasattr(model, "calculate_equilibrium_loss") and ei is not None:
                eq = model.calculate_equilibrium_loss(out, ei, g.x)
                loss = loss + 0.1 * eq
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            for g in val:
                out, y, _ei = forward_batch(g)
                if out is None:
                    continue
                v = crit(out, y)
                if not (torch.isnan(v) or torch.isinf(v)):
                    val_losses.append(v.item())
        val_m = float(np.mean(val_losses)) if val_losses else float("inf")
        sched.step(val_m)

        if val_m < best_val:
            best_val = val_m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = collect_preds(model, test, device, edge_attr_fn, india_node_id, False)
    return _metrics_dict(y_true, y_pred)


def _short_plot_label(name: str) -> str:
    n = name.replace("Baseline: persistence (ŷ = lag-1 log trade)", "Persistence")
    n = n.replace("GAT + sentiment + Gravity (hybrid, pretrained)", "Hybrid (PT)")
    n = n.replace("GAT + sentiment + Gravity (hybrid, trained)", "Hybrid (tr)")
    n = n.replace("GAT + sentiment + Gravity (hybrid)", "Hybrid")
    n = n.replace(" [pretrained checkpoint — not same train budget as rows above]", "")
    if len(n) > 28:
        n = n[:26] + "…"
    return n


def save_comparison_plots(rows: List[Dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [_short_plot_label(r["model"]) for r in rows]
    x = np.arange(len(labels))
    w = 0.35

    # --- Figure 1: R² log vs USD ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, [r["log_r2"] for r in rows], width=w, label="log R²", color="#3b82f6")
    ax.bar(x + w / 2, [r["usd_r2"] for r in rows], width=w, label="USD R²", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Model comparison — R² (log target vs USD level)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_r2.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: MAPE ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(labels, [r["mape_pct"] for r in rows], color="#a855f7")
    ax.set_xlabel("MAPE % (USD scale)")
    ax.set_title("Mean absolute percentage error")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_mape.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: RMSE log vs USD ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, [r["log_rmse"] for r in rows], width=w, label="log RMSE", color="#f97316")
    ax.bar(x + w / 2, [r["usd_rmse"] for r in rows], width=w, label="USD RMSE", color="#eab308")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE — log scale vs USD scale")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_rmse.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/processed")
    ap.add_argument("--out-dir", type=str, default="results")
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Baseline training epochs (default: 200 with --pretrained-hybrid, else 25; --quick forces 8)",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stop patience (default: 50 with --pretrained-hybrid, else 8; --quick forces 3)",
    )
    ap.add_argument("--quick", action="store_true", help="Fewer epochs + subsample graphs (smoke test only)")
    ap.add_argument("--graph-every", type=int, default=1, help="Use every Nth temporal graph")
    ap.add_argument(
        "--pretrained-hybrid",
        action="store_true",
        help="Evaluate hybrid from models/gravity_gnn_working.pt; train baselines with same hyperparams as gravity script.",
    )
    ap.add_argument(
        "--india-exports-only",
        action="store_true",
        help="Train/eval only edges with source = IND (matches dashboard export focus; usually harder).",
    )
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.quick:
        epochs, patience, every = 8, 3, 2
    else:
        every = args.graph_every
        if args.epochs is not None:
            epochs = args.epochs
        elif args.pretrained_hybrid:
            epochs = GRAVITY_SCRIPT_EPOCHS
        else:
            epochs = 25
        if args.patience is not None:
            patience = args.patience
        elif args.pretrained_hybrid:
            patience = GRAVITY_SCRIPT_PATIENCE
        else:
            patience = 8

    hd, nl, do, nh = args.hidden_dim, args.num_layers, args.dropout, args.heads

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = GraphDataLoader(Path(args.data_dir))
    graphs = loader.create_temporal_graphs()
    graphs = _subsample(graphs, every)
    dataset = TemporalDataset(graphs)
    train, val, test = dataset.split(train=0.7, val=0.15)
    nf = graphs[0].x.shape[1]
    ef = graphs[0].edge_attr.shape[1]

    india_id: Optional[int] = None
    if args.india_exports_only:
        india_id = loader.node_mapping.get("IND")
        if india_id is None:
            raise RuntimeError("IND not in node_mapping")

    rows: List[Dict] = []

    # --- Strong naive baseline (no NN): predict current log flow = lag-1 log flow ---
    y_t, y_p = collect_preds(None, test, device, None, india_id, persistence_lag1=True)
    rows.append({"model": "Baseline: persistence (ŷ = lag-1 log trade)", **_metrics_dict(y_t, y_p), "trained": False, "checkpoint": ""})

    specs: List[Tuple[str, nn.Module, Optional[Callable[[torch.Tensor], torch.Tensor]], bool]] = [
        ("LSTM (lags only)", LagLSTMRegressor(hidden=32), None, False),
        ("MLP (nodes + edges)", TabularEdgeMLP(nf, ef, hidden=hd, dropout=do), None, False),
        ("GCN", TradeGCN(nf, ef, hidden_dim=hd, num_layers=nl, dropout=do), None, False),
        ("GAT (no sentiment)", TradeGNN(nf, ef, hidden_dim=hd, num_layers=nl, dropout=do, heads=nh), _mask_sentiment, False),
        ("GAT + news sentiment", TradeGNN(nf, ef, hidden_dim=hd, num_layers=nl, dropout=do, heads=nh), None, False),
    ]

    for name, model, edge_fn, use_eq in specs:
        print(f"\n=== Training: {name} ===", flush=True)
        m = train_one(model, train, val, test, device, epochs, patience, edge_fn, use_eq, india_id)
        rows.append({"model": name, **m, "trained": True, "checkpoint": ""})

    hybrid_name = "GAT + sentiment + Gravity (hybrid, pretrained)"
    ckpt = ROOT / "models" / "gravity_gnn_working.pt"
    if args.pretrained_hybrid and ckpt.exists():
        print(f"\n=== Loading pretrained hybrid: {ckpt} ===", flush=True)
        model = GravityTradeGNN(nf, ef, hidden_dim=hd, num_layers=nl, dropout=do, heads=nh).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        sd = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        model.load_state_dict(sd, strict=False)
        test_d = _graphs_to_device(test, device)
        y_true, y_pred = collect_preds(model, test_d, device, None, india_id, False)
        rows.append(
            {
                "model": hybrid_name,
                **_metrics_dict(y_true, y_pred),
                "trained": False,
                "checkpoint": str(ckpt),
            }
        )
    else:
        print(f"\n=== Training: GAT + sentiment + Gravity (hybrid) ===", flush=True)
        model = GravityTradeGNN(nf, ef, hidden_dim=hd, num_layers=nl, dropout=do, heads=nh)
        m = train_one(model, train, val, test, device, epochs, patience, None, True, india_id)
        rows.append({"model": "GAT + sentiment + Gravity (hybrid, trained)", **m, "trained": True, "checkpoint": ""})

    fieldnames = [
        "model",
        "log_mse",
        "log_rmse",
        "log_mae",
        "log_r2",
        "usd_mse",
        "usd_rmse",
        "usd_mae",
        "usd_r2",
        "mape_pct",
        "mpe_pct",
        "trained",
        "checkpoint",
    ]
    csv_path = out_dir / "model_comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "epochs": epochs,
        "patience": patience,
        "graph_subsample_every": every,
        "quick": args.quick,
        "pretrained_hybrid": bool(args.pretrained_hybrid and ckpt.exists()),
        "india_exports_only": args.india_exports_only,
        "device": str(device),
        "n_graphs_total": len(graphs),
        "n_train_val_test": [len(train), len(val), len(test)],
        "hyperparams": {
            "hidden_dim": hd,
            "num_layers": nl,
            "dropout": do,
            "heads": nh,
            "optimizer": "Adam lr=1e-3 weight_decay=1e-5",
            "gravity_script_reference": "scripts/train_gravity_gnn.py",
        },
        "notes": [
            "log_* metrics are on log1p(trade) labels (model target).",
            "usd_* metrics are on expm1(log) trade USD; usd_r2 is usually lower than log_r2.",
            "With --pretrained-hybrid, baselines use the same epoch/patience budget as train_gravity_gnn.py unless overridden.",
        ],
        "results": rows,
    }
    json_path = out_dir / "model_comparison.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    save_comparison_plots(rows, out_dir)

    md_path = out_dir / "model_comparison.md"
    intro = (
        "**How to read:** `log_r2` is often high on log1p(trade); **`usd_r2`** and **MAPE** describe dollar-scale error. "
        "Figures: `model_comparison_r2.png`, `model_comparison_mape.png`, `model_comparison_rmse.png`."
    )
    if args.pretrained_hybrid and ckpt.exists():
        if not args.quick and args.epochs is None and args.patience is None:
            intro += (
                f" **Setup:** baselines trained **{epochs}** epochs / patience **{patience}** (matches `scripts/train_gravity_gnn.py`); "
                "hybrid evaluated from **pretrained** `models/gravity_gnn_working.pt`."
            )
        else:
            intro += (
                f" **Setup:** baselines trained **{epochs}** epochs / patience **{patience}**; "
                "hybrid from **pretrained** `models/gravity_gnn_working.pt`. "
                "(For full gravity-script budget omit `--epochs`/`--patience` and do not use `--quick` → 200 / 50.)"
            )
    lines = [
        "# Model comparison (held-out test months)",
        "",
        intro,
        "",
        f"- epochs={epochs}, patience={patience}, graph every={every}, device={device}",
        f"- hidden_dim={hd}, num_layers={nl}, dropout={do}, heads={nh}",
        f"- india_exports_only={args.india_exports_only}",
        f"- pretrained_hybrid={bool(args.pretrained_hybrid and ckpt.exists())}",
        "",
        "## A) Log-scale (training target)",
        "",
        "| Model | log-MSE | log-RMSE | log-MAE | log-R² |",
        "|-------|---------|----------|---------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['log_mse']:.6f} | {r['log_rmse']:.6f} | {r['log_mae']:.6f} | {r['log_r2']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## B) USD scale (expm1) — use for “realistic” error",
            "",
            "| Model | USD-MSE | USD-RMSE | USD-MAE | USD-R² | MAPE % | MPE % |",
            "|-------|---------|----------|---------|--------|--------|-------|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['usd_mse']:.2f} | {r['usd_rmse']:.2f} | {r['usd_mae']:.2f} | "
            f"{r['usd_r2']:.6f} | {r['mape_pct']:.2f} | {r['mpe_pct']:.2f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Saved:", csv_path)
    print("Saved:", json_path)
    print("Saved:", md_path)
    for png in ("model_comparison_r2.png", "model_comparison_mape.png", "model_comparison_rmse.png"):
        print("Saved:", out_dir / png)
    print("=" * 60)


if __name__ == "__main__":
    main()
