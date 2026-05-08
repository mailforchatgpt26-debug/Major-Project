"""
GravityTradeGNN — Anderson & van Wincoop Structural Gravity + GATConv
Run: /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /Users/midnight/Downloads/Major_project-main/scripts/run_gravity_gnn.py
"""
import os
import sys
import logging
from pathlib import Path

# Resolve project root from this file's location and chdir there
# so torch can create its debug dir without hitting macOS TCC restrictions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy loader/library logs
logging.getLogger("src.data.loaders").setLevel(logging.WARNING)
logging.getLogger("src.models.train").setLevel(logging.WARNING)
logging.getLogger("torch_geometric").setLevel(logging.WARNING)

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.gnn import GravityTradeGNN
from src.data.loaders import GraphDataLoader, TemporalDataset


# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN_DIM  = 128
NUM_LAYERS  = 3
HEADS       = 4
DROPOUT     = 0.3
LR          = 0.001
WEIGHT_DECAY= 1e-5
EPOCHS      = 100
PATIENCE    = 30
EQ_WEIGHT   = 0.1
SAVE_PATH   = str(PROJECT_ROOT / "models" / "gravity_gnn_working.pt")
# ─────────────────────────────────────────────────────────────────────────────


def print_header(text):
    print("\n" + "═" * 62)
    print(f"  {text}")
    print("═" * 62)

def print_section(text):
    print(f"\n── {text} " + "─" * (56 - len(text)))


def load_data():
    print_section("Loading Data")
    loader = GraphDataLoader(PROJECT_ROOT / "data" / "processed")
    graphs = loader.create_temporal_graphs()
    dataset = TemporalDataset(graphs)
    train, val, test = dataset.split(train=0.7, val=0.15)

    nf = graphs[0].x.shape[1]
    ef = graphs[0].edge_attr.shape[1]

    all_labels = np.concatenate([g.y.numpy() for g in graphs])
    print(f"  Graphs       : {len(graphs)}  (train={len(train)} | val={len(val)} | test={len(test)})")
    print(f"  Edges/graph  : {graphs[0].edge_index.shape[1]:,}")
    print(f"  Node features: {nf}   Edge features: {ef}")
    print(f"  Label range  : {all_labels.min():.2f} → {all_labels.max():.2f} (log scale)")
    return train, val, test, nf, ef


def build_model(nf, ef, device):
    print_section("Model Architecture")
    model = GravityTradeGNN(
        num_node_features=nf,
        num_edge_features=ef,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        heads=HEADS,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture : Anderson & van Wincoop Gravity + GATConv")
    print(f"  GAT layers   : {NUM_LAYERS}   Heads: {HEADS}   Hidden: {HIDDEN_DIM}")
    print(f"  Dropout      : {DROPOUT}   EqLoss weight: {EQ_WEIGHT}")
    print(f"  Parameters   : {params:,}")
    print(f"  Optimizer    : Adam  lr={LR}  weight_decay={WEIGHT_DECAY}")
    print(f"  Scheduler    : ReduceLROnPlateau  patience=10  factor=0.5")
    print(f"  Loss         : MSE + {EQ_WEIGHT} × EquilibriumLoss")
    return model


@torch.no_grad()
def evaluate(model, graphs, device, criterion):
    model.eval()
    preds, labels = [], []
    total_loss = 0.0
    n = 0
    for g in graphs:
        g = g.to(device)
        out = model(g.x, g.edge_index, g.edge_attr)
        loss = criterion(out, g.y)
        total_loss += loss.item()
        preds.extend(out.cpu().numpy())
        labels.extend(g.y.cpu().numpy())
        n += 1
    preds  = np.clip(np.array(preds),  -10, 30)
    labels = np.array(labels)
    mse  = mean_squared_error(labels, preds)
    mae  = mean_absolute_error(labels, preds)
    r2   = r2_score(labels, preds)
    rmse = np.sqrt(mse)
    # MAPE on original scale
    p_orig = np.maximum(np.expm1(preds),  1)
    l_orig = np.maximum(np.expm1(labels), 1)
    mape = np.mean(np.clip(np.abs((l_orig - p_orig) / l_orig), 0, 10)) * 100
    return total_loss / max(n, 1), {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def train(model, train_graphs, val_graphs, device, criterion, optimizer, scheduler):
    print_section("Training")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val R²':>7}  {'Val MAE':>8}  {'Val RMSE':>9}")
    print("  " + "-" * 56)

    best_val_loss = float("inf")
    best_metrics  = {}
    best_state    = None
    best_epoch    = 0
    patience_ctr  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_mse = total_eq = ok = 0
        for g in train_graphs:
            g = g.to(device)
            optimizer.zero_grad()
            out      = model(g.x, g.edge_index, g.edge_attr)
            mse_loss = criterion(out, g.y)
            eq_loss  = model.calculate_equilibrium_loss(out, g.edge_index, g.x)
            loss     = mse_loss + EQ_WEIGHT * eq_loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_mse += mse_loss.item()
            total_eq  += eq_loss.item()
            ok += 1

        if ok == 0:
            print(f"  {'ERR':>5}  all batches failed at epoch {epoch}")
            break

        train_loss = total_mse / ok + EQ_WEIGHT * (total_eq / ok)
        val_loss, vm = evaluate(model, val_graphs, device, criterion)
        scheduler.step(val_loss)

        marker = " ◀ best" if val_loss < best_val_loss else ""
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {vm['r2']:>7.4f}  {vm['mae']:>8.4f}  {vm['rmse']:>9.4f}{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = vm
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  ⏹  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    model.load_state_dict(best_state)
    return best_epoch, best_val_loss, best_metrics


def main():
    device = torch.device("cpu")

    print_header("GravityTradeGNN  —  A&vW Gravity + GATConv")
    print(f"  Device: {device}")

    train_g, val_g, test_g, nf, ef = load_data()
    model     = build_model(nf, ef, device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_epoch, best_val_loss, best_vm = train(model, train_g, val_g, device, criterion, optimizer, scheduler)

    # ── Validation summary ───────────────────────────────────────────
    print_section("Best Validation Results")
    print(f"  Best epoch   : {best_epoch}")
    print(f"  Val Loss     : {best_val_loss:.4f}")
    print(f"  Val R²       : {best_vm['r2']:.4f}")
    print(f"  Val RMSE     : {best_vm['rmse']:.4f}")
    print(f"  Val MAE      : {best_vm['mae']:.4f}")
    print(f"  Val MAPE     : {best_vm['mape']:.2f}%")

    # ── Test evaluation ──────────────────────────────────────────────
    print_section("Test Set Evaluation")
    _, tm = evaluate(model, test_g, device, criterion)
    print(f"  Test R²      : {tm['r2']:.4f}")
    print(f"  Test RMSE    : {tm['rmse']:.4f}")
    print(f"  Test MAE     : {tm['mae']:.4f}")
    print(f"  Test MAPE    : {tm['mape']:.2f}%")

    # ── Save ─────────────────────────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "num_node_features": nf,
            "num_edge_features": ef,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "heads": HEADS,
        },
        "best_val_metrics": best_vm,
        "test_metrics": tm,
    }, SAVE_PATH)

    print_section("Done")
    print(f"  Model saved  → {SAVE_PATH}")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
