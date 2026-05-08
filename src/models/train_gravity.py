"""
GravityTradeGNN Training Script — trains on corrected bilateral pharma data.
Saves to models/gravity_gnn_working.pt (overwrites previous checkpoint).
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.loaders import GraphDataLoader, TemporalDataset
from src.models.gnn import GravityTradeGNN
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_epoch(model, graphs, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss, n = 0.0, 0
    for g in graphs:
        g = g.to(device)
        optimizer.zero_grad()
        try:
            out = model(g.x, g.edge_index, g.edge_attr)
            loss = criterion(out, g.y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        except Exception as e:
            logger.warning(f"train batch error: {e}")
    return total_loss / n if n else float("inf")


@torch.no_grad()
def evaluate(model, graphs, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    preds_all, labels_all = [], []
    for g in graphs:
        g = g.to(device)
        try:
            out = model(g.x, g.edge_index, g.edge_attr)
            loss = criterion(out, g.y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss += loss.item()
            preds_all.extend(out.cpu().numpy())
            labels_all.extend(g.y.cpu().numpy())
            n += 1
        except Exception as e:
            logger.warning(f"eval batch error: {e}")

    if n == 0:
        return float("inf"), {"r2": 0.0, "rmse": float("inf"), "mae": float("inf")}

    p = np.clip(np.array(preds_all), -10, 30)
    l = np.array(labels_all)
    mse = mean_squared_error(l, p)
    metrics = {
        "r2":   float(r2_score(l, p)),
        "rmse": float(np.sqrt(mse)),
        "mae":  float(mean_absolute_error(l, p)),
        "mse":  float(mse),
    }
    return total_loss / n, metrics


def main():
    device = torch.device("cpu")
    data_dir = Path("data/processed")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("🧠 GravityTradeGNN — Training on corrected bilateral data")
    print("="*60)

    # Load data
    loader = GraphDataLoader(data_dir)
    graphs = loader.create_temporal_graphs()
    if len(graphs) < 3:
        raise RuntimeError(f"Only {len(graphs)} graphs — need at least 3.")

    dataset = TemporalDataset(graphs)
    train_g, val_g, test_g = dataset.split(train=0.7, val=0.15)

    num_node_features = graphs[0].x.shape[1]
    num_edge_features = graphs[0].edge_attr.shape[1]

    all_labels = np.concatenate([g.y.numpy() for g in graphs])
    print(f"Graphs: {len(graphs)}  train={len(train_g)} val={len(val_g)} test={len(test_g)}")
    print(f"Node features: {num_node_features}  Edge features: {num_edge_features}")
    print(f"Label range: {all_labels.min():.2f} – {all_labels.max():.2f}  mean={all_labels.mean():.2f}")

    # Model
    model = GravityTradeGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        heads=4,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # Training loop
    epochs, patience = 150, 25
    best_val_loss = float("inf")
    best_state = None
    best_metrics = {}
    no_improve = 0
    train_losses, val_losses = [], []

    print("\nEpoch  Train     Val      R²     LR")
    print("-"*50)
    for epoch in range(1, epochs + 1):
        tl = train_epoch(model, train_g, optimizer, criterion, device)
        vl, vm = evaluate(model, val_g, criterion, device)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = vm
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch:5d}  {tl:.4f}  {vl:.4f}  {vm['r2']:+.3f}  {lr:.5f}")

        if no_improve >= patience:
            print(f"\nEarly stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best and evaluate on test
    model.load_state_dict(best_state)
    _, test_metrics = evaluate(model, test_g, criterion, device)

    print("\n" + "="*60)
    print(f"Best Val  R²={best_metrics['r2']:.4f}  RMSE={best_metrics['rmse']:.4f}  MAE={best_metrics['mae']:.4f}")
    print(f"Test      R²={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f}  MAE={test_metrics['mae']:.4f}")
    print("="*60)

    # Save
    save_path = model_dir / "gravity_gnn_working.pt"
    torch.save({
        "model_state": best_state,
        "config": {
            "num_node_features": num_node_features,
            "num_edge_features": num_edge_features,
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.3,
            "heads": 4,
        },
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, save_path)
    print(f"\n✅ Saved → {save_path}")

    meta_path = model_dir / "gravity_gnn_working_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"best_val_metrics": best_metrics, "test_metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
