"""
Train GravityTradeGNN — Anderson & van Wincoop structural gravity + GATConv backbone.

Loss = MSE + 0.1 * equilibrium_loss
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gnn import GravityTradeGNN
from src.models.train import GNNTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GravityGNNTrainer(GNNTrainer):
    """GNNTrainer extended for GravityTradeGNN (GAT + A&vW gravity)."""

    def init_model(self, hidden_dim=128, num_layers=3, dropout=0.3, heads=4):
        logger.info("=" * 60)
        logger.info("🌍 INITIALIZING GravityTradeGNN")
        logger.info("   Anderson & van Wincoop gravity + GATConv backbone")
        logger.info("=" * 60)

        self.model = GravityTradeGNN(
            num_node_features=self.num_node_features,
            num_edge_features=self.num_edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads,
        ).to(self.device)

        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✓ Parameters      : {params:,}")
        logger.info(f"✓ Hidden dim      : {hidden_dim}")
        logger.info(f"✓ GAT layers      : {num_layers}")
        logger.info(f"✓ Attention heads : {heads}")
        logger.info(f"✓ Dropout         : {dropout}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        logger.info("✓ Optimizer: Adam (lr=0.001, wd=1e-5)")
        logger.info("✓ Scheduler: ReduceLROnPlateau")
        logger.info("=" * 60)

    def train_epoch(self, graphs):
        self.model.train()
        total_mse = 0.0
        total_eq = 0.0
        successful = 0

        for g in graphs:
            g = g.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(g.x, g.edge_index, g.edge_attr)
            mse_loss = self.criterion(out, g.y)
            eq_loss = self.model.calculate_equilibrium_loss(out, g.edge_index, g.x)
            loss = mse_loss + 0.1 * eq_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss — skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_mse += mse_loss.item()
            total_eq += eq_loss.item()
            successful += 1

        if successful == 0:
            return float("inf")

        avg_mse = total_mse / successful
        avg_eq  = total_eq  / successful
        logger.info(f"  MSE={avg_mse:.4f}  EqLoss={avg_eq:.4f}  total={avg_mse + 0.1*avg_eq:.4f}")
        return avg_mse + 0.1 * avg_eq


def main():
    print("\n" + "=" * 60)
    print("🌍 GravityTradeGNN — Anderson & van Wincoop + GATConv")
    print("=" * 60)

    # Fixed seed for reproducibility
    torch.manual_seed(9)
    np.random.seed(9)

    trainer = GravityGNNTrainer()
    train, val, test = trainer.prepare_data()

    trainer.init_model(hidden_dim=128, num_layers=3, dropout=0.3, heads=4)
    trainer.train(train, val, epochs=200, patience=50)
    trainer.evaluate(test)

    save_path = "models/gravity_gnn_working.pt"
    torch.save(
        {
            "model_state": trainer.model.state_dict(),
            "config": {
                "num_node_features": trainer.num_node_features,
                "num_edge_features": trainer.num_edge_features,
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.3,
                "heads": 4,
            },
        },
        save_path,
    )

    print("\n" + "=" * 60)
    print("✅ GravityTradeGNN TRAINING COMPLETE")
    print(f"   Best val R²  : {trainer.best_metrics['r2']:.4f}")
    print(f"   Best val MAE : {trainer.best_metrics['mae']:.4f}")
    print(f"   Best val RMSE: {trainer.best_metrics['rmse']:.4f}")
    print(f"   Saved → {save_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
