"""
Train the Causal & Equilibrium GNN Model
This script trains the new CausalTradeGNN model with equilibrium constraints.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.causal_gnn import CausalTradeGNN
from src.models.train import GNNTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CausalTrainer(GNNTrainer):
    """Extends GNNTrainer for Causal & Equilibrium Learning"""
    
    def init_model(self, hidden_dim=128, num_layers=3, dropout=0.3, heads=4):
        """Initialize CausalTradeGNN"""
        logger.info("🧠 INITIALIZING CAUSAL MODEL (Transformer Heads)")
        
        self.model = CausalTradeGNN(
            num_node_features=self.num_node_features,
            num_edge_features=self.num_edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads
        ).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        logger.info("✓ Causal Model ready with Equilibrium logic and Scheduler")

    def train_epoch(self, graphs):
        """Train with Equilibrium Loss"""
        self.model.train()
        total_mse = 0
        total_eq = 0
        successful = 0

        for g in graphs:
            g = g.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(g.x, g.edge_index, g.edge_attr)

            mse_loss = self.criterion(out, g.y)
            eq_loss = self.model.calculate_equilibrium_loss(out, g.edge_index, g.x)
            loss = mse_loss + 0.1 * eq_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss — skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_mse += mse_loss.item()
            total_eq += eq_loss.item()
            successful += 1

        if successful == 0:
            return float('inf')

        avg_mse = total_mse / successful
        avg_eq = total_eq / successful
        logger.info(f"  MSE={avg_mse:.4f}  EqLoss={avg_eq:.4f}  (weighted total={avg_mse + 0.1*avg_eq:.4f})")
        return avg_mse + 0.1 * avg_eq

def main():
    print("\n" + "="*60)
    print("🧠 CAUSAL GNN TRADE SIMULATION - TRAINING")
    print("="*60)
    
    trainer = CausalTrainer()
    train, val, test = trainer.prepare_data()
    
    trainer.init_model()
    trainer.train(train, val, epochs=100, patience=30)
    trainer.evaluate(test)
    
    # Save with a fixed name for the API to find reliably
    save_path = "models/causal_gnn_working.pt"
    torch.save({
        'model_state': trainer.model.state_dict(),
        'config': {
            'num_node_features': trainer.num_node_features,
            'num_edge_features': trainer.num_edge_features
        }
    }, save_path)
    
    print(f"\n✅ CAUSAL MODEL SAVED: {save_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
