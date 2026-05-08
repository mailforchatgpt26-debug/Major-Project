"""
Complete GNN Training Pipeline - FIXED METRICS
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.loaders import GraphDataLoader, TemporalDataset
from src.models.gnn import TradeGNN
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GNNTrainer:
    """GNN Training Pipeline"""
    
    def __init__(self, data_dir="data/processed", model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Force CPU
        self.device = torch.device("cpu")
        logger.info("💻 Using CPU (MPS doesn't support GAT operations)")
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.best_metrics = {'r2': 0, 'rmse': float('inf'), 'mape': float('inf')}
        self.train_losses = []
        self.val_losses = []
        self.metrics = {}
    
    def prepare_data(self) -> Tuple[List, List, List]:
        """Load and prepare data"""
        logger.info("="*60)
        logger.info("📊 LOADING DATA")
        logger.info("="*60)
        
        try:
            loader = GraphDataLoader(self.data_dir)
            graphs = loader.create_temporal_graphs()
            
            if len(graphs) < 3:
                raise ValueError(f"Only {len(graphs)} graphs created. Need at least 3.")
            
            dataset = TemporalDataset(graphs)
            train, val, test = dataset.split(train=0.7, val=0.15)
            
            logger.info(f"✓ Train: {len(train)} graphs")
            logger.info(f"✓ Val:   {len(val)} graphs")
            logger.info(f"✓ Test:  {len(test)} graphs")
            
            self.num_node_features = graphs[0].x.shape[1]
            self.num_edge_features = graphs[0].edge_attr.shape[1]
            
            logger.info(f"✓ Node features: {self.num_node_features}")
            logger.info(f"✓ Edge features: {self.num_edge_features}")
            
            # Check label range
            all_labels = []
            for g in graphs:
                all_labels.extend(g.y.numpy())
            all_labels = np.array(all_labels)
            logger.info(f"✓ Label range: {all_labels.min():.2f} to {all_labels.max():.2f} (log scale)")
            logger.info(f"✓ Label mean: {all_labels.mean():.2f}, std: {all_labels.std():.2f}")
            
            logger.info("="*60)
            
            return train, val, test
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def init_model(self, hidden_dim=128, num_layers=3, dropout=0.3, heads=4):
        """Initialize model"""
        logger.info("="*60)
        logger.info("🧠 INITIALIZING MODEL")
        logger.info("="*60)
        
        self.model = TradeGNN(
            num_node_features=self.num_node_features,
            num_edge_features=self.num_edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✓ Model parameters: {params:,}")
        logger.info(f"✓ Hidden dim: {hidden_dim}")
        logger.info(f"✓ Layers: {num_layers}")
        logger.info(f"✓ Attention heads: {heads}")
        logger.info(f"✓ Dropout: {dropout}")
        
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=1e-5  # Reduced weight decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=False
        )
        
        logger.info("✓ Optimizer: Adam (lr=0.001)")
        logger.info("✓ Scheduler: ReduceLROnPlateau")
        logger.info("="*60)
    
    def train_epoch(self, graphs: List) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        successful_batches = 0
        
        for g in graphs:
            g = g.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                out = self.model(g.x, g.edge_index, g.edge_attr)
                loss = self.criterion(out, g.y)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss detected, skipping batch")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                successful_batches += 1
                
            except Exception as e:
                logger.warning(f"Training error: {e}")
                continue
        
        if successful_batches == 0:
            return float('inf')
        
        return total_loss / successful_batches
    
    @torch.no_grad()
    def validate(self, graphs: List) -> Tuple[float, Dict]:
        """Validate"""
        self.model.eval()
        total_loss = 0
        preds, labels = [], []
        successful_batches = 0
        
        for g in graphs:
            g = g.to(self.device)
            
            try:
                out = self.model(g.x, g.edge_index, g.edge_attr)
                loss = self.criterion(out, g.y)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    preds.extend(out.cpu().numpy())
                    labels.extend(g.y.cpu().numpy())
                    successful_batches += 1
                
            except Exception as e:
                logger.warning(f"Validation error: {e}")
                continue
        
        if successful_batches == 0 or len(preds) == 0:
            return float('inf'), {'r2': 0, 'rmse': float('inf'), 'mape': float('inf'), 'mse': float('inf')}
        
        preds = np.array(preds)
        labels = np.array(labels)
        
        # Clip extreme values for stability
        preds = np.clip(preds, -10, 30)  # Reasonable range for log(trade_value)
        
        # Metrics on log scale
        mse = mean_squared_error(labels, preds)
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)
        
        # Convert to original scale for interpretable MAPE
        preds_orig = np.expm1(preds)
        labels_orig = np.expm1(labels)
        
        # Clip to avoid extreme MAPE
        preds_orig = np.maximum(preds_orig, 1)
        labels_orig = np.maximum(labels_orig, 1)
        
        # MAPE with clipping
        abs_pct_error = np.abs((labels_orig - preds_orig) / labels_orig)
        abs_pct_error = np.clip(abs_pct_error, 0, 10)  # Cap at 1000%
        mape = np.mean(abs_pct_error) * 100
        
        return total_loss / successful_batches, {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
    
    def train(self, train_graphs, val_graphs, epochs=100, patience=20):
        """Training loop"""
        logger.info("="*60)
        logger.info("🚀 TRAINING")
        logger.info("="*60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Patience: {patience}")
        logger.info(f"Device: {self.device}")
        logger.info("="*60)
        
        patience_counter = 0
        best_state = None
        best_epoch = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_graphs)
            self.train_losses.append(train_loss)
            
            if train_loss == float('inf'):
                logger.error("Training failed - all batches errored")
                break
            
            val_loss, val_metrics = self.validate(val_graphs)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if True:
                logger.info(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"R²: {val_metrics['r2']:.4f} | "
                    f"MAE: {val_metrics['mae']:.4f}"
                )
            
            if val_loss < self.best_val_loss and val_loss != float('inf'):
                self.best_val_loss = val_loss
                self.best_metrics = val_metrics
                best_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"⏹  Early stopping at epoch {epoch+1}")
                    break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("="*60)
            logger.info("✓ TRAINING COMPLETE")
            logger.info(f"Best epoch: {best_epoch+1}")
            logger.info(f"Best val loss: {self.best_val_loss:.4f}")
            logger.info(f"Best R²: {self.best_metrics['r2']:.4f}")
            logger.info(f"Best MAE: {self.best_metrics['mae']:.4f}")
            logger.info("="*60)
        else:
            logger.error("Training failed - no valid model")
            raise RuntimeError("Training failed")
    
    def evaluate(self, test_graphs):
        """Evaluate on test set"""
        logger.info("="*60)
        logger.info("📊 TEST EVALUATION")
        logger.info("="*60)
        
        _, metrics = self.validate(test_graphs)
        
        logger.info(f"R² Score:  {metrics['r2']:.4f}")
        logger.info(f"RMSE:      {metrics['rmse']:.4f}")
        logger.info(f"MAE:       {metrics['mae']:.4f}")
        logger.info(f"MAPE:      {metrics['mape']:.2f}%")
        logger.info("="*60)
        
        self.metrics['test'] = metrics
        return metrics
    
    def save(self):
        """Save model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_metrics': self.best_metrics,
            'config': {
                'num_node_features': self.num_node_features,
                'num_edge_features': self.num_edge_features,
                'device': str(self.device)
            }
        }
        
        path = self.model_dir / f"gnn_{timestamp}.pt"
        torch.save(checkpoint, path)
        logger.info(f"💾 Saved: {path}")
        
        metadata = {
            'timestamp': timestamp,
            'test_metrics': self.metrics.get('test', {}),
            'best_val_loss': float(self.best_val_loss),
            'best_val_metrics': self.best_metrics
        }
        
        meta_path = self.model_dir / f"gnn_{timestamp}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return timestamp


def main():
    """Main training"""
    print("\n" + "="*60)
    print("🧠 GNN TRADE FLOW PREDICTION - TRAINING")
    print("💻 CPU Mode (Mac Compatible)")
    print("="*60)
    
    try:
        trainer = GNNTrainer()
        
        train, val, test = trainer.prepare_data()
        
        trainer.init_model(
            hidden_dim=128,
            num_layers=3,
            dropout=0.3,
            heads=4
        )
        
        trainer.train(train, val, epochs=100, patience=20)
        
        trainer.evaluate(test)
        
        timestamp = trainer.save()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print(f"Saved as: {timestamp}")
        print(f"Best R²: {trainer.best_metrics['r2']:.4f}")
        print(f"Best MAE: {trainer.best_metrics['mae']:.4f}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()