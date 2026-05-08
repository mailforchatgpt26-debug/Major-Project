"""
Counterfactual Simulation Engine
Handles interventions (what-if scenarios) using the Causal GNN model.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from src.models.causal_gnn import CausalTradeGNN
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TradeSimulator:
    """
    Simulation Engine for GNN Trade Counterfactuals
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.model = None
        self.config = None
        self.node_mapping = None
        
        self.load_model()
        
    def load_model(self):
        """Loads the GNN model and configuration"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.config = checkpoint['config']
            
            self.model = CausalTradeGNN(
                num_node_features=self.config['num_node_features'],
                num_edge_features=self.config['num_edge_features'],
                hidden_dim=128, # assuming 128 as default
                num_layers=3
            ).to(self.device)
            
            # Load model state (we'll need a specialized training to fill this)
            # For now: fall back to current weights or initialize
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'], strict=False)
                
            self.model.eval()
            logger.info(f"✓ GNN Simulator loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load simulator: {e}")
            raise
    
    def run_intervention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        intervention: Dict[str, any]
    ) -> torch.Tensor:
        """
        Runs a 'What-If' intervention on the trade graph.
        
        Intervention format:
        {
            'node_id': 5, # or ISO3
            'feature': 'gdp_log', 
            'change': -0.2 # -20%
        }
        """
        with torch.no_grad():
            x_altered = x.clone().to(self.device)
            edge_attr_altered = edge_attr.clone().to(self.device)
            
            # 1. APPLY NODE-LEVEL INTERVENTION (e.g. GDP drop)
            if 'node_id' in intervention:
                idx = int(intervention['node_id'])
                feat_idx = int(intervention.get('feature_idx', 0)) # 0: gdp, 1: pop
                
                if intervention.get('is_absolute', False):
                    x_altered[idx, feat_idx] = float(intervention['value'])
                else:
                    x_altered[idx, feat_idx] += float(intervention['change'])
            
            # 2. APPLY EDGE-LEVEL INTERVENTION (e.g. Tariff increase)
            if 'edge_pair' in intervention:
                src, tgt = intervention['edge_pair']
                # find the specific edge in the edge_index
                mask = (edge_index[0] == src) & (edge_index[1] == tgt)
                if mask.any():
                    # modify specific edge attribute (like sentiment or a mock 'tariff' feature)
                    edge_attr_altered[mask, -1] += float(intervention.get('tariff_change', 0))
            
            # 3. PROPAGATE THROUGH CAUSAL GNN
            predictions = self.model(x_altered, edge_index, edge_attr_altered)
            
            return predictions

    def compare_scenarios(
        self, 
        baseline_graph: any, 
        intervention: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Compares Baseline trade (Prediction) vs. Counterfactual trade (Intervention).
        Returns the impact 'Delta' for each country pair.
        """
        x = baseline_graph.x
        edge_index = baseline_graph.edge_index
        edge_attr = baseline_graph.edge_attr
        
        # Original Prediction
        with torch.no_grad():
            baseline_pred = self.model(x, edge_index, edge_attr).cpu().numpy()
            
        # Counterfactual Prediction
        counterfactual_pred = self.run_intervention(x, edge_index, edge_attr, intervention).cpu().numpy()
        
        # Calculate Delta
        delta = counterfactual_pred - baseline_pred
        
        # Calculate Percentage Impact on original scale
        base_orig = np.expm1(baseline_pred)
        cf_orig = np.expm1(counterfactual_pred)
        pct_impact = (cf_orig - base_orig) / (base_orig + 1e-6) * 100
        
        return {
            'baseline': baseline_pred.tolist(),
            'counterfactual': counterfactual_pred.tolist(),
            'delta': delta.tolist(),
            'pct_impact': pct_impact.tolist(),
            'global_impact': float(np.mean(pct_impact))
        }
