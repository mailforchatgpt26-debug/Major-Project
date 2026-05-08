"""
GNN Architecture - Updated for your feature dimensions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from src.models.causal_gnn import StructuralGravityFramework


class TradeGNN(nn.Module):
    """Graph Attention Network for Trade Prediction"""
    
    def __init__(
        self,
        num_node_features: int = 4,  # gdp_log, pop_log, exports, imports
        num_edge_features: int = 10,  # Updated for your features
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 4
    ):
        super(TradeGNN, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(
                num_node_features, 
                hidden_dim, 
                heads=heads, 
                concat=True, 
                dropout=dropout
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * heads, 
                    hidden_dim, 
                    heads=heads, 
                    concat=True, 
                    dropout=dropout
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Last layer (single head)
        self.convs.append(
            GATConv(
                hidden_dim * heads, 
                hidden_dim, 
                heads=1, 
                concat=False, 
                dropout=dropout
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Edge prediction MLP
        mlp_in = hidden_dim * 2 + num_edge_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """Forward pass"""
        # Ensure float32
        x = x.float()
        edge_attr = edge_attr.float()
        
        # Node learning through GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            
            # Batch norm (handle single sample case)
            if x.shape[0] > 1:
                x = bn(x)
            
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.elu(x)
        
        # Edge prediction
        row, col = edge_index
        edge_emb = torch.cat([x[row], x[col], edge_attr], dim=1)
        
        # MLP prediction
        pred = self.edge_mlp(edge_emb)
        
        return pred.squeeze(-1)


class GravityTradeGNN(nn.Module):
    """
    GAT backbone (TradeGNN) + Structural Gravity prior + learned gate.
    Gravity score and GNN residual are blended per-edge via a sigmoid gate.
    Equilibrium loss can be applied during training.
    """

    def __init__(
        self,
        num_node_features: int = 4,
        num_edge_features: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 4,
    ):
        super(GravityTradeGNN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        # --- Gravity prior (same framework as CausalTradeGNN) ---
        self.gravity_module = StructuralGravityFramework(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_sizes=[128, 64],
            dropout=dropout,
        )

        # --- GAT layers (identical to TradeGNN) ---
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GATConv(num_node_features, hidden_dim, heads=heads, concat=True, dropout=dropout)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        self.convs.append(
            GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # --- GNN residual edge MLP (same as TradeGNN) ---
        mlp_in = hidden_dim * 2 + num_edge_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --- Learned gate: blends gravity score with GNN residual ---
        gate_in = hidden_dim * 2 + num_edge_features + 1  # edge_emb + gravity_score
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr):
        x = x.float()
        edge_attr = edge_attr.float()

        # Gravity prior
        gravity_score = self.gravity_module(x, edge_index, edge_attr)

        # GAT layers
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(h, edge_index)
            if h.shape[0] > 1:
                h = bn(h)
            if i < len(self.convs) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(h)

        # GNN residual prediction
        row, col = edge_index
        edge_emb = torch.cat([h[row], h[col], edge_attr], dim=1)
        gnn_residual = self.edge_mlp(edge_emb).squeeze(-1)

        # Learned gate
        gate_input = torch.cat([h[row], h[col], edge_attr, gravity_score.unsqueeze(-1)], dim=1)
        gate = self.gate(gate_input).squeeze(-1)

        # Hybrid prediction: gate blends gravity baseline + GAT residual
        pred = gate * gravity_score + (1.0 - gate) * gnn_residual
        return pred

    def forward_with_attention(self, x, edge_index, edge_attr):
        """Forward pass that also returns last-layer GAT attention weights and gate values."""
        x = x.float()
        edge_attr = edge_attr.float()

        gravity_score = self.gravity_module(x, edge_index, edge_attr)

        h = x
        attn_info = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            result = conv(h, edge_index, return_attention_weights=True)
            h, attn_info = result[0], result[1]  # (edge_index_with_loops, alpha [E, heads])
            if h.shape[0] > 1:
                h = bn(h)
            if i < len(self.convs) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(h)

        row, col = edge_index
        edge_emb = torch.cat([h[row], h[col], edge_attr], dim=1)
        gnn_residual = self.edge_mlp(edge_emb).squeeze(-1)

        gate_input = torch.cat([h[row], h[col], edge_attr, gravity_score.unsqueeze(-1)], dim=1)
        gate = self.gate(gate_input).squeeze(-1)

        pred = gate * gravity_score + (1.0 - gate) * gnn_residual
        return pred, attn_info, gate

    def calculate_equilibrium_loss(self, pred_trade, edge_index, x):
        """Same budget-constraint equilibrium loss as CausalTradeGNN."""
        row, col = edge_index
        trade_usd = torch.exp(pred_trade)

        num_nodes = x.size(0)
        node_exports = torch.zeros(num_nodes, device=x.device)
        node_imports = torch.zeros(num_nodes, device=x.device)
        node_exports.scatter_add_(0, row, trade_usd)
        node_imports.scatter_add_(0, col, trade_usd)

        gdp_usd = torch.exp(x[:, 0])
        trade_to_gdp_ratio = (node_exports + node_imports) / (gdp_usd + 1e-6)
        return torch.relu(trade_to_gdp_ratio - 1.5).mean()