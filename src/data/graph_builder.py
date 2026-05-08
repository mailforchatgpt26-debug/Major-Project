"""Build PyTorch Geometric graph data structures."""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from typing import Dict, List, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """Build graph structure for GNN."""
    
    def __init__(self):
        self.node_mapping = {}  # iso3 -> node_id
        self.reverse_mapping = {}  # node_id -> iso3
    
    def build_node_mapping(self, countries: List[str]) -> Dict[str, int]:
        """Create mapping from ISO3 to node IDs."""
        self.node_mapping = {country: idx for idx, country in enumerate(sorted(countries))}
        self.reverse_mapping = {idx: country for country, idx in self.node_mapping.items()}
        return self.node_mapping
    
    def build_edge_index(self, edges_df: pd.DataFrame) -> torch.Tensor:
        """Build edge_index tensor from edge dataframe."""
        source_nodes = edges_df['source_iso3'].map(self.node_mapping).values
        target_nodes = edges_df['target_iso3'].map(self.node_mapping).values
        
        edge_index = torch.tensor(
            np.stack([source_nodes, target_nodes]),
            dtype=torch.long
        )
        
        return edge_index
    
    def build_node_features(self, nodes_df: pd.DataFrame, feature_cols: List[str]) -> torch.Tensor:
        """Build node feature matrix."""
        # Ensure nodes are in correct order
        nodes_df = nodes_df.set_index('iso3').reindex(self.reverse_mapping.values())
        
        x = torch.tensor(
            nodes_df[feature_cols].values,
            dtype=torch.float32
        )
        
        return x
    
    def build_edge_features(self, edges_df: pd.DataFrame, feature_cols: List[str]) -> torch.Tensor:
        """Build edge feature matrix."""
        edge_attr = torch.tensor(
            edges_df[feature_cols].values,
            dtype=torch.float32
        )
        
        return edge_attr
    
    def build_graph(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        node_feature_cols: List[str],
        edge_feature_cols: List[str],
        target_col: str = 'trade_value_log'
    ) -> Data:
        """
        Build complete PyG Data object.
        
        Returns:
            PyG Data object with x, edge_index, edge_attr, y
        """
        # Build mappings
        countries = nodes_df['iso3'].unique().tolist()
        self.build_node_mapping(countries)
        
        # Build tensors
        x = self.build_node_features(nodes_df, node_feature_cols)
        edge_index = self.build_edge_index(edges_df)
        edge_attr = self.build_edge_features(edges_df, edge_feature_cols)
        
        # Target values (only for edges with labels)
        y = torch.tensor(
            edges_df[target_col].values,
            dtype=torch.float32
        ).unsqueeze(1)
        
        # Create mask for training (India -> partner edges only)
        train_mask = torch.tensor(
            (edges_df['source_iso3'] == 'IND').values,
            dtype=torch.bool
        )
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            train_mask=train_mask,
            num_nodes=len(countries)
        )
        
        logger.info(f"Built graph: {data}")
        return data