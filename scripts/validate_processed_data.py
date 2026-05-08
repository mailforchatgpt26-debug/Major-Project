"""
Validate processed data quality and completeness.

Usage:
    python scripts/validate_processed_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class DataValidator:
    """Validate processed data quality."""
    
    def __init__(self):
        self.processed_path = settings.PROJECT_ROOT / settings.PROCESSED_DATA_PATH
        self.errors = []
        self.warnings = []
    
    def check_files_exist(self) -> bool:
        """Check that all expected files exist."""
        print("\n📁 Checking files...")
        
        required_files = [
            'nodes.csv',
            'edges.csv',
            'node_mapping.json',
            'metadata.json'
        ]
        
        all_exist = True
        for filename in required_files:
            filepath = self.processed_path / filename
            exists = filepath.exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {filename}")
            
            if not exists:
                self.errors.append(f"Missing file: {filename}")
                all_exist = False
        
        return all_exist
    
    def validate_nodes(self, nodes: pd.DataFrame) -> bool:
        """Validate node dataframe."""
        print("\n🔍 Validating nodes.csv...")
        
        valid = True
        
        # Check required columns
        required_cols = ['iso3', 'node_id', 'year', 'gdp_log', 'pop_log']
        for col in required_cols:
            if col not in nodes.columns:
                self.errors.append(f"Nodes missing column: {col}")
                valid = False
        
        if not valid:
            return False
        
        # Check for missing values in critical columns
        for col in required_cols:
            missing = nodes[col].isna().sum()
            if missing > 0:
                self.errors.append(f"Nodes has {missing} missing values in {col}")
                valid = False
            else:
                print(f"  ✅ {col}: No missing values")
        
        # Check node_id uniqueness per year
        duplicates = nodes.groupby(['node_id', 'year']).size()
        if (duplicates > 1).any():
            self.errors.append("Duplicate node_id-year combinations found")
            valid = False
        else:
            print(f"  ✅ No duplicate node_id-year combinations")
        
        # Check ISO3 format
        invalid_iso3 = nodes[nodes['iso3'].str.len() != 3]
        if len(invalid_iso3) > 0:
            self.errors.append(f"Found {len(invalid_iso3)} invalid ISO3 codes")
            valid = False
        else:
            print(f"  ✅ All ISO3 codes valid (3 characters)")
        
        # Check for India
        if 'IND' not in nodes['iso3'].values:
            self.errors.append("India (IND) not found in nodes")
            valid = False
        else:
            india_node_id = nodes[nodes['iso3'] == 'IND']['node_id'].iloc[0]
            print(f"  ✅ India found (node_id: {india_node_id})")
        
        # Check value ranges
        if (nodes['gdp_log'] < 0).any():
            self.warnings.append("Some gdp_log values are negative")
        
        if (nodes['pop_log'] < 0).any():
            self.warnings.append("Some pop_log values are negative")
        
        print(f"  ℹ️  Total nodes: {len(nodes):,}")
        print(f"  ℹ️  Unique countries: {nodes['iso3'].nunique()}")
        print(f"  ℹ️  Year range: {nodes['year'].min()} - {nodes['year'].max()}")
        
        return valid
    
    def validate_edges(self, edges: pd.DataFrame, nodes: pd.DataFrame) -> bool:
        """Validate edge dataframe."""
        print("\n🔍 Validating edges.csv...")
        
        valid = True
        
        # Check required columns
        required_cols = [
            'source_iso3', 'target_iso3', 'source_node_id', 'target_node_id',
            'year', 'trade_value_log', 'split', 'is_india_export'
        ]
        
        for col in required_cols:
            if col not in edges.columns:
                self.errors.append(f"Edges missing column: {col}")
                valid = False
        
        if not valid:
            return False
        
        # Check for missing values in critical columns
        critical_cols = ['source_node_id', 'target_node_id', 'trade_value_log']
        for col in critical_cols:
            missing = edges[col].isna().sum()
            if missing > 0:
                self.errors.append(f"Edges has {missing} missing values in {col}")
                valid = False
            else:
                print(f"  ✅ {col}: No missing values")
        
        # Validate node IDs exist in nodes
        max_node_id = nodes['node_id'].max()
        invalid_source = edges[edges['source_node_id'] > max_node_id]
        invalid_target = edges[edges['target_node_id'] > max_node_id]
        
        if len(invalid_source) > 0 or len(invalid_target) > 0:
            self.errors.append(f"Found edges with invalid node IDs")
            valid = False
        else:
            print(f"  ✅ All node IDs valid")
        
        # Check split distribution
        split_counts = edges['split'].value_counts()
        print(f"  ℹ️  Split distribution:")
        for split, count in split_counts.items():
            pct = count / len(edges) * 100
            print(f"    - {split}: {count:,} ({pct:.1f}%)")
        
        if 'train' not in split_counts:
            self.errors.append("No training edges found")
            valid = False
        
        # Check India exports
        india_exports = edges['is_india_export'].sum()
        india_pct = india_exports / len(edges) * 100
        print(f"  ℹ️  India exports: {india_exports:,} ({india_pct:.1f}%)")
        
        if india_exports == 0:
            self.errors.append("No India export edges found")
            valid = False
        else:
            print(f"  ✅ India exports present")
        
        # Check for self-loops
        self_loops = edges[edges['source_node_id'] == edges['target_node_id']]
        if len(self_loops) > 0:
            self.warnings.append(f"Found {len(self_loops)} self-loop edges")
        else:
            print(f"  ✅ No self-loops")
        
        # Check target variable distribution
        trade_log_stats = edges['trade_value_log'].describe()
        print(f"  ℹ️  Trade value (log) stats:")
        print(f"    - Mean: {trade_log_stats['mean']:.2f}")
        print(f"    - Std: {trade_log_stats['std']:.2f}")
        print(f"    - Min: {trade_log_stats['min']:.2f}")
        print(f"    - Max: {trade_log_stats['max']:.2f}")
        
        # Check for infinite values
        inf_count = np.isinf(edges['trade_value_log']).sum()
        if inf_count > 0:
            self.errors.append(f"Found {inf_count} infinite values in trade_value_log")
            valid = False
        else:
            print(f"  ✅ No infinite values in target")
        
        print(f"  ℹ️  Total edges: {len(edges):,}")
        print(f"  ℹ️  Year range: {edges['year'].min()} - {edges['year'].max()}")
        
        return valid
    
    def validate_mapping(self, mapping: dict, nodes: pd.DataFrame) -> bool:
        """Validate node mapping."""
        print("\n🔍 Validating node_mapping.json...")
        
        valid = True
        
        if 'iso3_to_node' not in mapping or 'node_to_iso3' not in mapping:
            self.errors.append("Mapping missing required keys")
            return False
        
        iso3_to_node = mapping['iso3_to_node']
        node_to_iso3 = mapping['node_to_iso3']
        
        # Check bidirectional consistency
        for iso3, node_id in iso3_to_node.items():
            if str(node_id) not in node_to_iso3:
                self.errors.append(f"Inconsistent mapping for {iso3}")
                valid = False
        
        # Check all nodes have mapping
        unique_countries = nodes['iso3'].unique()
        for country in unique_countries:
            if country not in iso3_to_node:
                self.errors.append(f"Country {country} not in mapping")
                valid = False
        
        if valid:
            print(f"  ✅ Mapping consistent")
            print(f"  ℹ️  Mapped countries: {len(iso3_to_node)}")
        
        return valid
    
    def validate_metadata(self, metadata: dict) -> bool:
        """Validate metadata."""
        print("\n🔍 Validating metadata.json...")
        
        valid = True
        
        required_keys = [
            'num_nodes', 'num_edges', 'num_countries',
            'years', 'sectors', 'train_edges', 'val_edges', 'test_edges'
        ]
        
        for key in required_keys:
            if key not in metadata:
                self.errors.append(f"Metadata missing key: {key}")
                valid = False
            else:
                print(f"  ✅ {key}: {metadata[key]}")
        
        return valid
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("\n" + "="*60)
        print("🔍 DATA VALIDATION")
        print("="*60)
        
        # Check files exist
        if not self.check_files_exist():
            print("\n❌ File check failed. Run preprocessing first.")
            return False
        
        # Load data
        try:
            nodes = pd.read_csv(self.processed_path / 'nodes.csv')
            edges = pd.read_csv(self.processed_path / 'edges.csv')
            
            with open(self.processed_path / 'node_mapping.json') as f:
                mapping = json.load(f)
            
            with open(self.processed_path / 'metadata.json') as f:
                metadata = json.load(f)
        
        except Exception as e:
            self.errors.append(f"Failed to load files: {e}")
            return False
        
        # Run validations
        nodes_valid = self.validate_nodes(nodes)
        edges_valid = self.validate_edges(edges, nodes)
        mapping_valid = self.validate_mapping(mapping, nodes)
        metadata_valid = self.validate_metadata(metadata)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        
        all_valid = nodes_valid and edges_valid and mapping_valid and metadata_valid
        
        if all_valid:
            print("\n✅ ALL VALIDATIONS PASSED!")
            print("\n✨ Data is ready for model training!")
        else:
            print("\n❌ VALIDATION FAILED")
            print(f"\n🔴 Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        print("="*60 + "\n")
        
        return all_valid


def main():
    """Run validation."""
    validator = DataValidator()
    success = validator.run_validation()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)