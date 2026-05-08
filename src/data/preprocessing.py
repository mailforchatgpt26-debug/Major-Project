"""
Complete preprocessing pipeline for trade forecasting data.
Merges all data sources, creates features, and prepares for GNN training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from src.utils.logger import get_logger
from src.utils.config import get_settings, get_config
from src.utils.helpers import (
    log1p_transform, normalize_sentiment, create_lag_features,
    create_rolling_features, save_dataframe, ensure_directory
)
from src.data.loaders_preprocessing import DataLoader

logger = get_logger(__name__)
settings = get_settings()
config = get_config()


class DataPreprocessor:
    """Main preprocessing pipeline."""
    
    def __init__(self):
        self.config = config.get_pipeline_config()
        self.feature_config = config.get_features_config()
        self.processed_path = settings.PROJECT_ROOT / settings.PROCESSED_DATA_PATH
        ensure_directory(self.processed_path)
        
        self.node_mapping = {}  # iso3 -> node_id
        self.reverse_mapping = {}  # node_id -> iso3
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data sources."""
        logger.info("Step 1: Loading all data sources...")
        loader = DataLoader()
        return loader.load_all()
    
    # def merge_trade_with_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    #     """
    #     Merge trade data with all features.
        
    #     Args:
    #         data: Dictionary of loaded dataframes
        
    #     Returns:
    #         Merged dataframe with all features
    #     """
    #     logger.info("Step 2: Merging trade data with features...")
        
    #     trade = data['comtrade'].copy()
        
    #     # CRITICAL FIX: Check if Comtrade data is empty
    #     if len(trade) == 0:
    #         logger.error("❌ CRITICAL: Comtrade data is empty!")
    #         logger.error("Possible causes:")
    #         logger.error("  1. HS codes don't match Pharmaceuticals/Textiles")
    #         logger.error("  2. Flow column filtering issue (check flowCode vs flowDesc)")
    #         logger.error("  3. ISO3 mapping removed all countries")
    #         logger.error("\nPlease check your raw Comtrade file:")
    #         logger.error("  • What HS codes are present?")
    #         logger.error("  • What values are in flowCode/flowDesc columns?")
    #         logger.error("  • Are there valid country names/codes?")
    #         raise ValueError("Comtrade data is empty after loading. Cannot proceed.")
        
    #     logger.info(f"Starting with {len(trade):,} trade records")
        
    #     # Add World Bank node features for SOURCE countries
    #     logger.info("Merging source country features...")
    #     wb = data['world_bank'].copy()
    #     wb_source = wb.rename(columns={
    #         'iso3': 'source_iso3',
    #         'gdp_usd': 'source_gdp_usd',
    #         'population': 'source_population',
    #         'inflation_rate': 'source_inflation'
    #     })
        
    #     trade = trade.merge(
    #         wb_source[['source_iso3', 'year', 'source_gdp_usd', 'source_population', 'source_inflation']],
    #         on=['source_iso3', 'year'],
    #         how='left'
    #     )
        
    #     # Add World Bank node features for TARGET countries
    #     logger.info("Merging target country features...")
    #     wb_target = wb.rename(columns={
    #         'iso3': 'target_iso3',
    #         'gdp_usd': 'target_gdp_usd',
    #         'population': 'target_population',
    #         'inflation_rate': 'target_inflation'
    #     })
        
    #     trade = trade.merge(
    #         wb_target[['target_iso3', 'year', 'target_gdp_usd', 'target_population', 'target_inflation']],
    #         on=['target_iso3', 'year'],
    #         how='left'
    #     )
        
    #     logger.info(f"After World Bank merge: {len(trade):,} records")
        
    #     # Add CEPII distance features
    #     logger.info("Merging CEPII distance features...")
    #     cepii = data['cepii'].copy()
    #     trade = trade.merge(
    #         cepii,
    #         on=['source_iso3', 'target_iso3'],
    #         how='left'
    #     )
        
    #     # Add RTA (FTA) binary flag
    #     logger.info("Merging RTA (FTA) flags...")
    #     rtas = data['rtas'].copy()
    #     rtas['fta_binary'] = 1
    #     trade = trade.merge(
    #         rtas[['source_iso3', 'target_iso3', 'fta_binary']],
    #         on=['source_iso3', 'target_iso3'],
    #         how='left'
    #     )
    #     trade['fta_binary'] = trade['fta_binary'].fillna(0).astype(int)
        
    #     # Add GDELT sentiment
    #     gdelt = data.get('gdelt')
    #     if gdelt is not None and not gdelt.empty:
    #         logger.info("Merging GDELT sentiment...")
    #         # Match sentiment: try both (source->target) and (target->source)
    #         gdelt_1 = gdelt.rename(columns={
    #             'country_1_iso3': 'source_iso3',
    #             'country_2_iso3': 'target_iso3'
    #         })
            
    #         # Check if required columns exist
    #         if all(col in gdelt_1.columns for col in ['source_iso3', 'target_iso3', 'avg_tone']):
    #             trade = trade.merge(
    #                 gdelt_1[['source_iso3', 'target_iso3', 'year', 'month', 'avg_tone']],
    #                 on=['source_iso3', 'target_iso3', 'year', 'month'],
    #                 how='left'
    #             )
    #         else:
    #             logger.warning(f"⚠️  GDELT columns mismatch. Available: {list(gdelt_1.columns)}")
    #             trade['avg_tone'] = 0.0
    #     else:
    #          logger.warning("⚠️  No GDELT data available, setting sentiment to neutral (0)")
    #          trade['avg_tone'] = 0.0
    #          trade['sentiment_norm'] = 0.5  # Neutral
    
    #     # If no sentiment found, default to 0
    #     if 'avg_tone' in trade.columns:
    #         trade['avg_tone'] = trade['avg_tone'].fillna(0)
    #     else:
    #         trade['avg_tone'] = 0.0
        
    #     logger.info(f"After all merges: {len(trade):,} records")
        
    #     return trade
    
    """
Updated preprocessing to use calculated bilateral sentiment
ADD THIS TO src/data/preprocessing.py (replace merge_trade_with_features method)
"""

    def merge_trade_with_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge trade data with all features INCLUDING CALCULATED SENTIMENT.
        
        Args:
            data: Dictionary of loaded dataframes
        
        Returns:
            Merged dataframe with all features
        """
        logger.info("Step 2: Merging trade data with features...")
        
        trade = data['comtrade'].copy()
        
        if len(trade) == 0:
            logger.error("❌ CRITICAL: Comtrade data is empty!")
            raise ValueError("Comtrade data is empty after loading. Cannot proceed.")
        
        logger.info(f"Starting with {len(trade):,} trade records")
        
        # Add World Bank node features for SOURCE countries
        logger.info("Merging source country features...")
        wb = data['world_bank'].copy()
        wb_source = wb.rename(columns={
            'iso3': 'source_iso3',
            'gdp_usd': 'source_gdp_usd',
            'population': 'source_population',
            'inflation_rate': 'source_inflation'
        })
        
        trade = trade.merge(
            wb_source[['source_iso3', 'year', 'source_gdp_usd', 'source_population', 'source_inflation']],
            on=['source_iso3', 'year'],
            how='left'
        )
        
        # Add World Bank node features for TARGET countries
        logger.info("Merging target country features...")
        wb_target = wb.rename(columns={
            'iso3': 'target_iso3',
            'gdp_usd': 'target_gdp_usd',
            'population': 'target_population',
            'inflation_rate': 'target_inflation'
        })
        
        trade = trade.merge(
            wb_target[['target_iso3', 'year', 'target_gdp_usd', 'target_population', 'target_inflation']],
            on=['target_iso3', 'year'],
            how='left'
        )
        
        logger.info(f"After World Bank merge: {len(trade):,} records")
        
        # Add CEPII distance features
        cepii = data.get('cepii')
        if cepii is not None and not cepii.empty:
            logger.info("Merging CEPII distance features...")
            trade = trade.merge(
                cepii,
                on=['source_iso3', 'target_iso3'],
                how='left'
            )
        else:
            logger.warning("⚠️  No CEPII data found. Skipping distance merge.")
            if 'distance_km' not in trade.columns:
                trade['distance_km'] = np.nan
        
        # Add RTA (FTA) binary flag
        rtas = data.get('rtas')
        if rtas is not None and not rtas.empty:
            logger.info("Merging RTA (FTA) flags...")
            rtas['fta_binary'] = 1
            trade = trade.merge(
                rtas[['source_iso3', 'target_iso3', 'fta_binary']],
                on=['source_iso3', 'target_iso3'],
                how='left'
            )
        else:
            logger.warning("⚠️  No RTA data found. Skipping FTA merge.")
            
        if 'fta_binary' not in trade.columns:
            trade['fta_binary'] = 0
        trade['fta_binary'] = trade['fta_binary'].fillna(0).astype(int)
        
        # ============================================================
        # CRITICAL: Load CALCULATED bilateral sentiment
        # ============================================================
        logger.info("🎯 Loading CALCULATED bilateral sentiment...")
        
        sentiment_path = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "sentiment" / "bilateral_sentiment.csv"
        
        if sentiment_path.exists():
            bilateral_sentiment = pd.read_csv(sentiment_path)
            logger.info(f"Loaded {len(bilateral_sentiment):,} bilateral sentiment scores")
            
            # Merge sentiment (try both directions for country pairs)
            # First try: exact match
            trade = trade.merge(
                bilateral_sentiment[[
                    'country_1_iso3', 'country_2_iso3', 
                    'sentiment_score', 'sentiment_positive', 'sentiment_negative',
                    'confidence', 'article_count', 'trade_relevance'
                ]].rename(columns={
                    'country_1_iso3': 'source_iso3',
                    'country_2_iso3': 'target_iso3'
                }),
                on=['source_iso3', 'target_iso3'],
                how='left',
                suffixes=('', '_calculated')
            )
            
            # Second try: reversed pairs (bilateral is symmetric)
            missing_sentiment = trade['sentiment_score'].isna()
            if missing_sentiment.sum() > 0:
                reverse_sentiment = bilateral_sentiment.rename(columns={
                    'country_1_iso3': 'target_iso3',
                    'country_2_iso3': 'source_iso3'
                })
                
                trade_missing = trade[missing_sentiment].copy()
                trade_missing = trade_missing.merge(
                    reverse_sentiment[[
                        'source_iso3', 'target_iso3',
                        'sentiment_score', 'sentiment_positive', 'sentiment_negative',
                        'confidence', 'article_count', 'trade_relevance'
                    ]],
                    on=['source_iso3', 'target_iso3'],
                    how='left',
                    suffixes=('', '_rev')
                )
                
                # Fill in missing values
                for col in ['sentiment_score', 'sentiment_positive', 'sentiment_negative', 
                        'confidence', 'article_count', 'trade_relevance']:
                    if col + '_rev' in trade_missing.columns:
                        trade.loc[missing_sentiment, col] = trade_missing[col + '_rev'].values
            
            # Check coverage
            has_sentiment = trade['sentiment_score'].notna()
            sentiment_coverage = has_sentiment.sum() / len(trade) * 100
            logger.info(f"  ✓ Sentiment coverage: {sentiment_coverage:.1f}%")
            logger.info(f"  ✓ Avg sentiment: {trade.loc[has_sentiment, 'sentiment_score'].mean():.3f}")
            logger.info(f"  ✓ Avg confidence: {trade.loc[has_sentiment, 'confidence'].mean():.3f}")
            
            # Fill missing sentiment with neutral (0.0)
            trade['sentiment_score'] = trade['sentiment_score'].fillna(0.0)
            trade['sentiment_positive'] = trade['sentiment_positive'].fillna(0.33)
            trade['sentiment_negative'] = trade['sentiment_negative'].fillna(0.33)
            trade['confidence'] = trade['confidence'].fillna(0.0)
            trade['article_count'] = trade['article_count'].fillna(0)
            trade['trade_relevance'] = trade['trade_relevance'].fillna(0.0)
            
            # Normalize sentiment_score to [0, 1] range for model
            trade['sentiment_norm'] = (trade['sentiment_score'] + 1.0) / 2.0  # -1 to 1 → 0 to 1
            
            # Also keep raw GDELT tone for comparison (optional)
            if 'avg_tone' not in trade.columns:
                trade['avg_tone'] = trade['sentiment_score'] * 10  # Scale back to GDELT range
            
        else:
            logger.warning(f"⚠️  Calculated sentiment file not found: {sentiment_path}")
            logger.warning("⚠️  Using neutral sentiment (0.0) for all pairs")
            logger.warning("⚠️  Run: python src/pipelines/sentiment_analyzer.py first!")
            
            trade['sentiment_score'] = 0.0
            trade['sentiment_positive'] = 0.33
            trade['sentiment_negative'] = 0.33
            trade['sentiment_norm'] = 0.5
            trade['confidence'] = 0.0
            trade['article_count'] = 0
            trade['trade_relevance'] = 0.0
            trade['avg_tone'] = 0.0
        
        logger.info(f"After all merges: {len(trade):,} records")
        
        return trade
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            df: Merged trade dataframe
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Step 3: Engineering features...")
        
        # Log transformations
        logger.info("Applying log transformations...")
        df['gdp_log_source'] = log1p_transform(df['source_gdp_usd'])
        df['gdp_log_target'] = log1p_transform(df['target_gdp_usd'])
        df['pop_log_source'] = log1p_transform(df['source_population'])
        df['pop_log_target'] = log1p_transform(df['target_population'])
        df['distance_log'] = np.log1p(df['distance_km'])
        
        # Target variable (log-transformed trade value)
        df['trade_value_log'] = log1p_transform(df['trade_value_usd'])
        
        # Normalize sentiment
        df['sentiment_norm'] = normalize_sentiment(df['avg_tone'])
        
        # Create derived features
        logger.info("Creating derived features...")
        
        # GDP ratio
        df['gdp_ratio'] = np.log1p(
            (df['source_gdp_usd'] + 1) / (df['target_gdp_usd'] + 1)
        )
        
        # Population ratio
        df['pop_ratio'] = np.log1p(
            (df['source_population'] + 1) / (df['target_population'] + 1)
        )
        
        # Economic size (combined GDP)
        df['combined_gdp_log'] = np.log1p(
            df['source_gdp_usd'] + df['target_gdp_usd']
        )
        
        # Create lag features (previous trade values)
        logger.info("Creating lag features...")
        # time_step ensures monthly ordering: lag-1 = previous month, not previous year
        df['time_step'] = df['year'].astype(int) * 12 + df['month'].astype(int)
        df = df.sort_values(['source_iso3', 'target_iso3', 'hs_code', 'time_step'])

        df = create_lag_features(
            df,
            group_cols=['source_iso3', 'target_iso3', 'hs_code'],
            value_col='trade_value_log',
            lags=[1, 2, 3],
            sort_col='time_step'
        )

        # Create rolling features
        logger.info("Creating rolling features...")
        df = create_rolling_features(
            df,
            group_cols=['source_iso3', 'target_iso3', 'hs_code'],
            value_col='trade_value_log',
            windows=[3, 6],
            sort_col='time_step'
        )
        
        # Handle missing values in lag features
        # Fill with 0 for first occurrences
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
        df[lag_cols] = df[lag_cols].fillna(0)
        
        logger.info(f"After feature engineering: {df.shape[1]} columns")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        logger.info("Step 4: Handling missing values...")
        
        # Critical features - drop rows if missing
        critical_features = [
            'source_iso3', 'target_iso3', 'year', 
            'trade_value_usd', 'trade_value_log'
        ]
        
        initial_len = len(df)
        df = df.dropna(subset=critical_features)
        logger.info(f"Dropped {initial_len - len(df):,} rows with missing critical features")
        
        # Numeric features - fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df.loc[:, col] = df[col].fillna(median_val)
                logger.info(f"Filled {col} missing values with median: {median_val:.2f}")
        
        # Boolean features - fill with False
        bool_cols = ['shared_language', 'contiguous', 'fta_binary']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        
        return df
    
    def create_node_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create node dataframe with country-level features.
        
        Args:
            df: Merged trade dataframe
        
        Returns:
            Node dataframe with one row per country-year
        """
        logger.info("Step 5: Creating node dataframe...")
        
        # Get unique countries from both source and target
        source_nodes = df[['source_iso3', 'year', 'source_gdp_usd', 'source_population', 
                          'source_inflation', 'gdp_log_source', 'pop_log_source']].copy()
        source_nodes = source_nodes.rename(columns={
            'source_iso3': 'iso3',
            'source_gdp_usd': 'gdp_usd',
            'source_population': 'population',
            'source_inflation': 'inflation_rate',
            'gdp_log_source': 'gdp_log',
            'pop_log_source': 'pop_log'
        })
        
        target_nodes = df[['target_iso3', 'year', 'target_gdp_usd', 'target_population',
                          'target_inflation', 'gdp_log_target', 'pop_log_target']].copy()
        target_nodes = target_nodes.rename(columns={
            'target_iso3': 'iso3',
            'target_gdp_usd': 'gdp_usd',
            'target_population': 'population',
            'target_inflation': 'inflation_rate',
            'gdp_log_target': 'gdp_log',
            'pop_log_target': 'pop_log'
        })
        
        # Combine and deduplicate
        nodes = pd.concat([source_nodes, target_nodes], ignore_index=True)
        nodes = nodes.drop_duplicates(subset=['iso3', 'year'])
        
        # Create node IDs
        unique_countries = sorted(nodes['iso3'].unique())
        self.node_mapping = {country: idx for idx, country in enumerate(unique_countries)}
        self.reverse_mapping = {idx: country for country, idx in self.node_mapping.items()}
        
        nodes['node_id'] = nodes['iso3'].map(self.node_mapping)
        
        # Handle missing values
        nodes = nodes.fillna(nodes.median(numeric_only=True))
        
        logger.info(f"Created {len(nodes):,} node records for {len(unique_countries)} countries")
        
        return nodes.sort_values(['year', 'node_id'])
    
    def create_edge_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create edge dataframe with bilateral features.
        
        Args:
            df: Merged trade dataframe
        
        Returns:
            Edge dataframe
        """
        logger.info("Step 6: Creating edge dataframe...")
        
        # Select edge features
        edge_features = [
            'source_iso3', 'target_iso3', 'year', 'month',
            'hs_code', 'sector',
            'distance_km', 'distance_log',
            'shared_language', 'contiguous', 'fta_binary',
            'avg_tone', 'sentiment_norm',
            'trade_value_usd', 'trade_value_log',
            'trade_value_log_lag_1', 'trade_value_log_lag_2', 'trade_value_log_lag_3',
            'trade_value_log_rolling_mean_3', 'trade_value_log_rolling_mean_6'
        ]
        
        edges = df[[col for col in edge_features if col in df.columns]].copy()
        
        # Add source and target node IDs
        edges['source_node_id'] = edges['source_iso3'].map(self.node_mapping)
        edges['target_node_id'] = edges['target_iso3'].map(self.node_mapping)
        
        # Filter out edges where node IDs couldn't be mapped
        edges = edges.dropna(subset=['source_node_id', 'target_node_id'])
        edges['source_node_id'] = edges['source_node_id'].astype(int)
        edges['target_node_id'] = edges['target_node_id'].astype(int)
        
        # Create train/val/test split marker
        # train=2011-2023, val=2024, test=2025 (model never sees 2025 during training)
        max_year = int(edges['year'].max())
        val_year  = max_year - 1
        edges['split'] = 'train'
        edges.loc[edges['year'] == val_year,  'split'] = 'val'
        edges.loc[edges['year'] == max_year,  'split'] = 'test'
        
        # CRITICAL: Create India-focused mask for EVALUATION/PREDICTION
        # During training: Use ALL edges (learn global patterns)
        # During evaluation: Focus on India→partner edges
        edges['is_india_export'] = (edges['source_iso3'] == 'IND').astype(bool)
        
        logger.info(f"Created {len(edges):,} edges")
        logger.info(f"Train: {(edges['split']=='train').sum():,} | "
                   f"Val: {(edges['split']=='val').sum():,} | "
                   f"Test: {(edges['split']=='test').sum():,}")
        logger.info(f"Total edges (all countries): {len(edges):,}")
        logger.info(f"India export edges (prediction target): {edges['is_india_export'].sum():,} "
                   f"({edges['is_india_export'].sum()/len(edges)*100:.1f}%)")
        
        return edges
    
    def save_processed_data(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        """Save processed data to files."""
        logger.info("Step 7: Saving processed data...")
        
        # Save nodes
        nodes_path = self.processed_path / "nodes.csv"
        save_dataframe(nodes, nodes_path)
        logger.info(f"Saved nodes to {nodes_path}")
        
        # Save edges
        edges_path = self.processed_path / "edges.csv"
        save_dataframe(edges, edges_path)
        logger.info(f"Saved edges to {edges_path}")
        
        # Save node mapping
        mapping_path = self.processed_path / "node_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump({
                'node_to_iso3': self.reverse_mapping,
                'iso3_to_node': self.node_mapping
            }, f, indent=2)
        logger.info(f"Saved node mapping to {mapping_path}")
        
        # Save metadata
        metadata = {
            'num_nodes': len(nodes['node_id'].unique()),
            'num_edges': len(edges),
            'num_countries': len(self.node_mapping),
            'years': sorted(edges['year'].unique().tolist()),
            'sectors': edges['sector'].unique().tolist(),
            'train_edges': int((edges['split']=='train').sum()),
            'val_edges': int((edges['split']=='val').sum()),
            'test_edges': int((edges['split']=='test').sum()),
            'india_export_edges': int(edges['is_india_export'].sum())
        }
        
        metadata_path = self.processed_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        return metadata
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run complete preprocessing pipeline.
        
        Returns:
            Tuple of (nodes_df, edges_df, metadata)
        """
        logger.info("="*60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Load data
        data = self.load_data()
        
        # Merge and engineer features
        merged = self.merge_trade_with_features(data)
        featured = self.engineer_features(merged)
        cleaned = self.handle_missing_values(featured)
        
        # Create node and edge dataframes
        nodes = self.create_node_dataframe(cleaned)
        edges = self.create_edge_dataframe(cleaned)
        
        # Save processed data
        metadata = self.save_processed_data(nodes, edges)
        
        logger.info("="*60)
        logger.info("PREPROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Nodes: {len(nodes):,} | Edges: {len(edges):,}")
        logger.info(f"Output directory: {self.processed_path}")
        
        return nodes, edges, metadata


if __name__ == "__main__":
    # Run preprocessing pipeline
    preprocessor = DataPreprocessor()
    nodes, edges, metadata = preprocessor.run()
    
    print("\n✅ Preprocessing pipeline completed successfully!")
    print("\nMetadata:")
    print(json.dumps(metadata, indent=2))
    
    print("\nSample nodes:")
    print(nodes.head())
    
    print("\nSample edges:")
    print(edges.head())