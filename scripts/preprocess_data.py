"""
Main preprocessing script - Run this to process all data.

Usage:
    python scripts/preprocess_data.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import DataPreprocessor
from src.utils.logger import get_logger, setup_logger
from src.utils.config import get_config
import time

# Setup logging
logger = setup_logger(
    name="preprocessing",
    log_level="INFO",
    log_file="preprocessing.log",
    console=True
)


def main():
    """Run complete preprocessing pipeline."""
    
    logger.info("🚀 Starting data preprocessing pipeline...")
    start_time = time.time()
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Run pipeline
        nodes, edges, metadata = preprocessor.run()
        
        # Print summary
        duration = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"\nProcessed Data Summary:")
        print(f"  • Countries: {metadata['num_countries']}")
        print(f"  • Node records: {metadata['num_nodes']}")
        print(f"  • Edge records: {metadata['num_edges']}")
        print(f"  • Years: {min(metadata['years'])} - {max(metadata['years'])}")
        print(f"  • Sectors: {', '.join(metadata['sectors'])}")
        print(f"\nData Split:")
        print(f"  • Train: {metadata['train_edges']:,} edges")
        print(f"  • Validation: {metadata['val_edges']:,} edges")
        print(f"  • Test: {metadata['test_edges']:,} edges")
        print(f"\nIndia Exports: {metadata['india_export_edges']:,} edges")
        print("\n" + "="*60)
        print("\n📁 Output files:")
        print(f"  • data/processed/nodes.csv")
        print(f"  • data/processed/edges.csv")
        print(f"  • data/processed/node_mapping.json")
        print(f"  • data/processed/metadata.json")
        print("\n🎯 Next step: Run model training")
        print("  python src/models/train.py")
        print("="*60 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print("Check logs/preprocessing.log for details")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)