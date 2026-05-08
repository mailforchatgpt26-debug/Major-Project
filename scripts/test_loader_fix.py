"""
Quick test script to verify the Comtrade loader fix works.

Usage:
    python scripts/test_loader_fix.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import ComtradeLoader
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(
    name="test_loader",
    log_level="INFO",
    console=True
)


def test_loader():
    """Test the fixed Comtrade loader."""
    
    print("\n" + "="*60)
    print("🧪 TESTING FIXED COMTRADE LOADER")
    print("="*60 + "\n")
    
    try:
        # Initialize loader
        loader = ComtradeLoader()
        
        # Load data
        print("Loading Comtrade data...")
        df = loader.load()
        
        print("\n" + "="*60)
        print("📊 RESULTS")
        print("="*60)
        
        if len(df) == 0:
            print("\n❌ FAILED: Still getting 0 rows")
            print("\nPlease run diagnostic again:")
            print("  python scripts/diagnose_comtrade.py")
            return False
        
        print(f"\n✅ SUCCESS: Loaded {len(df):,} rows")
        
        # Show breakdown
        print(f"\n📦 Sector breakdown:")
        print(df['sector'].value_counts())
        
        print(f"\n🌍 Top source countries:")
        print(df['source_iso3'].value_counts().head(10))
        
        print(f"\n🎯 India exports:")
        india_exports = df[df['source_iso3'] == 'IND']
        print(f"  Total: {len(india_exports):,} rows")
        
        if len(india_exports) > 0:
            print(f"  Sectors: {india_exports['sector'].value_counts().to_dict()}")
        
        print("\n" + "="*60)
        print("✅ LOADER TEST PASSED!")
        print("="*60)
        print("\nYou can now run preprocessing:")
        print("  python scripts/preprocess_data.py")
        print("="*60 + "\n")
        
        return True
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Loader test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_loader()
    sys.exit(0 if success else 1)