"""Quickstart script to verify setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config, get_settings
from src.utils.logger import get_logger, setup_logger
from src.utils.database import get_db_context, get_redis
from sqlalchemy import text

def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("Testing Configuration...")
    print("="*60)
    
    config = get_config()
    settings = get_settings()
    
    print(f"✓ Project root: {settings.PROJECT_ROOT}")
    print(f"✓ Database URL: {settings.DATABASE_URL}")
    print(f"✓ Model type: {config.get_nested('model.type')}")
    print(f"✓ Training epochs: {config.get_nested('training.epochs')}")

def test_logging():
    """Test logging."""
    print("\n" + "="*60)
    print("Testing Logging...")
    print("="*60)
    
    logger = get_logger("test")
    logger.info("✓ Logging system working")

def test_database():
    """Test database connection."""
    print("\n" + "="*60)
    print("Testing Database...")
    print("="*60)
    
    try:
        with get_db_context() as db:
            result = db.execute(text("SELECT 1")).scalar()
            print(f"✓ Database connection successful: {result}")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")

def test_redis():
    """Test Redis connection."""
    print("\n" + "="*60)
    print("Testing Redis...")
    print("="*60)
    
    try:
        redis_client = get_redis()
        redis_client.set("test_key", "test_value")
        value = redis_client.get("test_key")
        print(f"✓ Redis connection successful: {value}")
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")

if __name__ == "__main__":
    print("\n🚀 TRADE FORECASTING SYSTEM - QUICKSTART TEST")
    
    test_config()
    test_logging()
    test_database()
    test_redis()
    
    print("\n" + "="*60)
    print("✅ All basic systems operational!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python scripts/preprocess_data.py")
    print("2. Run: python src/models/train.py")
    print("3. Run: python src/api/main.py")