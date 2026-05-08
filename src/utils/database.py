"""Database connection and session management."""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any, List
import redis
import json
from src.utils.config import get_settings, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
config = get_config()

# SQLAlchemy setup
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Get database session for dependency injection.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database session.
    
    Usage:
        with get_db_context() as db:
            result = db.execute("SELECT 1")
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        db.close()


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Execute a SQL query and return results as list of dicts.
    
    Args:
        query: SQL query string
        params: Query parameters
    
    Returns:
        List of dictionaries with query results
    """
    with get_db_context() as db:
        result = db.execute(text(query), params or {})
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]


def execute_insert(table: str, data: Dict[str, Any]) -> int:
    """
    Insert a single row and return the ID.
    
    Args:
        table: Table name
        data: Dictionary of column: value pairs
    
    Returns:
        Inserted row ID
    """
    columns = ', '.join(data.keys())
    placeholders = ', '.join([f':{k}' for k in data.keys()])
    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"
    
    with get_db_context() as db:
        result = db.execute(text(query), data)
        return result.scalar()


def execute_batch_insert(table: str, data_list: List[Dict[str, Any]], batch_size: int = 1000):
    """
    Insert multiple rows in batches.
    
    Args:
        table: Table name
        data_list: List of dictionaries
        batch_size: Number of rows per batch
    """
    if not data_list:
        return
    
    columns = ', '.join(data_list[0].keys())
    placeholders = ', '.join([f':{k}' for k in data_list[0].keys()])
    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    
    with get_db_context() as db:
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            db.execute(text(query), batch)
            
            if (i + batch_size) % 10000 == 0:
                logger.info(f"Inserted {i + batch_size}/{len(data_list)} rows into {table}")


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = :table_name
        )
    """
    with get_db_context() as db:
        result = db.execute(text(query), {"table_name": table_name})
        return result.scalar()


def get_table_row_count(table_name: str) -> int:
    """Get number of rows in a table."""
    query = f"SELECT COUNT(*) FROM {table_name}"
    with get_db_context() as db:
        result = db.execute(text(query))
        return result.scalar()


# =====================================================
# Redis Cache Functions
# =====================================================

_redis_client = None


def get_redis() -> redis.Redis:
    """
    Get Redis connection (singleton pattern).
    
    Returns:
        Redis client instance
    """
    global _redis_client
    
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            _redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching will be disabled.")
            _redis_client = None
    
    return _redis_client


def cache_set(key: str, value: Any, ttl: int = 900):
    """
    Set a value in Redis cache.
    
    Args:
        key: Cache key
        value: Value to cache (will be JSON serialized)
        ttl: Time to live in seconds (default: 15 minutes)
    """
    try:
        redis_client = get_redis()
        if redis_client:
            serialized = json.dumps(value)
            redis_client.setex(key, ttl, serialized)
    except Exception as e:
        logger.warning(f"Cache set failed for key {key}: {e}")


def cache_get(key: str) -> Optional[Any]:
    """
    Get a value from Redis cache.
    
    Args:
        key: Cache key
    
    Returns:
        Cached value or None if not found
    """
    try:
        redis_client = get_redis()
        if redis_client:
            value = redis_client.get(key)
            if value:
                return json.loads(value)
    except Exception as e:
        logger.warning(f"Cache get failed for key {key}: {e}")
    
    return None


def cache_delete(key: str):
    """Delete a key from Redis cache."""
    try:
        redis_client = get_redis()
        if redis_client:
            redis_client.delete(key)
    except Exception as e:
        logger.warning(f"Cache delete failed for key {key}: {e}")


def cache_clear_pattern(pattern: str):
    """
    Clear all cache keys matching a pattern.
    
    Args:
        pattern: Redis key pattern (e.g., "predictions:*")
    """
    try:
        redis_client = get_redis()
        if redis_client:
            keys = redis_client.keys(pattern)
            if keys:
                redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache keys matching '{pattern}'")
    except Exception as e:
        logger.warning(f"Cache clear pattern failed: {e}")


# =====================================================
# Database Utilities
# =====================================================

def get_country_id_mapping() -> Dict[str, int]:
    """
    Get mapping of ISO3 codes to country IDs.
    
    Returns:
        Dictionary {iso3: id}
    """
    cached = cache_get("country_id_mapping")
    if cached:
        return cached
    
    query = "SELECT iso3, id FROM countries"
    results = execute_query(query)
    mapping = {row['iso3']: row['id'] for row in results}
    
    cache_set("country_id_mapping", mapping, ttl=3600)  # Cache for 1 hour
    return mapping


def get_product_id_mapping() -> Dict[str, int]:
    """
    Get mapping of HS codes to product IDs.
    
    Returns:
        Dictionary {hs_code: id}
    """
    cached = cache_get("product_id_mapping")
    if cached:
        return cached
    
    query = "SELECT hs_code, id FROM products"
    results = execute_query(query)
    mapping = {row['hs_code']: row['id'] for row in results}
    
    cache_set("product_id_mapping", mapping, ttl=3600)
    return mapping


def ensure_country_exists(iso3: str, name: str, region: Optional[str] = None):
    """
    Ensure a country exists in the database, insert if not.
    
    Args:
        iso3: ISO3 country code
        name: Country name
        region: Geographic region (optional)
    """
    query = "SELECT id FROM countries WHERE iso3 = :iso3"
    result = execute_query(query, {"iso3": iso3})
    
    if not result:
        data = {"iso3": iso3, "name": name}
        if region:
            data["region"] = region
        execute_insert("countries", data)
        logger.info(f"Inserted country: {iso3} - {name}")
        
        # Clear cache
        cache_delete("country_id_mapping")


def ensure_product_exists(hs_code: str, description: str, sector: str):
    """
    Ensure a product exists in the database, insert if not.
    
    Args:
        hs_code: HS code
        description: Product description
        sector: Product sector (Pharmaceuticals/Textiles)
    """
    query = "SELECT id FROM products WHERE hs_code = :hs_code"
    result = execute_query(query, {"hs_code": hs_code})
    
    if not result:
        data = {"hs_code": hs_code, "description": description, "sector": sector}
        execute_insert("products", data)
        logger.info(f"Inserted product: {hs_code} - {description}")
        
        # Clear cache
        cache_delete("product_id_mapping")


def truncate_table(table_name: str):
    """
    Truncate a table (delete all rows).
    
    Args:
        table_name: Name of table to truncate
    """
    query = f"TRUNCATE TABLE {table_name} CASCADE"
    with get_db_context() as db:
        db.execute(text(query))
        logger.warning(f"Truncated table: {table_name}")


if __name__ == "__main__":
    # Test database functions
    print("Testing database utilities...")
    
    # Test connection
    with get_db_context() as db:
        result = db.execute(text("SELECT 1 as test")).scalar()
        print(f"✓ Database connection: {result}")
    
    # Test Redis
    redis_client = get_redis()
    if redis_client:
        cache_set("test_key", {"value": 123})
        cached = cache_get("test_key")
        print(f"✓ Redis cache: {cached}")
    
    # Test country mapping
    countries = get_country_id_mapping()
    print(f"✓ Country mapping: {len(countries)} countries")
    
    print("\n✅ All database utilities working!")