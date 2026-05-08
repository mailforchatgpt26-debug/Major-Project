"""
Redis caching layer for FastAPI
Caches predictions, news, and alerts to improve performance
"""
import redis
import json
import pickle
from typing import Optional, Any
from functools import wraps
import hashlib
from datetime import timedelta
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RedisCache:
    def __init__(self):
        """Initialize Redis connection"""
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_db = int(os.getenv('REDIS_DB', 0))
        
        try:
            self.client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False,  # We'll handle encoding
                socket_connect_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"✓ Redis connected: {redis_host}:{redis_port}")
            self.enabled = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Running without cache.")
            self.client = None
            self.enabled = False
    
    def _make_key(self, prefix: str, **kwargs) -> str:
        """Create cache key from parameters"""
        # Sort kwargs for consistent keys
        params = "&".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_hash = hashlib.md5(params.encode()).hexdigest()[:8]
        return f"{prefix}:{key_hash}:{params}"
    
    def get(self, prefix: str, **kwargs) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None
        
        try:
            key = self._make_key(prefix, **kwargs)
            value = self.client.get(key)
            
            if value:
                logger.debug(f"Cache HIT: {key}")
                return pickle.loads(value)
            else:
                logger.debug(f"Cache MISS: {key}")
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, value: Any, prefix: str, ttl: int = 900, **kwargs):
        """Set value in cache with TTL (default 15 minutes)"""
        if not self.enabled:
            return
        
        try:
            key = self._make_key(prefix, **kwargs)
            serialized = pickle.dumps(value)
            self.client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, prefix: str, **kwargs):
        """Delete specific cache entry"""
        if not self.enabled:
            return
        
        try:
            key = self._make_key(prefix, **kwargs)
            self.client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        if not self.enabled:
            return
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cache CLEAR: {len(keys)} keys matching '{pattern}'")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def flush_all(self):
        """Clear entire cache (use with caution!)"""
        if not self.enabled:
            return
        
        try:
            self.client.flushdb()
            logger.warning("Cache FLUSHED: All data cleared")
        except Exception as e:
            logger.error(f"Cache flush error: {e}")


# Global cache instance
cache = RedisCache()


# Decorator for caching endpoints
def cached(prefix: str, ttl: int = 900):
    """
    Decorator to cache endpoint results
    
    Usage:
        @cached(prefix="predictions", ttl=600)
        async def get_predictions(sector, month):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_result = cache.get(prefix, **kwargs)
            if cached_result is not None:
                logger.info(f"Returning cached result for {prefix}")
                return cached_result
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(result, prefix, ttl=ttl, **kwargs)
            
            return result
        return wrapper
    return decorator