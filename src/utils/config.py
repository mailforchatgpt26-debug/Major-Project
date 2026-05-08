"""
Configuration management for the trade forecasting system.
Loads environment variables and YAML configs.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # Database
    DATABASE_URL: str = Field(default="postgresql://postgres:postgres@localhost:5432/trade_forecasting")
    ASYNC_DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/trade_forecasting")
    
    # Redis
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    
    # Google Cloud
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(default=None)
    GCP_PROJECT_ID: Optional[str] = Field(default=None)
    
    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_RELOAD: bool = Field(default=True)
    CORS_ORIGINS: list = Field(default=["http://localhost:3000", "http://localhost:3001"])
    
    # Paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    RAW_DATA_PATH: str = Field(default="data/raw")
    PROCESSED_DATA_PATH: str = Field(default="data/processed")
    MODEL_PATH: str = Field(default="models/saved_models")
    LOG_PATH: str = Field(default="logs")
    
    # Model
    MODEL_VERSION: str = Field(default="v1.0")
    DEVICE: str = Field(default="cuda")
    
    # Training
    BATCH_SIZE: int = Field(default=32)
    LEARNING_RATE: float = Field(default=0.001)
    EPOCHS: int = Field(default=200)
    EARLY_STOPPING_PATIENCE: int = Field(default=20)
    
    # GDELT
    GDELT_FETCH_INTERVAL_MINUTES: int = Field(default=15)
    GDELT_LOOKBACK_DAYS: int = Field(default=7)
    
    # Alerts
    ALERT_SENTIMENT_THRESHOLD: float = Field(default=2.0)
    ALERT_PREDICTION_DROP_THRESHOLD: float = Field(default=0.15)
    ALERT_PREDICTION_SPIKE_THRESHOLD: float = Field(default=0.20)
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="logs/app.log")
    
    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production")
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"


class ConfigManager:
    """
    Manages loading and accessing configuration from YAML files and environment.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing config YAML files. Defaults to PROJECT_ROOT/configs
        """
        self.settings = Settings()
        
        if config_dir is None:
            config_dir = self.settings.PROJECT_ROOT / "configs"
        
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}
        
        # Load all YAML configs
        self._load_configs()
    
    def _load_configs(self):
        """Load all YAML configuration files."""
        if not self.config_dir.exists():
            print(f"Warning: Config directory {self.config_dir} does not exist")
            return
        
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            try:
                with open(config_file, 'r') as f:
                    self._configs[config_name] = yaml.safe_load(f)
                print(f"Loaded config: {config_name}")
            except Exception as e:
                print(f"Error loading {config_file}: {e}")
    
    def get(self, config_name: str, default: Any = None) -> Any:
        """
        Get configuration by name.
        
        Args:
            config_name: Name of the config (e.g., 'model_config', 'features')
            default: Default value if config not found
        
        Returns:
            Configuration dictionary or default
        """
        return self._configs.get(config_name, default)
    
    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            path: Dot-separated path (e.g., 'model.gnn_layers.0.in_channels')
            default: Default value if path not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config.get_nested('model.type')
            'GAT'
        """
        keys = path.split('.')
        value = self._configs
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list):
                try:
                    value = value[int(key)]
                except (ValueError, IndexError):
                    return default
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def resolve_path(self, relative_path: str) -> Path:
        """
        Resolve a relative path to absolute path from project root.
        
        Args:
            relative_path: Path relative to project root
        
        Returns:
            Absolute Path object
        """
        path = self.settings.PROJECT_ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('model_config', {})
    
    def get_features_config(self) -> Dict[str, Any]:
        """Get features configuration."""
        return self.get('features', {})
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.get('pipeline_config', {})
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        password_part = f":{self.settings.REDIS_PASSWORD}@" if self.settings.REDIS_PASSWORD else ""
        return f"redis://{password_part}{self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}/{self.settings.REDIS_DB}"


@lru_cache()
def get_config() -> ConfigManager:
    """
    Get singleton instance of ConfigManager.
    Uses LRU cache to ensure single instance.
    
    Returns:
        ConfigManager instance
    """
    return ConfigManager()


@lru_cache()
def get_settings() -> Settings:
    """
    Get singleton instance of Settings.
    
    Returns:
        Settings instance
    """
    return Settings()


# Convenience functions
def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return get_config().get_model_config()


def get_features_config() -> Dict[str, Any]:
    """Get features configuration."""
    return get_config().get_features_config()


def get_pipeline_config() -> Dict[str, Any]:
    """Get pipeline configuration."""
    return get_config().get_pipeline_config()


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    settings = get_settings()
    
    print("=" * 60)
    print("Configuration Test")
    print("=" * 60)
    print(f"Project Root: {settings.PROJECT_ROOT}")
    print(f"Database URL: {settings.DATABASE_URL}")
    print(f"Redis URL: {config.redis_url}")
    print(f"\nModel Type: {config.get_nested('model.type')}")
    print(f"Node Input Dim: {config.get_nested('model.node_input_dim')}")
    print(f"Training Epochs: {config.get_nested('training.epochs')}")
    print("=" * 60)