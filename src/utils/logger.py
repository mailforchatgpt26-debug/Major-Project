"""
Centralized logging configuration for the trade forecasting system.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Color the level name
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


def setup_logger(
    name: str = "trade_forecasting",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None,
    console: bool = True,
    file_rotation: str = "size",  # "size" or "time"
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up and configure logger.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (optional)
        log_dir: Directory for log files
        console: Whether to log to console
        file_rotation: Type of file rotation ("size" or "time")
        max_bytes: Maximum file size before rotation (for size-based rotation)
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        if log_dir is None:
            log_dir = Path("logs")
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = log_dir / log_file
        
        # Choose rotation strategy
        if file_rotation == "time":
            file_handler = TimedRotatingFileHandler(
                log_path,
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:  # size-based rotation
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "trade_forecasting") -> logging.Logger:
    """
    Get logger instance. Creates if doesn't exist.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Setup if not already configured
    if not logger.handlers:
        setup_logger(
            name=name,
            log_level="INFO",
            log_file="app.log",
            console=True
        )
    
    return logger


class LoggerContext:
    """Context manager for temporary logger configuration."""
    
    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize logger context.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.level = level
        self.old_level = logger.level
    
    def __enter__(self):
        """Enter context - set new level."""
        self.logger.setLevel(getattr(logging, self.level.upper()))
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore old level."""
        self.logger.setLevel(self.old_level)


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger: Logger instance (if None, creates default)
    
    Example:
        @log_function_call()
        def my_function(x, y):
            return x + y
    """
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"{func_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{func_name} failed after {duration:.2f}s: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(
        self,
        total: int,
        name: str = "Progress",
        logger: Optional[logging.Logger] = None,
        log_interval: int = 100
    ):
        """
        Initialize progress logger.
        
        Args:
            total: Total number of items
            name: Name of the operation
            logger: Logger instance
            log_interval: Log every N items
        """
        self.total = total
        self.name = name
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """
        Update progress.
        
        Args:
            n: Number of items completed
        """
        self.current += n
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            progress_pct = (self.current / self.total) * 100
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            
            if self.current < self.total:
                eta = (self.total - self.current) / rate if rate > 0 else 0
                self.logger.info(
                    f"{self.name}: {self.current}/{self.total} ({progress_pct:.1f}%) | "
                    f"Rate: {rate:.1f} items/s | ETA: {eta:.0f}s"
                )
            else:
                self.logger.info(
                    f"{self.name}: Completed {self.total} items in {elapsed:.1f}s | "
                    f"Avg rate: {rate:.1f} items/s"
                )
    
    def __enter__(self):
        """Context manager entry."""
        self.logger.info(f"{self.name}: Starting (total: {self.total})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.update(self.total - self.current)  # Ensure completion logged


# Module-level logger instance
_logger = None


def init_logging(config=None):
    """
    Initialize logging system with configuration.
    
    Args:
        config: Configuration object (optional)
    """
    global _logger
    
    log_level = "INFO"
    log_file = "app.log"
    
    if config:
        try:
            log_level = config.settings.LOG_LEVEL
            log_file = config.settings.LOG_FILE
        except AttributeError:
            pass
    
    _logger = setup_logger(
        name="trade_forecasting",
        log_level=log_level,
        log_file=log_file,
        console=True
    )
    
    return _logger


if __name__ == "__main__":
    # Test logging
    logger = setup_logger(
        name="test_logger",
        log_level="DEBUG",
        log_file="test.log",
        console=True
    )
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test progress logger
    with ProgressLogger(1000, name="Test Operation", logger=logger, log_interval=250) as progress:
        for i in range(1000):
            # Simulate work
            pass
            if (i + 1) % 50 == 0:
                progress.update(50)
    
    print("\nCheck test.log for file output")