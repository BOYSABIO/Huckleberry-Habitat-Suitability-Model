"""
Structured logging configuration for the pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "huckleberry_pipeline",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up structured logging for the pipeline.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        format_string: Custom format string (optional)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "huckleberry_pipeline") -> logging.Logger:
    """
    Get a logger instance with default configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_pipeline_step(step_name: str, logger: Optional[logging.Logger] = None):
    """
    Decorator to log pipeline step execution.
    
    Args:
        step_name: Name of the pipeline step
        logger: Logger instance (optional)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                log = get_logger()
            else:
                log = logger
            
            log.info(f"Starting pipeline step: {step_name}")
            try:
                result = func(*args, **kwargs)
                log.info(f"Completed pipeline step: {step_name}")
                return result
            except Exception as e:
                log.error(f"Failed pipeline step: {step_name} - {str(e)}")
                raise
        return wrapper
    return decorator 