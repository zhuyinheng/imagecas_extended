"""
Logging configuration for the ImageCAS Extended package.

Overview:
---------
This module provides a flexible and feature-rich logging system for Python projects, with the following features:
- Rich console output with colors and formatting using the Rich library
- Automatic file logging with timestamped log files
- Progress bar integration for long-running tasks
- Timer context manager for measuring code execution time
- Debug mode with detailed logging and rich tracebacks
- Automatic module name detection for organized logging
- Easy-to-use API for consistent logging across the project

Usage:
------
Import and use the provided functions/classes to enable advanced logging, progress bars, and timing in your scripts.
"""

import logging
import os
import time
import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

# Install rich exception handler for better tracebacks
install_rich_traceback(show_locals=True)

# Create a custom theme for log messages
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "debug": "green",
    "critical": "red bold",
})

# Create a console object with the custom theme
console = Console(theme=custom_theme)

# Create a global progress bar object
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    console=console,
    expand=True,
)

# Store the last created logger
_last_logger: Optional[logging.Logger] = None

def get_caller_module_name() -> str:
    """Get the name of the module that called this function."""
    # Traverse the call stack to find the caller's module name
    frame = inspect.currentframe()
    while frame:
        # Skip frames from this module
        if frame.f_code.co_filename != __file__:
            filename = os.path.basename(frame.f_code.co_filename)
            if filename.endswith('.py'):
                return os.path.splitext(filename)[0]
        frame = frame.f_back
    return "unknown"

def get_logger(
    name: str = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    debug: bool = False,
) -> logging.Logger:
    """Get or create a logger with the specified configuration.
    
    If called without parameters, returns the last created logger.
    Otherwise, creates a new logger with the specified configuration.
    
    Args:
        name: Name of the logger. If None, use the root logger
        log_file: Path to the log file. If None, logs will only be printed to console
        level: Logging level
        debug: Whether to use debug mode with more detailed logging
        
    Returns:
        Configured logger instance
    """
    global _last_logger
    
    # If no parameters provided, return the last created logger
    if name is None and log_file is None and level == logging.INFO and not debug:
        if _last_logger is not None:
            return _last_logger
        # If no logger was created yet, create a default one
        name = get_caller_module_name()
    
    # Get the logger
    logger = logging.getLogger(name)
    
    # Remove existing handlers to avoid duplicate logs
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Set the logging level
    logger.setLevel(logging.DEBUG if debug else level)
    
    # Define log formats
    debug_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    info_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    # Add console handler with rich formatting
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
        level=logging.DEBUG if debug else level,
    )
    logger.addHandler(rich_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # If log_file is a directory, generate a timestamped log file name
        if os.path.isdir(log_file):
            caller_name = get_caller_module_name()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_file, f"{caller_name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            fmt=debug_format,  # Always use debug format for file logging
            datefmt="%m/%d/%y %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        logger.addHandler(file_handler)
    
    # Prevent logs from being propagated to the root logger
    logger.propagate = False
    
    # Store this logger as the last created one
    _last_logger = logger
    
    return logger

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, description: str, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger()  # Use the last created logger if not provided
        self.description = description
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        self.logger.info(f"[bold blue]⏱️ {self.description}[/] took [bold green]{self.duration:.2f}[/] seconds")

def create_progress_bar(total: int, description: str = "Processing") -> Any:
    """Create a progress bar that stays at the bottom of the terminal.
    
    Args:
        total: Total number of items to process
        description: Description of the progress bar
        
    Returns:
        A progress bar object that can be used with a context manager
    """
    return progress.add_task(description, total=total)

def update_progress(task_id: Any, advance: int = 1) -> None:
    """Update the progress bar.
    
    Args:
        task_id: The task ID returned by create_progress_bar
        advance: Number of items completed
    """
    progress.update(task_id, advance=advance)

def start_progress() -> None:
    """Start the progress display."""
    progress.start()

def stop_progress() -> None:
    """Stop the progress display."""
    progress.stop() 