import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def get_logger(
    name: str = "Agentic Rag",
    log_file: str = "logs/agentic-rag.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3
) -> logging.Logger:
    """Configure and return a logger with console + rotating file handler."""

    logger = logging.getLogger(name)

    if not logger.handlers:  # prevent duplicate logs
        logger.setLevel(logging.DEBUG)

        # Ensure log directory exists
        log_path = Path(log_file).parent.parent / "logs"
        log_path.mkdir(parents=True, exist_ok=True)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # File handler (rotating)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
