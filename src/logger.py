import logging
import os
from rich.logging import RichHandler

def get_logger(name: str, log_dir: str = "logs", level: str = "INFO", log_to_file: bool = True, file_name: str = "app.log"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console (pretty) handler
    console_handler = RichHandler(rich_tracebacks=True, show_time=False)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, file_name))
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)

    return logger
