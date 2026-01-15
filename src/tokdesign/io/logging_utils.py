"""
logging_utils.py
==========

Central logging utilities.

Responsibilities
----------------
• Configure a consistent logger
• Log to:
    - console (stdout)
    - file (run.log)
• Avoid duplicate handlers
• Provide simple interface for all scripts

Design principles
-----------------
• One logger per run
• Human-readable format
• Timestamped entries
"""

import logging
from pathlib import Path


# ============================================================
# LOGGER SETUP
# ============================================================

def setup_logger(log_path: Path, level: str = "INFO") -> logging.Logger:
    """
    Create and configure a logger.

    Logs to:
    • console
    • file (log_path)

    Parameters
    ----------
    log_path : Path
        Path to log file (e.g. run_dir/run.log)
    level : str
        Logging level:
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """

    log_path = Path(log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = "tokdesign"
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # Set level
    level = level.upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Format
    fmt = (
        "%(asctime)s | "
        "%(levelname)-8s | "
        "%(name)s | "
        "%(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # --------------------------------------------------------
    # Console handler
    # --------------------------------------------------------
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --------------------------------------------------------
    # File handler
    # --------------------------------------------------------
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logger.level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug("Logger initialized")
    logger.info(f"Logging to file: {log_path}")

    return logger


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("Testing logging.py")

    test_log = Path("/tmp/tokamak_test.log")
    logger = setup_logger(test_log, level="DEBUG")

    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    print(f"Check log file: {test_log}")
    print("logging.py self-test passed")
