"""Logging utilities: terminal output formatting, stream multiplexing, and log management."""

import re
import sys
import textwrap
from contextlib import redirect_stdout, nullcontext
from pathlib import Path
from typing import IO, Callable


def wrap_text(text: str, width: int = 100, initial_indent: str = "", subsequent_indent: str = "\t") -> str:
    """Wrap text to fit within width with optional indentation.

    Args:
        text: The text to wrap.
        width: Maximum line width (default: 100).
        initial_indent: Prefix for the first line (e.g. "  - reasoning: ").
        subsequent_indent: Prefix for continuation lines (default: 4 spaces).

    Example:
        print(wrap_text(reasoning, width=100, initial_indent="  - reasoning: "))
    """
    return textwrap.fill(text, width=width, initial_indent=initial_indent, subsequent_indent=subsequent_indent)


def color_text(text: str, color: str = "red", bold: bool = True) -> str:
    """Format text with ANSI color codes for terminal output."""
    colors = {
        "black": 30, "red": 31, "green": 32, "yellow": 33,
        "blue": 34, "magenta": 35, "cyan": 36, "white": 37
    }
    code = colors.get(color, 37)
    prefix = f"\033[{1 if bold else 0};{code}m"
    suffix = "\033[0m"
    return f"{prefix}{text}{suffix}"


class Tee:
    """Fan out writes to a display stream and a log file, stripping ANSI from the log."""

    _ANSI = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, display: IO[str], log: IO[str]) -> None:
        self._display = display
        self._log = log

    def write(self, data: str) -> None:
        self._display.write(data)
        self._log.write(self._ANSI.sub("", data))

    def flush(self) -> None:
        self._display.flush()
        self._log.flush()


def log_context(log_file: str | IO[str] | None):
    """Context manager that redirects all stdout to both the console and log_file.

    Any print() calls within the block — including from third-party code — are
    captured automatically. If log_file is None, acts as a no operation.

    Args:
        log_file: A path string, an open writable file object, or None.

    Example:
        with log_context("run.log"):
            print("captured to console and file")
            some_library_function()  # its prints are captured too
    """
    if log_file is None:
        return nullcontext()
    if isinstance(log_file, (str, Path)):
        log_file = open(log_file, "a")
    return redirect_stdout(Tee(sys.stdout, log_file))


def make_logger(log_file: str | IO[str], overwrite: bool = True) -> Callable[[str], None]:
    """Return a print-like callable that writes to both stdout and log_file.

    Args:
        log_file: A path string or an open writable file object.
        overwrite: If True (default), overwrite the log file; if False, append to it.

    Returns:
        A callable that accepts a message string, prints it, and flushes to log_file.

    Example:
        log = make_logger("run.log")
        log("starting task...")
    """
    if isinstance(log_file, (str, Path)):
        log_file = open(log_file, "w" if overwrite else "a")

    def log(message: str = "") -> None:
        print(message)
        log_file.write(Tee._ANSI.sub("", message) + "\n")
        log_file.flush()

    return log
