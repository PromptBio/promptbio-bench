"""Syntax checking utilities for various programming languages."""

import ast
import logging
import os
import subprocess
import tempfile
from typing import Optional, Tuple

from utils.format import get_file_extension

logger = logging.getLogger(__name__)


def check_syntax(file_path: str, content: str) -> Tuple[bool, Optional[str]]:
    """
    Check syntax errors in script.
    
    Args:
        file_path: Path to the script file
        content: Content of the script file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    ext = get_file_extension(file_path)
    
    if ext == "py":
        return check_python_syntax(content)
    elif ext == "r":
        return check_r_syntax(file_path, content)
    elif ext in {"sh", "bash"}:
        return check_shell_syntax(file_path, content)
    elif ext == "pl":
        return check_perl_syntax(file_path, content)
    elif ext == "jl":
        return check_julia_syntax(file_path, content)
    else:
        # Unknown language, skip syntax check
        return True, None


def check_python_syntax(content: str) -> Tuple[bool, Optional[str]]:
    """Check Python syntax using ast.parse()."""
    try:
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f" (near: {e.text.strip()})"
        return False, error_msg
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def check_r_syntax(file_path: str, content: str) -> Tuple[bool, Optional[str]]:
    """Check R syntax by attempting to parse with R."""
    try:
        # Use R's parse() function to check syntax
        with tempfile.NamedTemporaryFile(mode='w', suffix='.r', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Escape backslashes for Windows paths in R string
            escaped_path = tmp_path.replace("\\", "\\\\")
            result = subprocess.run(
                ["R", "--slave", "--no-restore", "--no-save", "-e", f"parse('{escaped_path}')"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0:
                return True, None
            else:
                # Extract error message from stderr
                error_lines = result.stderr.strip().split('\n')
                error_msg = error_lines[-1] if error_lines else "Syntax error detected"
                return False, error_msg
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except FileNotFoundError:
        # R not installed, skip syntax check
        logger.warning("R not found, skipping syntax check")
        return True, "syntax check skipped: R not installed"
    except subprocess.TimeoutExpired:
        return False, "Syntax check timed out"
    except Exception as e:
        logger.warning(f"R syntax check failed: {e}")
        return True, None  # Don't fail validation if check itself fails


def check_shell_syntax(file_path: str, content: str) -> Tuple[bool, Optional[str]]:
    """Check shell/bash syntax using bash -n."""
    try:
        # Determine shell type
        shell = "bash" if get_file_extension(file_path) in {"bash", "sh"} else "sh"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            result = subprocess.run(
                [shell, "-n", tmp_path],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0:
                return True, None
            else:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Syntax error detected"
                return False, error_msg
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except FileNotFoundError:
        logger.warning(f"{shell} not found, skipping syntax check")
        return True, f"syntax check skipped: {shell} not installed"
    except subprocess.TimeoutExpired:
        return False, "Syntax check timed out"
    except Exception as e:
        logger.warning(f"Shell syntax check failed: {e}")
        return True, None


def check_perl_syntax(file_path: str, content: str) -> Tuple[bool, Optional[str]]:
    """Check Perl syntax using perl -c."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            result = subprocess.run(
                ["perl", "-c", tmp_path],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0:
                return True, None
            else:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Syntax error detected"
                return False, error_msg
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except FileNotFoundError:
        logger.warning("Perl not found, skipping syntax check")
        return True, "syntax check skipped: Perl not installed"
    except subprocess.TimeoutExpired:
        return False, "Syntax check timed out"
    except Exception as e:
        logger.warning(f"Perl syntax check failed: {e}")
        return True, None


def check_julia_syntax(file_path: str, content: str) -> Tuple[bool, Optional[str]]:
    """Check Julia syntax using julia parsefile()."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Use Julia's parsefile() function to check syntax without executing
            escaped_path = tmp_path.replace("\\", "\\\\")
            result = subprocess.run(
                ["julia", "--check-bounds=no", "--startup-file=no", "--compile=min", "-e", f"parsefile(\"{escaped_path}\")"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            # Julia may have warnings but still parse correctly
            # Check if it's a syntax error vs runtime error
            if "syntax:" in result.stderr.lower() or "parse error" in result.stderr.lower() or "LoadError" in result.stderr:
                error_msg = result.stderr.strip()
                return False, error_msg
            return True, None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except FileNotFoundError:
        logger.warning("Julia not found, skipping syntax check")
        return True, "syntax check skipped: Julia not installed"
    except subprocess.TimeoutExpired:
        return False, "Syntax check timed out"
    except Exception as e:
        logger.warning(f"Julia syntax check failed: {e}")
        return True, None

