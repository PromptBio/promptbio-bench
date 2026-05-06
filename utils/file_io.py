"""File I/O utilities for reading text files and file samples."""
import os
import gzip
import logging

logger = logging.getLogger(__name__)


def read_file(file_path: str | os.PathLike, max_chars: int | None = 1000, handle_binary: bool = False, encoding: str = 'utf-8', errors: str = 'ignore') -> str | None:
    """
    Read file content, handling text, binary, and compressed files.
    
    Args:
        file_path: Path to the file
        max_chars: Maximum number of characters to return (default: 1000). Set to None to disable truncation.
        handle_binary: If True, show error messages/hex preview for binary files. If False, return None (default: False)
        encoding: File encoding (default: 'utf-8')
        errors: How to handle encoding errors (default: 'ignore')
        
    Returns:
        File content as string, error message string, or None if handle_binary=False
    """
    if not os.path.exists(file_path):
        return None if not handle_binary else f"<error reading file: file not found>"
    
    def _is_binary(file_path):
        """Detect if file is binary by checking first bytes."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)  # Read first 8KB
                if b'\x00' in chunk:  # Null bytes indicate binary
                    return True
                # Try to decode as UTF-8
                chunk.decode('utf-8', errors='strict')
                return False
        except (UnicodeDecodeError, UnicodeError):
            return True
        except Exception:
            return False
    
    def _is_compressed(file_path):
        """Detect if file is compressed by checking magic numbers."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(2)  # Read first 2 bytes
                # Gzip magic number: 1f 8b
                if header == b'\x1f\x8b':
                    return True
                # Check extension as fallback
                return str(file_path).endswith('.gz')
        except Exception:
            return False
    
    try:
        # Check if file is compressed
        if _is_compressed(file_path):
            with gzip.open(file_path, 'rt', encoding=encoding, errors=errors) as f:
                content = f.read()
        else:
            # Check if file is binary
            if _is_binary(file_path):
                if not handle_binary:
                    return None
                with open(file_path, 'rb') as f:
                    binary_content = f.read(100)
                    return f"<binary file, first 100 bytes (hex): {binary_content.hex()[:200]}...>"
            
            # Read as text
            with open(file_path, 'r', encoding=encoding, errors=errors) as f:
                content = f.read()
        
        # Truncate if too long
        if max_chars is not None and len(content) > max_chars:
            return f"{content[:max_chars]}... (truncated, total {len(content)} chars)"
        return content
    except Exception as e:
        if not handle_binary:
            return None
        return f"<error reading file: {e}>"

