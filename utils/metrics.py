import numpy as np
import scipy.stats as stats

from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from numpy.typing import ArrayLike
from typing import Optional, Tuple, Union


def pearson_correlation(x: ArrayLike, y: ArrayLike, nan_policy: str = "propagate") -> float:
    """
    Calculate the Pearson correlation coefficient between two 1D arrays.

    Args:
        x: Input array-like of shape (n,).
        y: Input array-like of shape (n,).
        nan_policy: {'propagate', 'raise', 'omit'}
            Defines how to handle when input contains nan. Default is 'propagate'.

    Returns:
        Pearson correlation coefficient or NaN if input is not valid or correlation is undefined.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape.")

        # Use scipy.stats.pearsonr for robust and standard computation
        corr, _ = stats.pearsonr(x, y, nan_policy=nan_policy)
        return corr
    except Exception:
        return np.nan

def spearman_correlation(x: ArrayLike, y: ArrayLike, nan_policy: str = "propagate") -> float:
    """
    Calculate the Spearman correlation coefficient between two 1D arrays.

    Args:
        x: Input array-like of shape (n,).
        y: Input array-like of shape (n,).
        nan_policy: {'propagate', 'raise', 'omit'}
    
    Returns:
        Spearman correlation coefficient or NaN if input is not valid or correlation is undefined.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape.")
        
        # Use scipy.stats.spearmanr for robust and standard computation
        corr, _ = stats.spearmanr(x, y, nan_policy=nan_policy)
        return corr
    except Exception:
        return np.nan

def kendall_correlation(x: ArrayLike, y: ArrayLike, nan_policy: str = "propagate") -> float:
    """
    Calculate the Kendall correlation coefficient between two 1D arrays.

    Args:
        x: Input array-like of shape (n,).
        y: Input array-like of shape (n,).
        nan_policy: {'propagate', 'raise', 'omit'}
    
    Returns:
        Kendall correlation coefficient or NaN if input is not valid or correlation is undefined.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape.")
        
        # Use scipy.stats.kendalltau for robust and standard computation
        corr, _ = stats.kendalltau(x, y, nan_policy=nan_policy)
        return corr
    except Exception:
        return np.nan

def cosine_similarity(x: ArrayLike, y: ArrayLike) -> float:
    """
    Calculate the cosine similarity between two 1D arrays.

    Args:
        x: Input array-like of shape (n,).
        y: Input array-like of shape (n,).

    Returns:
        Cosine similarity as a float or NaN if input is not valid or similarity is undefined.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape.")

        # Reshape to (1, n_features) for sklearn
        sim = sklearn_cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]
        return float(sim)
    except Exception:
        return np.nan

def hellinger_distance(x: ArrayLike, y: ArrayLike) -> float:
    """
    Calculate the Hellinger distance between two 1D probability distributions.

    Hellinger distance = (1 / sqrt(2)) * sqrt(sum((sqrt(p_i) - sqrt(q_i))^2)).

    Args:
        x: Input array-like of shape (n,). Non-negative values (will be normalized).
        y: Input array-like of shape (n,). Non-negative values (will be normalized).

    Returns:
        Hellinger distance as a float or NaN if input is invalid or distance is undefined.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape.")

        if np.any(x < 0) or np.any(y < 0):
            raise ValueError("Inputs must be non-negative for Hellinger distance.")

        # Normalize to probability distributions
        x_sum = np.sum(x)
        y_sum = np.sum(y)
        if x_sum == 0 or y_sum == 0:
            return np.nan

        x = x / x_sum
        y = y / y_sum

        # Compute Hellinger distance
        distance = (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(x) - np.sqrt(y))
        return float(distance)
    except Exception:
        return np.nan

def mean_absolute_error(x: ArrayLike, y: ArrayLike) -> float:
    """
    Calculate the mean absolute error (MAE) between two arrays.

    MAE = mean(|x_i - y_i|).

    Supports both vectors (1D) and matrices (2D or higher dimensions).
    The error is calculated element-wise and then averaged.

    Args:
        x: Input array-like (can be vector or matrix).
        y: Input array-like (must have the same shape as x).

    Returns:
        Mean absolute error as a float or NaN if input is invalid or error is undefined.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape.")

        return float(np.mean(np.abs(x - y)))
    except Exception:
        return np.nan

def mean_squared_error(x: ArrayLike, y: ArrayLike) -> float:
    """
    Calculate the mean squared error (MSE) between two arrays.

    MSE = mean((x_i - y_i)^2).

    Supports both vectors (1D) and matrices (2D or higher dimensions).
    The error is calculated element-wise and then averaged.

    Args:
        x: Input array-like (can be vector or matrix).
        y: Input array-like (must have the same shape as x).

    Returns:
        Mean squared error as a float or NaN if input is invalid or error is undefined.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape.")

        # Calculate mean squared error
        diff = x - y
        return float(np.mean(diff ** 2))

    except Exception:
        return np.nan

def relative_absolute_error(x: ArrayLike, y: ArrayLike, x0_match_rae: float = 0.0, x0_mismatch_rae: float = np.inf) -> Union[np.ndarray, float]:
    """
    Compute the relative absolute error (RAE) between two arrays.

    RAE = |x - y| / |x|

    Works for both 1D and 2D arrays. Division-by-zero elements in |x|
    are handled based on x0_match_rae and x0_mismatch_rae parameters.

    Args:
        x: Input array-like of shape (n,) or (m, n).
        y: Input array-like of the same shape as x.
        x0_match_rae: RAE value when x=0 and y=0 (both zero). 
                     Default: 0.0 (perfect match).
        x0_mismatch_rae: RAE value when x=0 and y≠0 (only x is zero).
                        Default: np.inf (no match).

    Returns:
        A NumPy array of the same shape as x containing relative absolute errors.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")

        denom = np.abs(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            rae = np.abs(x - y) / denom
        
        # Handle division by zero cases
        x0_mask = denom == 0
        if x0_mask.any():
            y0_mask = np.abs(y) == 0
            rae[x0_mask & y0_mask] = x0_match_rae
            rae[x0_mask & ~y0_mask] = x0_mismatch_rae

        return rae
    except Exception:
        return np.nan

def rae_similarity(x: ArrayLike, y: ArrayLike, clip: bool = True) -> np.ndarray:
    """
    Calculate similarity using relative absolute error (RAE) for arrays.
    Converts RAE to similarity: similarity = 1 - RAE
    Handles edge cases:
    - Both values are zero → returns 1.0 (perfect match)
    - One value is zero, other is not → returns 0.0 (no match)
    - Otherwise → returns 1 - RAE (clipped to [0, 1] if clip=True)
    
    Args:
        x: First array (used as denominator in RAE calculation)
        y: Second array to compare against x (must have same shape as x)
        clip: If True, clip similarity to [0, 1] range. Default: True
        
    Returns:
        Array of similarity scores between 0.0 and 1.0 (if clip=True)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    
    # relative_absolute_error already handles zeros:
    # - x=0, y=0: RAE = 0.0 → sim = 1.0 (perfect match, already correct)
    # - x=0, y≠0: RAE = inf → sim = -inf (needs to be set to 0.0)
    # - x≠0, y=0: RAE = |x|/|x| = 1.0 → sim = 0.0 (already correct)
    rae = relative_absolute_error(x, y)
    sim = 1.0 - rae
    # Handle case where x=0 and y≠0: RAE is inf, so sim is -inf, set to 0.0
    sim[(x == 0) & (y != 0)] = 0.0
    # Clip to [0, 1] if requested
    return np.clip(sim, 0.0, 1.0) if clip else sim

def min_max_similarity(x: float, y: float) -> float:
    """
    Calculate similarity using min/max ratio.
    
    This is a symmetric similarity metric: min(x, y) / max(x, y)
    Equivalent to RAE-based similarity when using max(|x|, |y|) as denominator.
    
    Handles edge cases:
    - Both values are zero → returns 1.0 (perfect match)
    - One value is zero, other is not → returns 0.0 (no match)
    - Otherwise → returns min(x, y) / max(x, y)
    
    Note: Works best with non-negative values. For negative values, consider
    using absolute values or a different similarity metric.
    
    Args:
        x: First value
        y: Second value
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if x == 0 and y == 0:
        return 1.0  # Both zero, perfect match
    elif x == 0 or y == 0:
        return 0.0  # One is zero, other is not
    else:
        return min(x, y) / max(x, y)

def normalized_difference(x: ArrayLike, y: ArrayLike) -> float:
    """
    Calculate the normalized difference between two arrays.

    ND = (x - y) / (x + y)

    Args:
        x: Input array-like of shape (n,).
        y: Input array-like of shape (n,).

    Returns:
        Normalized difference as a float or NaN if input is invalid or difference is undefined.
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape.")

        return float(np.mean((x - y) / (x + y)))
    except Exception:
        return np.nan

def _count_indicators(x: ArrayLike, y: ArrayLike) -> Tuple[int, int, int]:
    """Validate binary indicator arrays and return (x_count, y_count, overlapped)."""
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape.")
    if not (np.isin(x, [0, 1]).all() and np.isin(y, [0, 1]).all()):
        raise ValueError("Input arrays must contain only 0 and 1.")
    x_count = int(np.sum(x))
    y_count = int(np.sum(y))
    overlapped = int(np.sum(x & y))
    return x_count, y_count, overlapped

def precision_recall_f1(x_count: Optional[int] = None, y_count: Optional[int] = None, overlapped: Optional[int] = None,
                        x: Optional[ArrayLike] = None, y: Optional[ArrayLike] = None) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score from overlap counts or binary indicator arrays.

    Call with either (x_count, y_count, overlapped) or (x=..., y=...).
    For set intersection: overlapped = len(x_set & y_set). Precision = overlapped/y_count,
    recall = overlapped/x_count.

    Args:
        x_count: Total count for x (regions, rows, or items). Required if arrays not provided.
        y_count: Total count for y. Required if arrays not provided.
        overlapped: Number of items in the intersection. Required if arrays not provided.
        x: Optional 1D binary array (0/1) for first set. Must pair with y.
        y: Optional 1D binary array (0/1) for second set. Must pair with x.

    Returns:
        Tuple of (precision, recall, f1_score). Returns (nan, nan, nan) if given invalid indicator arrays.
    """
    if x is not None and y is not None:
        try:
            x_count, y_count, overlapped = _count_indicators(x, y)
        except Exception:
            return np.nan, np.nan, np.nan
    if x_count is None or y_count is None or overlapped is None:
        raise ValueError("Provide either (x_count, y_count, overlapped) or (x=..., y=...).")
    precision = overlapped / y_count if y_count > 0 else 0.0
    recall = overlapped / x_count if x_count > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score

def jaccard_similarity(x_count: Optional[int] = None, y_count: Optional[int] = None, overlapped: Optional[int] = None,
                       x: Optional[ArrayLike] = None, y: Optional[ArrayLike] = None) -> float:
    """Calculate Jaccard similarity from summary counts or binary indicator arrays.

    Call with either (x_count, y_count, overlapped) or (x=..., y=...).
    Jaccard = |X∩Y| / |X∪Y|. Union is derived as x_count + y_count - overlapped.

    Args:
        x_count: Size of first set |X|. Required if arrays not provided.
        y_count: Size of second set |Y|. Required if arrays not provided.
        overlapped: Size of intersection |X∩Y|. Required if arrays not provided.
        x: Optional 1D binary array (0/1) for first set. Must pair with y.
        y: Optional 1D binary array (0/1) for second set. Must pair with x.

    Returns:
        Jaccard similarity in [0, 1], or 1.0 when union is 0 (empty sets). Returns nan if invalid indicator arrays.
    """
    if x is not None and y is not None:
        try:
            x_count, y_count, overlapped = _count_indicators(x, y)
        except Exception:
            return np.nan
    if x_count is None or y_count is None or overlapped is None:
        raise ValueError("Provide either (x_count, y_count, overlapped) or (x=..., y=...).")
    union = x_count + y_count - overlapped
    if union <= 0:
        return 1.0
    return overlapped / union if overlapped >= 0 else 0.0
