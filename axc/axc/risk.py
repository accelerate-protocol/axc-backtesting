import numpy as np
from enum import Enum


class TailType(Enum):
    UP = 1
    DOWN = 2


# Alternative more concise version using numpy's built-in functions
def calculate_es(samples: np.ndarray, alpha: float, direction: TailType) -> np.ndarray:
    """
    Calculate Expected Shortfall using vectorized operations.
    """
    sort_matrix = np.sort(samples, axis=0)

    # Calculate percentile based on direction
    a = alpha * 100 if direction == TailType.UP else 100 * (1.0 - alpha)
    percentile = np.percentile(sort_matrix, a, axis=0)

    # Create mask and calculate mean
    if direction == TailType.UP:
        return np.mean(sort_matrix, axis=0, where=(sort_matrix >= percentile[None, :]))
    else:
        return np.mean(sort_matrix, axis=0, where=(sort_matrix <= percentile[None, :]))


def calculate_var(samples: np.ndarray, alpha: float, direction: TailType) -> np.ndarray:
    """
    Calculate Value at Risk using vectorized operations.
    """
    sort_matrix = np.sort(samples, axis=0)

    # Calculate percentile based on direction
    a = alpha * 100 if direction == TailType.UP else 100 * (1.0 - alpha)
    return np.percentile(sort_matrix, a, axis=0)


__all__ = ["TailType", "calculate_es", "calculate_var"]
