import numpy as np


def compute_max_disagreement(X: np.ndarray) -> float:
    """
    Computes the maximum disagreement across all agents at the current iteration:
    max_{i, j} ||X_i - X_j||_inf
    
    Args:
        X: 2D array of shape (N, 3), where each row is an agent's belief vector.
    """
    if X.shape[0] <= 1:
        return 0.0
    
    max_diff = 0.0
    n = X.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            diff = float(np.max(np.abs(X[i] - X[j])))
            if diff > max_diff:
                max_diff = diff
    return round(max_diff, 5)


def has_converged(prev_X: np.ndarray, curr_X: np.ndarray, tol: float = 1e-4) -> bool:
    """
    Checks if consecutive iterations have changed by less than tolerance tol:
    max_{i, c} |curr_X[i, c] - prev_X[i, c]| < tol
    """
    max_change = float(np.max(np.abs(curr_X - prev_X)))
    return max_change < tol


def compute_stationary_distribution(W: np.ndarray) -> np.ndarray:
    """
    Computes the stationary distribution (left eigenvector) pi such that:
    pi * W = pi, sum(pi) = 1.0.
    This reflects the asymptotic influence of each agent under the DeGroot dynamics.
    """
    n = W.shape[0]
    # Solve (W^T - I) pi = 0 subject to sum(pi) = 1
    eigenvalues, eigenvectors = np.linalg.eig(W.T)
    # Find eigenvalue closest to 1.0
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.real(eigenvectors[:, idx])
    total = np.sum(stationary)
    if total != 0:
        stationary = stationary / total
    else:
        stationary = np.full(n, 1.0 / n)
    return np.maximum(0.0, stationary)
