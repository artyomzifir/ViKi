"""Pure trajectory-cost helpers used by the batch retarget solver."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def huber_loss(norms: np.ndarray, delta: float) -> np.ndarray:
    """Huber ``rho_delta`` evaluated on non-negative residual norms."""
    x = np.asarray(norms, dtype=np.float64)
    if delta <= 0.0:
        raise ValueError("Huber delta must be positive")
    return np.where(x <= delta, 0.5 * x * x, delta * (x - 0.5 * delta))


def huber_irls_weights(norms: np.ndarray, delta: float) -> np.ndarray:
    """Weights whose weighted least squares majorises the Huber data term."""
    x = np.asarray(norms, dtype=np.float64)
    if delta <= 0.0:
        raise ValueError("Huber delta must be positive")
    return np.sqrt(np.where(x <= delta, 1.0, delta / np.maximum(x, 1e-12)))


def difference_matrix(n_frames: int, order: int, dt: float) -> sparse.csr_matrix:
    """First/second temporal finite-difference matrix, scaled to SI time."""
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    if order not in (1, 2):
        raise ValueError("only first and second differences are supported")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    if n_frames <= order:
        return sparse.csr_matrix((0, n_frames), dtype=np.float64)
    if order == 1:
        return sparse.diags(
            (-np.ones(n_frames - 1), np.ones(n_frames - 1)),
            (0, 1), shape=(n_frames - 1, n_frames), format="csr",
        ) / dt
    return sparse.diags(
        (np.ones(n_frames - 2), -2.0 * np.ones(n_frames - 2), np.ones(n_frames - 2)),
        (0, 1, 2), shape=(n_frames - 2, n_frames), format="csr",
    ) / (dt * dt)


def joint_difference_matrix(
    n_frames: int, n_joints: int, order: int, dt: float
) -> sparse.csr_matrix:
    """Temporal difference matrix for a frame-major flattened joint vector."""
    return sparse.kron(
        difference_matrix(n_frames, order, dt),
        sparse.eye(n_joints, format="csr"),
        format="csr",
    )
