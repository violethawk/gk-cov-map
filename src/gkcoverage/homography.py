from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def solve_homography(pixel_xy: ArrayLike, plane_xy: ArrayLike) -> NDArray[np.float64]:
    """Solve a projective transform from four or more point correspondences.

    Returns H such that [x_m, y_m, 1]^T ~ H [u_px, v_px, 1]^T.
    Coordinates are normalized before DLT for numerical stability.
    """

    src = np.asarray(pixel_xy, dtype=float)
    dst = np.asarray(plane_xy, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 4:
        raise ValueError("pixel_xy and plane_xy must be matching Nx2 arrays with N>=4")

    def normalize(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        centroid = points.mean(axis=0)
        shifted = points - centroid
        mean_dist = np.linalg.norm(shifted, axis=1).mean()
        if mean_dist <= 0:
            raise ValueError("degenerate point configuration")
        scale = np.sqrt(2.0) / mean_dist
        transform = np.array(
            [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]]
        )
        homogeneous = np.c_[points, np.ones(points.shape[0])]
        normalized = (transform @ homogeneous.T).T[:, :2]
        return normalized, transform

    src_n, t_src = normalize(src)
    dst_n, t_dst = normalize(dst)
    rows: list[list[float]] = []
    for (u, v), (x, y) in zip(src_n, dst_n, strict=True):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h_n = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_n @ t_src
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography")
    return h / h[2, 2]


def project_points(h: ArrayLike, pixel_xy: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(h, dtype=float)
    points = np.asarray(pixel_xy, dtype=float)
    single = points.ndim == 1
    points = np.atleast_2d(points)
    homogeneous = np.c_[points, np.ones(points.shape[0])]
    mapped = (matrix @ homogeneous.T).T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise ValueError("point projects to infinity")
    mapped = mapped[:, :2] / mapped[:, 2, None]
    return mapped[0] if single else mapped
