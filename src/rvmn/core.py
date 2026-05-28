from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


@dataclass(frozen=True)
class RVMNParams:
    """Parameter set for the RVMN mask family."""

    lam: float
    rho: float
    theta: float


@dataclass(frozen=True)
class Metrics:
    """Image comparison metrics."""

    mse: float
    rmse: float
    psnr: float
    ssim: float
    fom: float
    pcc: float
    mae: float


@dataclass(frozen=True)
class SearchResult:
    """Best RVMN output from a parameter search."""

    params: RVMNParams
    image: np.ndarray
    metrics: Metrics


@dataclass(frozen=True)
class ComparisonResult:
    """RVMN, Sobel, and Prewitt outputs with their metrics."""

    rvmn: SearchResult
    sobel_image: np.ndarray
    sobel_metrics: Metrics
    prewitt_image: np.ndarray
    prewitt_metrics: Metrics


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a grayscale uint8 image from a BGR/RGB/grayscale input."""
    if image is None:
        raise ValueError("image must not be None")
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unsupported image shape: {image.shape}")


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    """Normalize an array to the [0, 255] uint8 range."""
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("values must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("values must contain only finite numbers")
    if float(values.max()) == float(values.min()):
        return np.zeros(values.shape, dtype=np.uint8)
    return cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def generate_rvmn_masks(lam: float, rho: float, theta: float) -> list[np.ndarray]:
    """Generate four directional RVMN masks.

    This follows the current notebook implementation:
    a1 = 1, a2 depends on lambda, a3 depends on theta, and a4 depends on rho.
    """
    if 1 + abs(lam) == 0 or 1 + 2 * theta == 0 or 1 + 3 * rho == 0:
        raise ValueError("lam, rho, and theta produce invalid mask coefficients")

    a1 = 1.0
    a2 = 5 / (6 * (1 + abs(lam)))
    a3 = 5 / (12 * (1 + 2 * theta))
    a4 = 5 / (18 * (1 + 3 * rho))

    mask_1 = np.array(
        [
            [-a3, 0, a1],
            [-a2, 0, a2],
            [-a1, 0, a3],
        ],
        dtype=np.float64,
    )

    mask_2 = np.array(
        [
            [-a1, -a2, -a3],
            [0, 0, 0],
            [a3, a2, a1],
        ],
        dtype=np.float64,
    )

    mask_3 = np.array(
        [
            [0, a4, a1],
            [-a4, 0, a4],
            [-a1, -a4, 0],
        ],
        dtype=np.float64,
    )

    mask_4 = np.array(
        [
            [a1, a4, 0],
            [a4, 0, -a4],
            [0, -a4, -a1],
        ],
        dtype=np.float64,
    )

    return [mask_1, mask_2, mask_3, mask_4]


def apply_rvmn(gray: np.ndarray, lam: float, rho: float, theta: float) -> np.ndarray:
    """Apply RVMN directional masks to a grayscale image."""
    gray = to_grayscale(gray)
    combined = np.zeros_like(gray, dtype=np.float64)
    for mask in generate_rvmn_masks(lam, rho, theta):
        combined += np.abs(cv2.filter2D(gray, cv2.CV_64F, mask))
    return normalize_to_uint8(combined)


def apply_sobel(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Sobel edge detection and normalize to uint8."""
    gray = to_grayscale(gray)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return normalize_to_uint8(np.abs(sobel_x) + np.abs(sobel_y))


def apply_prewitt(gray: np.ndarray) -> np.ndarray:
    """Apply Prewitt edge detection and normalize to uint8."""
    gray = to_grayscale(gray)
    prewitt_kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_ky = prewitt_kx.T
    prewitt_x = cv2.filter2D(gray, cv2.CV_64F, prewitt_kx)
    prewitt_y = cv2.filter2D(gray, cv2.CV_64F, prewitt_ky)
    return normalize_to_uint8(np.abs(prewitt_x) + np.abs(prewitt_y))


def compute_metrics(reference: np.ndarray, processed: np.ndarray) -> Metrics:
    """Compute MSE, RMSE, PSNR, SSIM, FOM, PCC, and MAE."""
    reference = to_grayscale(reference)
    processed = to_grayscale(processed)
    if reference.shape != processed.shape:
        raise ValueError("reference and processed images must have the same shape")
    if reference.size == 0:
        raise ValueError("images must not be empty")

    ref_f = reference.astype(np.float64).flatten()
    proc_f = processed.astype(np.float64).flatten()

    mse = float(np.mean((ref_f - proc_f) ** 2))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0 else float(psnr_fn(reference, processed, data_range=255))
    min_side = min(reference.shape)
    if min_side < 3:
        ssim = 1.0 if np.array_equal(reference, processed) else float("nan")
    else:
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        ssim = float(ssim_fn(reference, processed, data_range=255, win_size=win_size))
    fom = 1.0 / (1.0 + 0.1 * mse) if mse > 0 else 1.0
    mae = float(np.mean(np.abs(ref_f - proc_f)))

    ref_c = ref_f - ref_f.mean()
    proc_c = proc_f - proc_f.mean()
    denom = math.sqrt(float((ref_c**2).sum() * (proc_c**2).sum()))
    if denom > 0:
        pcc = float(np.dot(ref_c, proc_c) / denom)
    else:
        pcc = 1.0 if np.array_equal(reference, processed) else 0.0

    return Metrics(mse=mse, rmse=rmse, psnr=psnr, ssim=ssim, fom=fom, pcc=pcc, mae=mae)


def _coerce_values(name: str, values: Iterable[float]) -> list[float]:
    coerced = [float(value) for value in values]
    if not coerced:
        raise ValueError(f"{name} must contain at least one value")
    if not all(math.isfinite(value) for value in coerced):
        raise ValueError(f"{name} must contain only finite numbers")
    return coerced


def search_best_params(
    gray: np.ndarray,
    lambda_values: Iterable[float] | None = None,
    rho_values: Iterable[float] | None = None,
    theta_values: Iterable[float] | None = None,
) -> SearchResult:
    """Search RVMN parameters and return the output with the highest PSNR."""
    gray = to_grayscale(gray)
    lambda_values = _coerce_values(
        "lambda_values",
        lambda_values if lambda_values is not None else np.round(np.arange(0.1, 1.0, 0.1), 2),
    )
    rho_values = _coerce_values(
        "rho_values",
        rho_values if rho_values is not None else np.round(np.arange(0.1, 1.0, 0.1), 2),
    )
    theta_values = _coerce_values(
        "theta_values",
        theta_values if theta_values is not None else np.round(np.arange(0.1, 1.0, 0.1), 2),
    )

    best: SearchResult | None = None
    for lam, rho, theta in product(lambda_values, rho_values, theta_values):
        image = apply_rvmn(gray, lam, rho, theta)
        metrics = compute_metrics(gray, image)
        if best is None or metrics.psnr > best.metrics.psnr:
            best = SearchResult(
                params=RVMNParams(lam, rho, theta),
                image=image,
                metrics=metrics,
            )

    assert best is not None
    return best


def theta_sweep(
    gray: np.ndarray,
    lam: float,
    rho: float,
    theta_values: Iterable[float] | None = None,
) -> list[tuple[float, Metrics]]:
    """Evaluate RVMN across theta values while holding lambda and rho fixed."""
    gray = to_grayscale(gray)
    theta_values = _coerce_values(
        "theta_values",
        theta_values if theta_values is not None else np.round(np.arange(0.1, 1.0, 0.1), 2),
    )
    return [
        (float(theta), compute_metrics(gray, apply_rvmn(gray, lam, rho, float(theta))))
        for theta in theta_values
    ]


def compare_edges(
    gray: np.ndarray,
    lambda_values: Iterable[float] | None = None,
    rho_values: Iterable[float] | None = None,
    theta_values: Iterable[float] | None = None,
) -> ComparisonResult:
    """Compare the best RVMN result with Sobel and Prewitt on one image."""
    gray = to_grayscale(gray)
    rvmn = search_best_params(gray, lambda_values, rho_values, theta_values)
    sobel = apply_sobel(gray)
    prewitt = apply_prewitt(gray)
    return ComparisonResult(
        rvmn=rvmn,
        sobel_image=sobel,
        sobel_metrics=compute_metrics(gray, sobel),
        prewitt_image=prewitt,
        prewitt_metrics=compute_metrics(gray, prewitt),
    )
