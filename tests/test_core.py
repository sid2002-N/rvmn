import numpy as np
import pytest

import rvmn


def test_apply_rvmn_returns_uint8_same_shape():
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)

    result = rvmn.apply_rvmn(gray, lam=0.1, rho=0.9, theta=0.3)

    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_search_best_params_returns_result():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)

    result = rvmn.search_best_params(
        gray,
        lambda_values=[0.1],
        rho_values=[0.9],
        theta_values=[0.3],
    )

    assert result.params.lam == 0.1
    assert result.params.rho == 0.9
    assert result.params.theta == 0.3
    assert result.image.shape == gray.shape


def test_compare_edges_returns_all_outputs_and_metrics():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)

    result = rvmn.compare_edges(
        gray,
        lambda_values=[0.1],
        rho_values=[0.9],
        theta_values=[0.3],
    )

    assert result.rvmn.image.shape == gray.shape
    assert result.sobel_image.shape == gray.shape
    assert result.prewitt_image.shape == gray.shape
    assert result.rvmn.metrics.rmse >= 0
    assert result.sobel_metrics.rmse >= 0
    assert result.prewitt_metrics.rmse >= 0


def test_compute_metrics_handles_small_images():
    gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    metrics = rvmn.compute_metrics(gray, gray)

    assert metrics.mse == 0
    assert np.isinf(metrics.psnr)
    assert metrics.ssim == 1
    assert metrics.pcc == 1


def test_normalize_to_uint8_rejects_non_finite_values():
    values = np.array([1.0, np.nan])

    with pytest.raises(ValueError, match="finite"):
        rvmn.normalize_to_uint8(values)


def test_search_best_params_rejects_empty_ranges():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)

    with pytest.raises(ValueError, match="lambda_values"):
        rvmn.search_best_params(gray, lambda_values=[], rho_values=[0.1], theta_values=[0.1])
