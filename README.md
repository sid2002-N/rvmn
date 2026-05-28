# rvmn

RVMN is a small Python package for directional edge detection using custom
RVMN-style masks, plus Sobel/Prewitt comparison helpers.

## Install

```bash
pip install rvmn
```

For local development from this folder:

```bash
pip install -e ".[dev,plots]"
```

## Python Usage

```python
import cv2
import rvmn

image = cv2.imread("image.png")
gray = rvmn.to_grayscale(image)

result = rvmn.apply_rvmn(gray, lam=0.1, rho=0.9, theta=0.3)
sobel = rvmn.apply_sobel(gray)
prewitt = rvmn.apply_prewitt(gray)

metrics = rvmn.compute_metrics(gray, result)
print(metrics.psnr, metrics.rmse)
```

Search for the best parameter combination:

```python
search = rvmn.search_best_params(gray)

print(search.params)
print(search.metrics.psnr)
cv2.imwrite("rvmn_output.png", search.image)
```

Compare RVMN, Sobel, and Prewitt in one call:

```python
comparison = rvmn.compare_edges(gray)

print(comparison.rvmn.params)
print(comparison.sobel_metrics.psnr)
print(comparison.prewitt_metrics.psnr)
```

## Command Line

```bash
rvmn image.png --output results
```

This writes `rvmn_output.png`, `sobel_output.png`, and `prewitt_output.png`.

Use fixed RVMN parameters instead of a full search:

```bash
rvmn image.png --lambda 0.1 --rho 0.9 --theta 0.3 --output results
```

Print machine-readable metrics:

```bash
rvmn image.png --json
```

## Build

```bash
python -m build
```

The build configuration includes only `src/`, `tests/`, `README.md`, and
`pyproject.toml` in the source archive, and only the `rvmn` package in the wheel.
Generated notebooks, images, PDFs, result folders, caches, and old `dist/`
contents are excluded.

## Notes

The coefficient formulas are packaged from the current notebook implementation.
Keep citation and formula details close to your paper/report when publishing.
