from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .core import (
    Metrics,
    RVMNParams,
    SearchResult,
    apply_prewitt,
    apply_rvmn,
    apply_sobel,
    compare_edges,
    compute_metrics,
    to_grayscale,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RVMN edge detection on an image.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("-o", "--output", default="results", help="Directory for output images.")
    parser.add_argument("--no-resize", action="store_true", help="Do not resize before searching.")
    parser.add_argument("--size", type=int, default=256, help="Resize width/height used for searching.")
    parser.add_argument("--lambda", dest="lam", type=float, help="Fixed lambda value. Requires --rho and --theta.")
    parser.add_argument("--rho", type=float, help="Fixed rho value. Requires --lambda and --theta.")
    parser.add_argument("--theta", type=float, help="Fixed theta value. Requires --lambda and --rho.")
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    return parser


def _metrics_dict(metrics: Metrics) -> dict[str, float]:
    return {
        "mse": metrics.mse,
        "rmse": metrics.rmse,
        "psnr": metrics.psnr,
        "ssim": metrics.ssim,
        "fom": metrics.fom,
        "pcc": metrics.pcc,
        "mae": metrics.mae,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Image not found: {image_path}")

    if not args.no_resize:
        image = cv2.resize(image, (args.size, args.size))

    gray = to_grayscale(image)
    fixed_values = [args.lam, args.rho, args.theta]
    if any(value is not None for value in fixed_values) and any(value is None for value in fixed_values):
        raise SystemExit("--lambda, --rho, and --theta must be provided together")

    if all(value is not None for value in fixed_values):
        rvmn_image = apply_rvmn(gray, args.lam, args.rho, args.theta)
        result = SearchResult(
            params=RVMNParams(args.lam, args.rho, args.theta),
            image=rvmn_image,
            metrics=compute_metrics(gray, rvmn_image),
        )
        sobel = apply_sobel(gray)
        prewitt = apply_prewitt(gray)
        sobel_metrics = compute_metrics(gray, sobel)
        prewitt_metrics = compute_metrics(gray, prewitt)
        label = "RVMN"
    else:
        comparison = compare_edges(gray)
        result = comparison.rvmn
        sobel = comparison.sobel_image
        prewitt = comparison.prewitt_image
        sobel_metrics = comparison.sobel_metrics
        prewitt_metrics = comparison.prewitt_metrics
        label = "Best"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "rvmn_output.png"), result.image)
    cv2.imwrite(str(output_dir / "sobel_output.png"), sobel)
    cv2.imwrite(str(output_dir / "prewitt_output.png"), prewitt)

    if args.json:
        print(
            json.dumps(
                {
                    "params": {
                        "lambda": result.params.lam,
                        "rho": result.params.rho,
                        "theta": result.params.theta,
                    },
                    "rvmn": _metrics_dict(result.metrics),
                    "sobel": _metrics_dict(sobel_metrics),
                    "prewitt": _metrics_dict(prewitt_metrics),
                    "output": str(output_dir),
                },
                indent=2,
            )
        )
        return 0

    print(f"{label}: lambda={result.params.lam}, rho={result.params.rho}, theta={result.params.theta}")
    print(f"RVMN:    PSNR={result.metrics.psnr:.3f} dB  RMSE={result.metrics.rmse:.3f}")
    print(f"Sobel:   PSNR={sobel_metrics.psnr:.3f} dB  RMSE={sobel_metrics.rmse:.3f}")
    print(f"Prewitt: PSNR={prewitt_metrics.psnr:.3f} dB  RMSE={prewitt_metrics.rmse:.3f}")
    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
