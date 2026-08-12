"""SNR estimation for ZTF difference images using quadratic centroiding.

This module measures SNR on difference images by:
- reading the per-image seeing/FWHM value from the FITS header,
- using an aperture radius of 2 x FWHM,
- centering the source with ``photutils.centroids.centroid_quadratic``,
- estimating background noise with sigma clipping instead of an annulus.

The main entry point is :func:`process_difference_image`.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_quadratic


FWHM_HEADER_KEYS = ("SEEING", "FWHM", "FWHM_PX", "PSF_FWHM")


def first_2d_data_and_header(img_path: Path):
    """Return the first 2D image plane and its header from a FITS file."""
    with fits.open(img_path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if isinstance(data, np.ndarray) and data.ndim == 2:
                return np.asarray(data, dtype=float), hdu.header
    raise ValueError(f"No 2D image data found in {img_path}")


def pixel_scale_arcsec(header):
    """Estimate pixel scale in arcsec/pixel from FITS WCS keywords."""
    if header is not None:
        cd11 = header.get("CD1_1")
        cd12 = header.get("CD1_2")
        cd21 = header.get("CD2_1")
        cd22 = header.get("CD2_2")
        if all(v is not None for v in [cd11, cd12, cd21, cd22]):
            sx = np.sqrt(cd11**2 + cd21**2)
            sy = np.sqrt(cd12**2 + cd22**2)
            return float(np.mean([sx, sy]) * 3600.0)

        cdelt1 = header.get("CDELT1")
        cdelt2 = header.get("CDELT2")
        if cdelt1 is not None and cdelt2 is not None:
            return float(np.mean([abs(cdelt1), abs(cdelt2)]) * 3600.0)

    return 1.01


def _as_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(int(value), int(minimum))
    return value if value % 2 == 1 else value + 1


def _safe_center_crop(data: np.ndarray, cx: float, cy: float, size: int) -> tuple[np.ndarray, int, int]:
    size = _odd_at_least(size, 5)
    half = size // 2
    x0 = int(round(cx))
    y0 = int(round(cy))
    x1 = max(0, x0 - half)
    x2 = min(data.shape[1], x0 + half + 1)
    y1 = max(0, y0 - half)
    y2 = min(data.shape[0], y0 + half + 1)
    return np.asarray(data[y1:y2, x1:x2], dtype=float), x1, y1


def _quadratic_centroid(data: np.ndarray, guess_x: float, guess_y: float, box_size: int) -> tuple[float, float]:
    """Centroid the source using a local stamp around the brightest peak."""
    stamp, x1, y1 = _safe_center_crop(data, guess_x, guess_y, box_size)
    if stamp.size == 0:
        return float(guess_x), float(guess_y)

    stamp = np.asarray(stamp, dtype=float)
    finite = np.isfinite(stamp)
    if not finite.any():
        return float(guess_x), float(guess_y)

    stamp = stamp.copy()
    stamp[~finite] = np.nanmedian(stamp[finite])

    try:
        cx, cy = centroid_quadratic(stamp)
    except Exception:
        return float(guess_x), float(guess_y)

    if not np.isfinite(cx) or not np.isfinite(cy):
        return float(guess_x), float(guess_y)

    return float(x1 + cx), float(y1 + cy)


def _conservative_centroid(
    data: np.ndarray, center_x: float, center_y: float, fwhm_px: float, max_shift_fwhm: float = 3.0
) -> tuple[float, float]:
    """
    Quadratic centroid in a small stamp around the frame center with sanity checks.

    The stamp is limited to ~2*FWHM to avoid grabbing dipole subtraction artifacts.
    If the centroid shifts by more than `max_shift_fwhm * fwhm_px` from the center,
    the function falls back to returning the original center.
    """
    box_size = _odd_at_least(int(round(4.0 * float(fwhm_px))), 11)
    stamp, x1, y1 = _safe_center_crop(data, center_x, center_y, box_size)

    if stamp.size == 0:
        return float(center_x), float(center_y)

    stamp = stamp.copy().astype(float)
    finite = np.isfinite(stamp)
    if not finite.any():
        return float(center_x), float(center_y)

    stamp[~finite] = np.nanmedian(stamp[finite])

    # Only positive signal — ignore negative dipole lobes from subtraction artifacts
    stamp_pos = np.clip(stamp, 0.0, None)
    if stamp_pos.max() == 0:
        return float(center_x), float(center_y)

    try:
        cx, cy = centroid_quadratic(stamp_pos)
    except Exception:
        return float(center_x), float(center_y)

    if not np.isfinite(cx) or not np.isfinite(cy):
        return float(center_x), float(center_y)

    result_x = float(x1 + cx)
    result_y = float(y1 + cy)

    shift = float(np.sqrt((result_x - float(center_x)) ** 2 + (result_y - float(center_y)) ** 2))
    if shift > float(max_shift_fwhm) * float(fwhm_px):
        return float(center_x), float(center_y)

    return float(result_x), float(result_y)


def _global_sigma_clipped_background_stats(
    data: np.ndarray,
    cx: float,
    cy: float,
    ap_r: float,
    sigma: float = 3.0,
    maxiters: int = 5,
    source_mask_scale: float = 1.5,
    min_valid_pixel: float = -5000.0,
):
    """Compute global background stats with a circular source mask around the target.
    
    Parameters
    ----------
    min_valid_pixel : float
        Exclude pixels below this value (e.g., edge padding in ZTF cutouts).
    """
    values = np.asarray(data, dtype=float)
    yy, xx = np.indices(values.shape, dtype=float)
    source_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (float(source_mask_scale) * float(ap_r)) ** 2

    # Mask source region, NaN pixels, and edge padding (pixels below threshold)
    nan_mask = ~np.isfinite(values)
    edge_mask = values < float(min_valid_pixel)
    combined_mask = source_mask | nan_mask | edge_mask

    mean, median, std = sigma_clipped_stats(values, mask=combined_mask, sigma=sigma, maxiters=maxiters)
    return float(mean), float(median), float(std)


def _sigma_clipped_background_stats(pixels: np.ndarray, sigma: float = 3.0, maxiters: int = 5):
    """Compute background mean/median/std after sigma clipping."""
    pixels = np.asarray(pixels, dtype=float)
    pixels = pixels[np.isfinite(pixels)]
    if pixels.size == 0:
        return np.nan, np.nan, np.nan

    mean, median, std = sigma_clipped_stats(pixels, sigma=sigma, maxiters=maxiters)
    return float(mean), float(median), float(std)


def _extract_fwhm_px(header, keys: Iterable[str] = FWHM_HEADER_KEYS) -> tuple[float, str]:
    for key in keys:
        value = header.get(key) if header is not None else None
        fwhm_px = _as_float(value)
        if np.isfinite(fwhm_px) and fwhm_px > 0:
            return float(fwhm_px), key
    raise ValueError(f"Could not find a valid FWHM/seeing value in header keys: {', '.join(keys)}")


@dataclass
class DifferenceImageSNR:
    image_path: str
    source_x_px: float
    source_y_px: float
    fwhm_px: float
    fwhm_header_key: str
    aperture_radius_px: float
    aperture_sum: float
    background_mean: float
    background_median: float
    background_std: float
    aperture_area_px: float
    net_flux: float
    flux_err: float
    snr: float
    pixel_scale_arcsec_per_px: float


def process_difference_image(
    image_path,
    *,
    background_box_factor: float = 6.0,
    sigma: float = 3.0,
    maxiters: int = 5,
    min_valid_pixel: float = -5000.0,
    **kwargs,
):
    """Measure SNR for one ZTF difference image.

    Parameters
    ----------
    image_path:
        Path to the FITS difference image.
    background_box_factor:
        Legacy parameter kept for backward compatibility; background is estimated globally.
    sigma, maxiters:
        Sigma-clipping controls passed to ``SigmaClip`` and ``sigma_clipped_stats``.
    min_valid_pixel:
        Exclude pixels below this value from background statistics (e.g., edge padding).
    """
    p = Path(image_path)
    data, header = first_2d_data_and_header(p)

    fwhm_px, fwhm_key = _extract_fwhm_px(header)
    aperture_radius_px = 2.0 * fwhm_px

    center_x = (data.shape[1] - 1) / 2.0
    center_y = (data.shape[0] - 1) / 2.0
    source_x_px, source_y_px = _conservative_centroid(
        data, center_x, center_y, fwhm_px, max_shift_fwhm=3.0
    )

    aperture = CircularAperture([(source_x_px, source_y_px)], r=aperture_radius_px)
    ap_tbl = aperture_photometry(data, aperture)
    aperture_sum = float(np.asarray(ap_tbl["aperture_sum"])[0])
    aperture_area_px = float(aperture.area)

    # Use a global sigma-clipped background with only the source neighborhood masked.
    # This keeps background noise estimates stable and independent of local box size choices.
    background_mean, background_median, background_std = _global_sigma_clipped_background_stats(
        data,
        source_x_px,
        source_y_px,
        aperture_radius_px,
        sigma=sigma,
        maxiters=maxiters,
        min_valid_pixel=min_valid_pixel,
    )

    if not np.isfinite(background_median):
        background_mean, background_median, background_std = _sigma_clipped_background_stats(
            np.asarray(data, dtype=float).ravel(), sigma=sigma, maxiters=maxiters
        )

    net_flux = aperture_sum - background_median * aperture_area_px
    flux_err = float(np.sqrt(aperture_area_px) * background_std) if np.isfinite(background_std) else np.nan
    snr = float(net_flux / flux_err) if np.isfinite(flux_err) and flux_err > 0 else np.nan

    return DifferenceImageSNR(
        image_path=str(p),
        source_x_px=float(source_x_px),
        source_y_px=float(source_y_px),
        fwhm_px=float(fwhm_px),
        fwhm_header_key=fwhm_key,
        aperture_radius_px=float(aperture_radius_px),
        aperture_sum=float(aperture_sum),
        background_mean=float(background_mean),
        background_median=float(background_median),
        background_std=float(background_std),
        aperture_area_px=float(aperture_area_px),
        net_flux=float(net_flux),
        flux_err=float(flux_err),
        snr=float(snr),
        pixel_scale_arcsec_per_px=float(pixel_scale_arcsec(header)),
    )


def process_difference_images(image_paths, **kwargs) -> pd.DataFrame:
    """Process many images and return a DataFrame of SNR measurements."""
    rows = []
    for image_path in image_paths:
        try:
            result = process_difference_image(image_path, **kwargs)
            d = result.__dict__.copy()
            # Mark successful processing so callers can filter by status
            d.setdefault("status", "ok")
            rows.append(d)
        except Exception as exc:
            rows.append(
                {
                    "image_path": str(image_path),
                    "status": f"error: {exc}",
                }
            )
    return pd.DataFrame(rows)


def _collect_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if any(ch in raw for ch in ["*", "?", "["]):
            paths.extend(sorted(Path(match) for match in glob.glob(raw)))
        elif p.is_dir():
            paths.extend(sorted(p.glob("*.fits")))
            paths.extend(sorted(p.glob("*.fits.fz")))
        else:
            paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure SNR on ZTF difference images using quadratic centroiding.")
    parser.add_argument("images", nargs="+", help="One or more FITS images, directories, or glob patterns")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--background-box-factor", type=float, default=6.0, help="Unused; background is now estimated globally (kept for backward compatibility)")
    parser.add_argument("--sigma", type=float, default=3.0, help="Sigma clipping threshold")
    parser.add_argument("--maxiters", type=int, default=5, help="Maximum sigma-clipping iterations")
    parser.add_argument("--min-valid-pixel", type=float, default=-5000.0, help="Exclude pixels below this value from background stats (e.g., edge padding)")
    args = parser.parse_args(argv)

    paths = _collect_paths(args.images)
    if not paths:
        raise SystemExit("No input FITS files found")

    df = process_difference_images(
        paths,
        background_box_factor=args.background_box_factor,
        sigma=args.sigma,
        maxiters=args.maxiters,
        min_valid_pixel=args.min_valid_pixel,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
    else:
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())