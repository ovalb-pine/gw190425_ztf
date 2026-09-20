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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_quadratic


FWHM_HEADER_KEYS = ("SEEING", "FWHM", "FWHM_PX", "PSF_FWHM")

_NAME_PAT = re.compile(
    r"(?P<object_id>\d+)_RA(?P<ra>[-+\d\.]+)_DEC(?P<dec>[-+\d\.]+)_"
    r"(?P<filter>z[gri])_(?P<filefracday>\d+)__ztf_\d+_"
    r"(?P<field>\d{6})_(?P<filter2>z[gri])_c(?P<ccdid>\d+)_o_q(?P<qid>\d)_scimrefdiffimg"
)


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


def _sample_null_apertures(
    data: np.ndarray,
    seed_x: float,
    seed_y: float,
    ap_r: float,
    fwhm_px: float,
    background_median: float,
    aperture_area_px: float,
    n_samples: int = 100,
    min_sep_fwhm: float = 4.0,
    max_sep_fwhm: float = 12.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build an empirical null distribution of net flux at off-source positions.

    Critically, each null measurement is run through the *same* biased procedure
    as the on-source one: seed a position, let ``_conservative_centroid`` snap to
    the local peak, then do aperture photometry there. Because ``centroid_quadratic``
    always finds *some* local maximum, comparing the on-source flux to a naive
    Gaussian sigma is unfair -- it ignores that the position itself was chosen by
    searching for a peak. Sampling many off-source, source-free locations with the
    identical peak-searching procedure captures that selection bias directly, so
    the resulting distribution is a fair baseline: "how much net flux does this
    same peak-picking algorithm manufacture out of pure background alone?"
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = data.shape
    pad = ap_r * 2.0
    fluxes = []
    attempts = 0
    max_attempts = n_samples * 20

    while len(fluxes) < n_samples and attempts < max_attempts:
        attempts += 1
        angle = rng.uniform(0.0, 2.0 * np.pi)
        sep = rng.uniform(min_sep_fwhm, max_sep_fwhm) * fwhm_px
        x = seed_x + sep * np.cos(angle)
        y = seed_y + sep * np.sin(angle)
        if not (pad <= x <= (w - 1 - pad) and pad <= y <= (h - 1 - pad)):
            continue

        cx, cy = _conservative_centroid(data, x, y, fwhm_px, max_shift_fwhm=3.0)
        aperture = CircularAperture([(cx, cy)], r=ap_r)
        ap_tbl = aperture_photometry(data, aperture)
        aperture_sum = float(np.asarray(ap_tbl["aperture_sum"])[0])
        fluxes.append(aperture_sum - background_median * aperture_area_px)

    return np.asarray(fluxes, dtype=float)


def _empirical_significance(
    net_flux: float,
    null_net_flux: np.ndarray,
    sigma_clip_sigma: float = 3.0,
    sigma_clip_maxiters: int = 5,
) -> tuple[float, float, float, float, int]:
    """Compare an on-source net flux to an empirical off-source null distribution.

    Returns (null_median, null_std, snr_empirical, p_value_empirical, n_used).
    The null stats are themselves sigma-clipped so that a stray real neighbor
    landing in one of the null apertures doesn't inflate the spread.
    ``p_value_empirical`` is the fraction of null draws that equal or exceed the
    observed flux -- a distribution-free check that doesn't assume the
    (peak-picking-biased) null is Gaussian.
    """
    n = int(np.isfinite(null_net_flux).sum())
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan, n

    finite_null = null_net_flux[np.isfinite(null_net_flux)]
    _, null_median, null_std = sigma_clipped_stats(
        finite_null, sigma=sigma_clip_sigma, maxiters=sigma_clip_maxiters
    )

    if np.isfinite(null_std) and null_std > 0:
        snr_empirical = float((net_flux - null_median) / null_std)
    else:
        snr_empirical = np.nan

    p_value_empirical = float(np.mean(finite_null >= net_flux))

    return float(null_median), float(null_std), snr_empirical, p_value_empirical, n


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


def parse_image_name(name: str):
    """Parse the standard ZTF diff-image filename into key metadata."""
    match = _NAME_PAT.search(name)
    if not match:
        return None
    return {
        "object_id": match.group("object_id"),
        "ra": float(match.group("ra")),
        "dec": float(match.group("dec")),
        "filter": match.group("filter"),
        "filefracday": match.group("filefracday"),
        "field": int(match.group("field")),
        "ccdid": int(match.group("ccdid")),
        "qid": int(match.group("qid")),
    }


def _extract_zero_point(header: dict) -> float:
    for key in ("MAGZP", "MAGZERO", "ZP", "ZEROPT", "ZEROPNT"):
        try:
            zp = float(header.get(key))
        except Exception:
            continue
        if np.isfinite(zp):
            return float(zp)
    return np.nan


@dataclass
class DifferenceImageSNR:
    image_path: str
    source_x_px: float
    source_y_px: float
    object_id: str
    ra: float
    dec: float
    filter: str
    filefracday: str
    field: int
    ccdid: int
    qid: int
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
    null_flux_median: float
    null_flux_std: float
    n_null_apertures: int
    snr_empirical: float
    p_value_empirical: float
    significance_tier: str
    mag: float
    mag_err: float
    upper_limit: float
    detection: bool
    zero_point: float
    pixel_scale_arcsec_per_px: float


def process_difference_image(
    image_path,
    *,
    background_box_factor: float = 6.0,
    sigma: float = 3.0,
    maxiters: int = 5,
    min_valid_pixel: float = -5000.0,
    center_x: float | None = None,
    center_y: float | None = None,
    n_null_apertures: int = 100,
    n_null_apertures_refine: int = 400,
    refine_near_boundary: bool = True,
    refine_low: float = 2.0,
    refine_high: float = 6.0,
    marginal_snr_threshold: float = 3.0,
    secure_snr_threshold: float = 5.0,
    null_min_sep_fwhm: float = 4.0,
    null_max_sep_fwhm: float = 12.0,
    null_seed: int | None = None,
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

    if center_x is None or not np.isfinite(center_x):
        center_x = (data.shape[1] - 1) / 2.0
    if center_y is None or not np.isfinite(center_y):
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

    # Empirical null test: does this same peak-picking procedure manufacture
    # comparable net flux out of pure background at off-source positions? This
    # is what actually decides `detection` -- the naive `snr` above compares to
    # a background sigma that doesn't account for source_x_px/source_y_px having
    # been chosen by searching for a peak, and is kept only as a diagnostic field.
    rng = np.random.default_rng(null_seed) if null_seed is not None else np.random.default_rng(
        abs(hash(str(p))) % (2**32)
    )

    def _run_null(n_samples: int, seed_rng: np.random.Generator):
        flux = _sample_null_apertures(
            data,
            center_x,
            center_y,
            aperture_radius_px,
            fwhm_px,
            background_median,
            aperture_area_px,
            n_samples=n_samples,
            min_sep_fwhm=null_min_sep_fwhm,
            max_sep_fwhm=null_max_sep_fwhm,
            rng=seed_rng,
        )
        return _empirical_significance(net_flux, flux, sigma_clip_sigma=sigma, sigma_clip_maxiters=maxiters)

    null_flux_median, null_flux_std, snr_empirical, p_value_empirical, n_null_apertures_used = _run_null(
        n_null_apertures, rng
    )

    # The 3-5 sigma band is exactly the science-critical regime for this pipeline
    # (faint transients), and it's also exactly where a modest null sample (~100)
    # has the most sampling uncertainty on its own std -- a borderline call here
    # could flip either way just from null-sampling noise, not from the actual
    # source. So: for candidates landing near this boundary, spend extra null
    # draws to pin down snr_empirical more precisely, rather than paying that
    # cost uniformly on every image (most of which aren't borderline at all).
    if refine_near_boundary and np.isfinite(snr_empirical) and (
        refine_low <= snr_empirical <= refine_high
    ):
        null_flux_median, null_flux_std, snr_empirical, p_value_empirical, n_null_apertures_used = _run_null(
            n_null_apertures_refine, rng
        )

    if np.isfinite(snr_empirical):
        if snr_empirical >= secure_snr_threshold:
            significance_tier = "secure"
        elif snr_empirical >= marginal_snr_threshold:
            significance_tier = "marginal"
        else:
            significance_tier = "not_significant"
    else:
        # Null sampling failed (e.g. too close to a frame edge to place off-source
        # apertures) -- fall back to the naive snr but flag it as unverified so
        # downstream code can treat it with appropriate caution rather than
        # silently trusting a metric known to have a peak-selection bias.
        significance_tier = "undetermined_fallback_naive"

    info = parse_image_name(p.name)
    object_id = info["object_id"] if info else ""
    ra = float(info["ra"]) if info else np.nan
    dec = float(info["dec"]) if info else np.nan
    filt = info["filter"] if info else ""
    filefracday = info["filefracday"] if info else ""
    field = int(info["field"]) if info else -1
    ccdid = int(info["ccdid"]) if info else -1
    qid = int(info["qid"]) if info else -1
    zero_point = _extract_zero_point(header)

    # Detection now rests on the bias-corrected empirical significance alone.
    # (Requiring the naive `snr` to also pass added no real protection -- it's
    # biased in the same direction as a true source -- and only risked rejecting
    # genuine faint transients in the 3-5 sigma band this pipeline cares about.)
    if significance_tier == "undetermined_fallback_naive":
        is_detection = bool(np.isfinite(snr) and snr >= float(marginal_snr_threshold))
        effective_snr = snr
    else:
        is_detection = bool(snr_empirical >= float(marginal_snr_threshold))
        effective_snr = snr_empirical

    if is_detection and np.isfinite(zero_point) and np.isfinite(net_flux) and net_flux > 0:
        mag = float(zero_point - 2.5 * np.log10(net_flux))
        mag_err = float(1.0857362047581294 * flux_err / net_flux) if np.isfinite(flux_err) and flux_err > 0 else np.nan
        upper_limit = np.nan
        detection = True
    elif np.isfinite(zero_point) and np.isfinite(flux_err) and flux_err > 0:
        upper_flux = float(sigma) * flux_err
        mag = np.nan
        mag_err = np.nan
        upper_limit = float(zero_point - 2.5 * np.log10(max(upper_flux, 1e-12)))
        detection = False
    else:
        mag = np.nan
        mag_err = np.nan
        upper_limit = np.nan
        detection = False

    return DifferenceImageSNR(
        image_path=str(p),
        source_x_px=float(source_x_px),
        source_y_px=float(source_y_px),
        object_id=str(object_id),
        ra=float(ra),
        dec=float(dec),
        filter=str(filt),
        filefracday=str(filefracday),
        field=int(field),
        ccdid=int(ccdid),
        qid=int(qid),
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
        null_flux_median=float(null_flux_median) if np.isfinite(null_flux_median) else np.nan,
        null_flux_std=float(null_flux_std) if np.isfinite(null_flux_std) else np.nan,
        n_null_apertures=int(n_null_apertures_used),
        snr_empirical=float(snr_empirical) if np.isfinite(snr_empirical) else np.nan,
        p_value_empirical=float(p_value_empirical) if np.isfinite(p_value_empirical) else np.nan,
        significance_tier=str(significance_tier),
        mag=float(mag) if np.isfinite(mag) else np.nan,
        mag_err=float(mag_err) if np.isfinite(mag_err) else np.nan,
        upper_limit=float(upper_limit) if np.isfinite(upper_limit) else np.nan,
        detection=bool(detection),
        zero_point=float(zero_point) if np.isfinite(zero_point) else np.nan,
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