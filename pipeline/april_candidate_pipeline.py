"""End-to-end April candidate pipeline for ZTF cutouts."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import yaml
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("PyYAML is required to read config.yaml") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))

from braai_batch import build_triplet_batch, load_braai_model, predict_braai_batch
from ztf_downloads.fetch_sci_ref_for_snr import _build_exact_coord_index, compute_host_offset
from ztf_downloads.snr_photometry_quadratic import process_difference_image, process_difference_images
from ztf_downloads.blind_detect import collect_full_diff_index, parse_full_diff_name, stage1_blind_search
from ztf_downloads.ztf_download import build_cutout_url, build_sci_url, build_ref_url, download_file


def load_config(config_path: str | Path = "config.yaml") -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return data


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return None


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def normalize_object_id(value) -> str:
    text = re.sub(r"\.0*$", "", str(value).strip())
    if not text:
        return ""
    if "e" in text.lower() or "." in text:
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text


def _extract_object_id(value) -> str:
    if value is None:
        return ""
    text = normalize_object_id(value)
    if not text:
        return ""
    match = re.search(r"(\d{8,})", text)
    return match.group(1) if match else text


def _ensure_object_id_column(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a catalog with a normalized object_id column.

    Some input CSVs (for example the raw final-class table) do not have an explicit
    object_id column. In that case, synthesize one from the best available ID field;
    if none exists, fall back to the row number as a stable surrogate identifier.
    """
    if df is None:
        return pd.DataFrame()
    work = df.copy()
    if "object_id" in work.columns:
        work["object_id"] = work["object_id"].astype(str).map(normalize_object_id)
        return work

    for candidate in ("objid", "objID", "ID", "id", "objectid"):
        if candidate in work.columns:
            work["object_id"] = work[candidate].map(_extract_object_id)
            work["object_id"] = work["object_id"].astype(str).map(normalize_object_id)
            return work

    work.insert(0, "object_id", [str(i + 1) for i in range(len(work))])
    return work


def _build_final_rownum_map(final_class_df: pd.DataFrame) -> dict:
    """Return a mapping from normalized object id -> 1-based row number string.

    This lets downstream steps name files/directories by the galaxy row number
    (like the catalog2 output) while keeping the original object_id values
    in dataframes.
    """
    mapping: dict = {}
    if final_class_df is None:
        return mapping
    # reset_index to ensure contiguous 0..N-1 ordering regardless of original index
    for idx, row in final_class_df.reset_index(drop=True).iterrows():
        candidate = None
        for col in ("objID_SDSS-DR16", "object_id", "objid", "id"):
            if col in row:
                candidate = row.get(col)
                break
        if candidate is None or pd.isna(candidate):
            continue
        oid = _extract_object_id(candidate)
        if oid:
            mapping[oid] = str(int(idx) + 1)
    return mapping


def _build_object_coordinate_index(final_class_df: pd.DataFrame | None) -> dict[str, tuple[float, float]]:
    """Index catalog coordinates by both catalog object ID and row-number ID."""
    if final_class_df is None:
        return {}
    index = {str(key): value for key, value in _build_exact_coord_index(final_class_df).items()}
    ra_columns = ("ra_fin", "RA_fin", "ra_SDSS-DR16", "RA", "ra")
    dec_columns = ("dec_fin", "DEC_fin", "dec_SDSS-DR16", "DEC", "dec")
    ra_column = next((column for column in ra_columns if column in final_class_df.columns), None)
    dec_column = next((column for column in dec_columns if column in final_class_df.columns), None)
    if ra_column is None or dec_column is None:
        return index
    for row_number, (_, row) in enumerate(final_class_df.reset_index(drop=True).iterrows(), start=1):
        ra = _safe_float(row.get(ra_column))
        dec = _safe_float(row.get(dec_column))
        if np.isfinite(ra) and np.isfinite(dec):
            index.setdefault(str(row_number), (ra, dec))
    return index


def _norm_key_name(name: str) -> str:
    if "__ztf_" in name:
        return name.split("__", 1)[1]
    return name


def parse_diff_name(name: str):
    pattern = re.compile(
        r"(?P<object_id>B?\d+)_RA(?P<ra>[-+\d\.]+)_DEC(?P<dec>[-+\d\.]+)_"
        r"(?P<filter>z[gri])_(?P<filefracday>\d+)__ztf_\d+_"
        r"(?P<field>\d{6})_(?P<filter2>z[gri])_c(?P<ccdid>\d+)_o_q(?P<qid>\d)_scimrefdiffimg"
    )
    match = pattern.search(name)
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


def _parse_filefracday_datetime(filefracday: str | None):
    """Parse the ZTF filefracday token as YYYYMMDD plus fractional day.

    Examples: 20220219 -> 2022-02-19 00:00:00
              20220219536262 -> 2022-02-19 12:52:13.036800
    """
    if filefracday is None:
        return pd.NaT
    ffd = str(filefracday).strip()
    if not ffd:
        return pd.NaT
    if len(ffd) < 8:
        return pd.NaT

    base = pd.to_datetime(ffd[:8], errors="coerce", format="%Y%m%d")
    if pd.isna(base):
        return pd.NaT

    frac_text = ffd[8:]
    if not frac_text:
        return base
    if frac_text.isdigit():
        try:
            frac = float(f"0.{frac_text}")
            return base + pd.to_timedelta(frac * 86400.0, unit="s")
        except Exception:
            pass
    return base


def resolve_cli_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    repo_root = Path(__file__).resolve().parent
    candidates = [Path.cwd() / path, repo_root / path, repo_root / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def build_file_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file():
            index.setdefault(_norm_key_name(path.name), path)
    return index


def collect_diff_index(diff_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not diff_root.exists():
        return pd.DataFrame(rows)

    for path in sorted(diff_root.rglob("*.fits")) + sorted(diff_root.rglob("*.fits.fz")):
        parsed = parse_diff_name(path.name)
        if parsed is None:
            rows.append({"image_path": str(path), "image_name": path.name, "parse_status": "unparsed"})
            continue
        rows.append(
            {
                "image_path": str(path),
                "image_name": path.name,
                "parse_status": "ok",
                "object_id": normalize_object_id(parsed["object_id"]),
                "filefracday": parsed["filefracday"],
                "filter": parsed["filter"],
                "field": parsed["field"],
                "ccdid": parsed["ccdid"],
                "qid": parsed["qid"],
                "ra": parsed["ra"],
                "dec": parsed["dec"],
            }
        )
    index_df = pd.DataFrame(rows)
    if index_df.empty or "parse_status" not in index_df.columns:
        return index_df

    # The same physical exposure (object/field/ccd/quadrant/filter/filefracday)
    # can end up saved under multiple filenames if it was downloaded more than
    # once with a slightly different cutout center (e.g. before download centers
    # were pinned to the fixed galaxy catalog position). Keep exactly one file
    # per exposure so it isn't measured/counted more than once downstream.
    # Sorting by image_path first makes the kept survivor deterministic.
    ok_rows = index_df[index_df["parse_status"] == "ok"].sort_values("image_path")
    dedup_keys = ["object_id", "field", "ccdid", "qid", "filefracday", "filter"]
    ok_rows = ok_rows.drop_duplicates(subset=dedup_keys, keep="first")
    other_rows = index_df[index_df["parse_status"] != "ok"]
    return pd.concat([ok_rows, other_rows], ignore_index=True)


def measure_fwhm_from_image(
    diff_path: Path, center_x: float, center_y: float, box_size: int = 21
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Measure FWHM of the detection in the difference image at the given pixel position.

    Returns (FWHM, FWHM uncertainty, fit residual RMS, reduced chi-square,
    PSF-vs-spike delta chi-square, PSF amplitude SNR, source-vs-background
    delta chi-square, and PSF residual RMS in background-noise units.
    """
    try:
        from astropy.io import fits
        from astropy.modeling import models, fitting
        import numpy as np

        with fits.open(diff_path, memmap=False) as hdul:
            image_hdu = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    image_hdu = hdu
                    break
            if image_hdu is None:
                return (np.nan,) * 8
            raw_data = image_hdu.data
            if np.ma.isMaskedArray(raw_data):
                data = np.asarray(raw_data.filled(np.nan), dtype=float)
                valid_mask = ~np.ma.getmaskarray(raw_data)
            else:
                data = np.asarray(raw_data, dtype=float)
                valid_mask = np.ones(data.shape, dtype=bool)

            # Respect a same-sized FITS mask/quality extension when present.
            for hdu in hdul:
                name = str(getattr(hdu, "name", "")).upper()
                if (
                    hdu is not image_hdu
                    and hdu.data is not None
                    and getattr(hdu.data, "shape", None) == data.shape
                    and any(token in name for token in ("MASK", "QUALITY", "DQ"))
                ):
                    valid_mask &= np.asarray(hdu.data) == 0
            data[~valid_mask] = np.nan

        # Crop region around the detection
        half = box_size // 2
        cy, cx = int(round(center_y)), int(round(center_x))
        y1 = max(0, cy - half)
        y2 = min(data.shape[0], cy + half + 1)
        x1 = max(0, cx - half)
        x2 = min(data.shape[1], cx + half + 1)
        cutout = data[y1:y2, x1:x2]

        if cutout.size == 0 or np.count_nonzero(np.isfinite(cutout)) < 5:
            return (np.nan,) * 8

        # Keep the Gaussian fixed on the stage-1 centroid; only fit its shape.
        yy, xx = np.indices(cutout.shape)

        # Fit after subtracting a local background. Convert the exact subpixel
        # centroid into cutout coordinates rather than recentering on a pixel peak.
        background = float(np.nanmedian(cutout))
        fit_data = cutout - background
        fit_center_x = float(center_x) - x1
        fit_center_y = float(center_y) - y1

        finite_fit_data = fit_data[np.isfinite(fit_data)]
        gauss = models.Gaussian2D(
            amplitude=float(np.nanmax(finite_fit_data)),
            x_mean=fit_center_x,
            y_mean=fit_center_y,
            x_stddev=2.0,
            y_stddev=2.0,
        )
        gauss.x_mean.fixed = True
        gauss.y_mean.fixed = True

        fitter = fitting.LevMarLSQFitter()
        fitted = fitter(gauss, xx, yy, fit_data, filter_non_finite=True)
        model_data = fitted(xx, yy)
        residuals = fit_data - model_data
        finite_residuals = residuals[np.isfinite(residuals)]
        fit_rms = (
            float(np.sqrt(np.mean(np.square(finite_residuals))))
            if finite_residuals.size
            else np.nan
        )
        yy_center, xx_center = np.indices(cutout.shape)
        background_mask = (
            np.square(xx_center - fit_center_x) + np.square(yy_center - fit_center_y) > 25.0
        )
        background_values = fit_data[background_mask & np.isfinite(fit_data)]
        if background_values.size >= 5:
            noise = 1.4826 * np.median(
                np.abs(background_values - np.median(background_values))
            )
            if not np.isfinite(noise) or noise <= 0:
                noise = float(np.std(background_values, ddof=1))
        else:
            noise = np.nan
        delta_chi2 = np.nan
        psf_amplitude_snr = np.nan
        source_delta_chi2 = np.nan
        psf_residual_rms_sigma = np.nan
        if np.isfinite(noise) and noise > 0 and finite_residuals.size:
            finite_mask = np.isfinite(fit_data)
            observed = fit_data[finite_mask]
            psf_values = model_data[finite_mask]
            psf_chi2 = float(np.sum(np.square((observed - psf_values) / noise)))
            spike_mask = np.zeros(cutout.shape, dtype=float)
            nearest_x = int(np.clip(round(fit_center_x), 0, cutout.shape[1] - 1))
            nearest_y = int(np.clip(round(fit_center_y), 0, cutout.shape[0] - 1))
            spike_mask[nearest_y, nearest_x] = 1.0
            spike_values = spike_mask[finite_mask]
            spike_amplitude = float(
                np.sum(spike_values * observed) / np.sum(np.square(spike_values))
            )
            spike_model = spike_amplitude * spike_values
            spike_chi2 = float(np.sum(np.square((observed - spike_model) / noise)))
            delta_chi2 = spike_chi2 - psf_chi2
            background_chi2 = float(np.sum(np.square(observed / noise)))
            source_delta_chi2 = background_chi2 - psf_chi2
            psf_residual_rms_sigma = float(
                np.sqrt(np.mean(np.square((observed - psf_values) / noise)))
            )
        degrees_of_freedom = finite_residuals.size - 4
        reduced_chi2 = (
            float(np.sum(np.square(finite_residuals / noise)) / degrees_of_freedom)
            if np.isfinite(noise) and noise > 0 and degrees_of_freedom > 0
            else np.nan
        )

        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.35482 * sigma
        fwhm_x = 2.35482 * abs(float(fitted.x_stddev.value))
        fwhm_y = 2.35482 * abs(float(fitted.y_stddev.value))
        fwhm = np.mean([fwhm_x, fwhm_y])

        covariance = fitter.fit_info.get("param_cov")
        fwhm_error = np.nan
        if covariance is not None and np.shape(covariance) == (4, 4):
            # With the two centroid parameters fixed, free-parameter order is
            # amplitude, x_stddev, y_stddev, theta.
            sign_x = np.sign(float(fitted.x_stddev.value))
            sign_y = np.sign(float(fitted.y_stddev.value))
            width_variance = (2.35482 / 2.0) ** 2 * (
                covariance[1, 1]
                + covariance[2, 2]
                + 2.0 * sign_x * sign_y * covariance[1, 2]
            )
            if np.isfinite(width_variance) and width_variance >= 0:
                fwhm_error = float(np.sqrt(width_variance))
            amplitude_variance = covariance[0, 0]
            if np.isfinite(amplitude_variance) and amplitude_variance > 0:
                psf_amplitude_snr = float(
                    abs(float(fitted.amplitude.value)) / np.sqrt(amplitude_variance)
                )

        if np.isfinite(fwhm) and fwhm > 0:
            return (
                float(fwhm),
                fwhm_error,
                fit_rms,
                reduced_chi2,
                delta_chi2,
                psf_amplitude_snr,
                source_delta_chi2,
                psf_residual_rms_sigma,
            )
        return (
            np.nan,
            np.nan,
            fit_rms,
            reduced_chi2,
            delta_chi2,
            psf_amplitude_snr,
            source_delta_chi2,
            psf_residual_rms_sigma,
        )
    except Exception:
        return (np.nan,) * 8

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


def _load_image_and_wcs_uncached(path):
    with fits.open(path, memmap=False) as hdul:
        image_hdu = None
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and hdu.data.ndim == 2:
                image_hdu = hdu
                break
        if image_hdu is None:
            raise ValueError(f"No 2D image found in {path}")

        raw_data = image_hdu.data
        if np.ma.isMaskedArray(raw_data):
            data = np.asarray(raw_data.filled(np.nan), dtype=np.float32)
            valid_mask = ~np.ma.getmaskarray(raw_data)
        else:
            data = np.asarray(raw_data, dtype=np.float32)
            valid_mask = np.ones(data.shape, dtype=bool)

        # Keep FITS mask/quality information so it survives triplet creation.
        for hdu in hdul:
            name = str(getattr(hdu, "name", "")).upper()
            if (
                hdu is not image_hdu
                and hdu.data is not None
                and getattr(hdu.data, "shape", None) == data.shape
                and any(token in name for token in ("MASK", "QUALITY", "DQ"))
            ):
                valid_mask &= np.asarray(hdu.data) == 0

        valid_mask &= np.isfinite(data)
        data[~valid_mask] = np.nan
        return data, WCS(image_hdu.header), valid_mask


_load_image_and_wcs = lru_cache(maxsize=6)(_load_image_and_wcs_uncached)


def _sky_coordinates_from_pixel(path: Path, x_px: float, y_px: float) -> tuple[float, float] | None:
    """Convert a 2D image pixel position to sky coordinates using its WCS."""
    if not np.isfinite(x_px) or not np.isfinite(y_px):
        return None
    try:
        _, wcs, _ = _load_image_and_wcs(path)
        image_wcs = wcs.celestial if getattr(wcs, "naxis", 2) > 2 else wcs
        ra, dec = image_wcs.all_pix2world(float(x_px), float(y_px), 0)
        ra = float(np.asarray(ra).reshape(-1)[0])
        dec = float(np.asarray(dec).reshape(-1)[0])
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    except Exception:
        pass
    return None


def _pixel_coordinates_from_sky(path: Path, ra: float, dec: float) -> tuple[float, float] | None:
    """Convert a sky position to pixels in a specific 2D FITS image."""
    if not np.isfinite(ra) or not np.isfinite(dec):
        return None
    try:
        _, wcs, _ = _load_image_and_wcs(path)
        image_wcs = wcs.celestial if getattr(wcs, "naxis", 2) > 2 else wcs
        x_px, y_px = image_wcs.all_world2pix(float(ra), float(dec), 0)
        x_px = float(np.asarray(x_px).reshape(-1)[0])
        y_px = float(np.asarray(y_px).reshape(-1)[0])
        if np.isfinite(x_px) and np.isfinite(y_px):
            return x_px, y_px
    except Exception:
        pass
    return None


def _crop(data, cx, cy, size):
    half = size // 2

    cx = int(round(cx))
    cy = int(round(cy))

    out = np.full((size, size), -5000.0, dtype=np.float32)

    x1 = max(0, cx - half)
    x2 = min(data.shape[1], cx + half + 1)
    y1 = max(0, cy - half)
    y2 = min(data.shape[0], cy + half + 1)

    crop = data[y1:y2, x1:x2]

    ox = half - (cx - x1)
    oy = half - (cy - y1)

    out[
        oy:oy + crop.shape[0],
        ox:ox + crop.shape[1]
    ] = crop

    return out


def _crop_mask(valid_mask, cx, cy, size):
    """Crop a validity mask, marking pixels outside the source image invalid."""
    half = size // 2
    cx = int(round(cx))
    cy = int(round(cy))
    out = np.zeros((size, size), dtype=bool)
    x1 = max(0, cx - half)
    x2 = min(valid_mask.shape[1], cx + half + 1)
    y1 = max(0, cy - half)
    y2 = min(valid_mask.shape[0], cy + half + 1)
    crop = valid_mask[y1:y2, x1:x2]
    ox = half - (cx - x1)
    oy = half - (cy - y1)
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def build_triplet(
    diff_path,
    sci_path,
    ref_path,
    *,
    ra,
    dec,
    size=63,
    ref_flip_lr=False,
):
    """Build a triplet by centering each image on the source sky position."""
    if not np.isfinite(float(ra)) or not np.isfinite(float(dec)):
        raise ValueError("Finite RA and Dec are required to build a WCS-aligned triplet")

    def load_and_crop(path):
        data, wcs, valid_mask = _load_image_and_wcs(path)
        # ZTF headers can contain non-celestial axes.  Project through the
        # celestial WCS so the result is always a 2D pixel position.
        image_wcs = wcs.celestial if getattr(wcs, "naxis", 2) > 2 else wcs
        try:
            x_px, y_px = image_wcs.all_world2pix(float(ra), float(dec), 0)
        except Exception:
            try:
                x_px, y_px = image_wcs.all_world2pix([[float(ra), float(dec)]], 0)[0]
            except Exception:
                # Cutouts requested with build_cutout_url are centered on the
                # requested sky position.  Use that guaranteed cutout center
                # if an old or incomplete FITS WCS cannot be evaluated.
                x_px = (data.shape[1] - 1) / 2.0
                y_px = (data.shape[0] - 1) / 2.0
        # Astropy may return scalar NumPy arrays even for scalar coordinates.
        # Convert them before _crop calls round(), which requires real scalars.
        x_px = float(np.asarray(x_px).reshape(-1)[0])
        y_px = float(np.asarray(y_px).reshape(-1)[0])
        if not np.isfinite(x_px) or not np.isfinite(y_px):
            raise ValueError(f"Could not project source coordinates into {path}")
        return (
            _crop(data, x_px, y_px, size),
            _crop_mask(valid_mask, x_px, y_px, size),
            (float(x_px), float(y_px)),
        )

    diff, diff_mask, diff_position = load_and_crop(diff_path)
    sci, sci_mask, sci_position = load_and_crop(sci_path)
    ref, ref_mask, ref_position = load_and_crop(ref_path)

    source_positions = {
        "sci": sci_position,
        "ref": ref_position,
        "diff": diff_position,
    }
    triplet = np.stack([sci, ref, diff], axis=-1)
    valid_masks = np.stack([sci_mask, ref_mask, diff_mask], axis=-1)
    triplet[~valid_masks] = np.nan
    return triplet, source_positions, valid_masks


def stage0_diff_index(diff_root: Path, out_root: Path, limit_images: int = 0, object_ids: Iterable[str] | None = None, *, resume: bool = True, full_frames: bool = False) -> pd.DataFrame:
    out_path = ensure_dir(out_root / "stage0") / "april_stage0_diff_index.csv"
    if resume:
        cached = load_csv_if_exists(out_path)
        if cached is not None:
            return cached

    df = collect_full_diff_index(diff_root) if full_frames else collect_diff_index(diff_root)
    if object_ids and not full_frames:
        wanted = {str(_extract_object_id(v)) for v in object_ids if str(v).strip()}
        df = df[df["object_id"].astype(str).isin(wanted)].copy()
    if limit_images and limit_images > 0:
        df = df.head(int(limit_images)).copy()
    write_csv(df, out_path)
    return df


def _measure_stage1_image_task(task: tuple) -> dict:
    """Process one FITS image in a worker process for stage1 SNR measurement."""
    image_path, sky_position, sigma, maxiters, min_valid_pixel, null_seed = task
    image_path = Path(image_path)
    if sky_position is None:
        parsed = parse_diff_name(image_path.name) or {}
        fallback_ra = _safe_float(parsed.get("ra"))
        fallback_dec = _safe_float(parsed.get("dec"))
        sky_position = (
            (fallback_ra, fallback_dec)
            if np.isfinite(fallback_ra) and np.isfinite(fallback_dec)
            else None
        )

    center = _pixel_coordinates_from_sky(image_path, *sky_position) if sky_position is not None else None
    try:
        record = process_difference_image(
            image_path,
            sigma=sigma,
            maxiters=maxiters,
            min_valid_pixel=min_valid_pixel,
            center_x=center[0] if center is not None else None,
            center_y=center[1] if center is not None else None,
            null_seed=null_seed,
        )
        record_dict = record.__dict__.copy()
        centroid_sky = _sky_coordinates_from_pixel(
            image_path,
            record_dict.get("source_x_px", np.nan),
            record_dict.get("source_y_px", np.nan),
        )
        if centroid_sky is not None:
            record_dict["centroid_ra"] = centroid_sky[0]
            record_dict["centroid_dec"] = centroid_sky[1]
        record_dict["status"] = "ok"
        return record_dict
    except Exception as exc:
        return {"image_path": str(image_path), "status": f"error: {exc}"}


def stage1_quadratic_snr(
    diff_index: pd.DataFrame,
    out_root: Path,
    *,
    final_class_df: pd.DataFrame | None = None,
    snr_threshold: float,
    sigma: float,
    maxiters: int,
    min_valid_pixel: float,
    max_workers: int = 1,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage1_dir = ensure_dir(out_root / "stage1")
    images_path = stage1_dir / "april_stage1_quadratic_snr_images.csv"
    objects_path = stage1_dir / "april_stage1_quadratic_snr_objects.csv"

    if resume:
        cached_images = load_csv_if_exists(images_path)
        cached_objects = load_csv_if_exists(objects_path)
        if (
            cached_images is not None
            and cached_objects is not None
            and {"centroid_ra", "centroid_dec", "snr_empirical"}.issubset(cached_images.columns)
        ):
            return cached_images, cached_objects

    if diff_index.empty:
        empty = diff_index.copy()
        write_csv(empty, images_path)
        write_csv(empty, objects_path)
        return empty, empty

    coord_index = _build_object_coordinate_index(final_class_df)
    tasks = []
    for _, image_row in diff_index.iterrows():
        image_path = Path(str(image_row["image_path"]))
        object_id = _extract_object_id(image_row.get("object_id", ""))
        sky_position = coord_index.get(object_id)
        tasks.append(
            (
                str(image_path),
                sky_position,
                sigma,
                maxiters,
                min_valid_pixel,
                abs(hash(str(image_path))) % (2**32),
            )
        )

    worker_count = max(1, int(max_workers))
    if worker_count == 1:
        rows = list(
            tqdm(
                map(_measure_stage1_image_task, tasks),
                total=len(tasks),
                desc="Measuring empirical SNR",
                unit="image",
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(
                tqdm(
                    executor.map(_measure_stage1_image_task, tasks, chunksize=16),
                    total=len(tasks),
                    desc="Measuring empirical SNR",
                    unit="image",
                )
            )

    rows = pd.DataFrame(rows)
    if "image_path" not in rows.columns:
        rows["image_path"] = diff_index["image_path"].values[: len(rows)]
    merged = diff_index.merge(rows, on="image_path", how="left", suffixes=("", "_phot"))
    merged["snr"] = pd.to_numeric(merged.get("snr"), errors="coerce")
    merged["snr_empirical"] = pd.to_numeric(merged.get("snr_empirical"), errors="coerce")
    pass_df = merged[merged["snr_empirical"] >= float(snr_threshold)].copy()

    summary = (
        pass_df.groupby("object_id", dropna=False)
        .agg(
            best_snr_empirical=("snr_empirical", "max"),
            best_snr_naive=("snr", "max"),
            n_pass_images=("image_path", "count"),
            first_obs_date=("filefracday", "min"),
            last_obs_date=("filefracday", "max"),
        )
        .reset_index()
    )
    # Keep the historical best_snr name, but define it as the empirical metric.
    summary["best_snr"] = summary["best_snr_empirical"]

    write_csv(merged, images_path)
    write_csv(summary, objects_path)
    return merged, summary


def run_stage1(diff_index, out_root, config, final_class_df, resume):
    """Galaxy-position SNR (cutouts) or blind full-frame search, chosen by config['blind_search']."""
    if bool(config.get("blind_search", False)):
        return stage1_blind_search(
            diff_index,
            out_root,
            snr_threshold=float(config.get("snr_threshold", 3.0)),
            min_valid_pixel=float(config.get("min_valid_pixel", -4000.0)),
            min_detections=int(config.get("blind_min_detections", 1)),
            min_epochs=int(config.get("blind_min_epochs", 1)),
            link_radius_arcsec=float(config.get("blind_link_radius_arcsec", 2.0)),
            max_workers=int(config.get("snr_workers", 4)),
            resume=resume,
        )
    return stage1_quadratic_snr(
        diff_index,
        out_root,
        final_class_df=final_class_df,
        snr_threshold=float(config.get("snr_threshold", 3.0)),
        sigma=float(config.get("sigma", 3.0)),
        maxiters=int(config.get("maxiters", 5)),
        min_valid_pixel=float(config.get("min_valid_pixel", -5000.0)),
        max_workers=int(config.get("snr_workers", 4)),
        resume=resume,
    )


def _make_triplet_manifest_row(row: pd.Series, diff_path: Path, sci_path: Path, ref_path: Path, status: str, ref_sep_arcsec: float, triplet: np.ndarray | None = None) -> dict:
    record = row.to_dict()
    record.update(
        {
            "diff_path": str(diff_path),
            "sci_path": str(sci_path),
            "ref_path": str(ref_path),
            "status": status,
            "ref_match_sep_arcsec": ref_sep_arcsec,
            "is_complete_triplet": status == "ok",
        }
    )
    if triplet is not None:
        record["triplet_ready"] = True
    return record


def stage2_triplets(
    snr_images: pd.DataFrame,
    out_root: Path,
    *,
    sci_root: Path,
    ref_root: Path,
    final_class_df: pd.DataFrame,
    download_missing: bool,
    triplet_size: int,
    ref_flip_lr: bool,
    max_workers: int,
    chunk_size: int = 200,
    resume: bool = True,
    full_frames: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sci_index = build_file_index(sci_root)
    ref_index = build_file_index(ref_root)
    coord_index = _build_object_coordinate_index(final_class_df)

    # Map object ids to final-class row numbers for file/dir naming
    final_rownum_map = _build_final_rownum_map(final_class_df)

    stage2_dir = ensure_dir(out_root / "stage2")
    triplet_dir = ensure_dir(stage2_dir / "triplets")
    manifest_path = triplet_dir / "april_stage2_triplets_manifest.csv"
    objects_path = triplet_dir / "april_stage2_triplets_objects.csv"

    manifest_rows: list[dict] = []
    complete_rows: list[dict] = []

    # A candidate can have several detections (one diff image per epoch). Each
    # image's own SNR centroid wobbles by a fraction of a pixel from noise, so
    # measuring it independently per image made every triplet for the same
    # candidate crop around a slightly different sky position -- the first
    # triplet built would land at a different center than the rest. Instead,
    # anchor every triplet for a candidate on ONE fixed sky position: the
    # measured centroid of its best (highest-SNR) detection, falling back to
    # the catalog position. All sci/ref/diff cutouts for that candidate are
    # then built (and downloaded) around that same anchor.
    candidate_center_lookup: dict[str, tuple[float, float]] = {}
    candidate_center_source: dict[str, str] = {}
    if not snr_images.empty:
        ranked = snr_images.copy()
        ranked["_snr_rank"] = pd.to_numeric(ranked.get("snr_empirical"), errors="coerce").fillna(-np.inf)
        ranked["_object_id"] = ranked.apply(
            lambda r: _extract_object_id(r.get("object_id", (parse_diff_name(Path(str(r["image_path"])).name) or {}).get("object_id"))),
            axis=1,
        )
        ranked = ranked[ranked["_object_id"].astype(bool)]
        ranked = ranked.sort_values("_snr_rank", ascending=False)
        best_per_object = ranked.drop_duplicates(subset=["_object_id"], keep="first")
        for _, best_row in best_per_object.iterrows():
            obj = best_row["_object_id"]
            best_diff_path = Path(str(best_row["image_path"]))
            best_parsed = parse_diff_name(best_diff_path.name) or {}
            centroid_ra = _safe_float(best_row.get("centroid_ra", np.nan))
            centroid_dec = _safe_float(best_row.get("centroid_dec", np.nan))
            if np.isfinite(centroid_ra) and np.isfinite(centroid_dec):
                candidate_center_lookup[obj] = (centroid_ra, centroid_dec)
                candidate_center_source[obj] = "stage1_centroid_wcs"
                continue
            sx = _safe_float(best_row.get("source_x_px", np.nan))
            sy = _safe_float(best_row.get("source_y_px", np.nan))
            centroid_sky = _sky_coordinates_from_pixel(best_diff_path, sx, sy)
            if centroid_sky is not None:
                candidate_center_lookup[obj] = centroid_sky
                candidate_center_source[obj] = "best_detection_centroid_wcs"
                continue
            coords_from_id = coord_index.get(obj)
            if coords_from_id is not None:
                candidate_center_lookup[obj] = coords_from_id
                candidate_center_source[obj] = "final_class_by_id"
                continue
            fallback_ra = _safe_float(best_row.get("ra", best_parsed.get("ra")))
            fallback_dec = _safe_float(best_row.get("dec", best_parsed.get("dec")))
            if np.isfinite(fallback_ra) and np.isfinite(fallback_dec):
                candidate_center_lookup[obj] = (fallback_ra, fallback_dec)
                candidate_center_source[obj] = "snr_or_diff"

    cached_manifest = None
    cached_complete = None
    if resume:
        cached_manifest = load_csv_if_exists(manifest_path)
        cached_complete = load_csv_if_exists(objects_path)
        if cached_manifest is not None and not {"centroid_ra", "centroid_dec"}.issubset(cached_manifest.columns):
            cached_manifest = None
        if cached_complete is not None and not {"centroid_ra", "centroid_dec"}.issubset(cached_complete.columns):
            cached_complete = None
        # Triplets created before mask persistence cannot be trusted for
        # morphology plots. Rebuild them rather than silently reusing them.
        if cached_complete is not None and any(
            not Path(str(row.get("triplet_path", ""))).with_name("triplet_valid_mask.npy").exists()
            for _, row in cached_complete.iterrows()
            if str(row.get("triplet_path", ""))
        ):
            cached_manifest = None
            cached_complete = None
            manifest_rows.clear()
            complete_rows.clear()

        if cached_manifest is not None:
            manifest_rows.extend(cached_manifest.to_dict("records"))
        if cached_complete is not None:
            complete_rows.extend(cached_complete.to_dict("records"))

    completed_ids = set()
    if cached_complete is not None and not cached_complete.empty and {"image_path", "triplet_path"}.issubset(cached_complete.columns):
        completed_ids = {
            str(row["image_path"])
            for _, row in cached_complete.iterrows()
            if str(row.get("image_path", ""))
            and Path(str(row.get("triplet_path", ""))).exists()
        }
    elif cached_manifest is not None and not cached_manifest.empty and "triplet_ready" in cached_manifest.columns:
        ready_rows = cached_manifest[cached_manifest["triplet_ready"].fillna(False).astype(bool)]
        if {"image_path", "triplet_path"}.issubset(ready_rows.columns):
            still_valid = ready_rows[
                ready_rows["image_path"].astype(str).astype(bool)
                & ready_rows["triplet_path"].apply(lambda p: Path(str(p)).exists())
                & ready_rows["triplet_path"].apply(
                    lambda p: Path(str(p)).with_name("triplet_valid_mask.npy").exists()
                )
            ]
            completed_ids = set(still_valid["image_path"].astype(str))
            # objects.csv was missing/empty even though these triplets already
            # exist on disk (per the manifest). Restore them into complete_rows
            # so they still reach stage2_braai for scoring, instead of being
            # marked "already done" and silently dropped from the pipeline.
            complete_rows.extend(still_valid.to_dict("records"))

    if completed_ids:
        snr_images = snr_images[~snr_images["image_path"].astype(str).isin(completed_ids)].copy()

    def build_one(row: pd.Series) -> dict:
        diff_path = Path(row["image_path"])
        parsed = parse_diff_name(diff_path.name) or parse_full_diff_name(diff_path.name)
        if parsed is None:
            return _make_triplet_manifest_row(row, diff_path, Path(""), Path(""), "bad_diff_name", np.nan)

        object_id = _extract_object_id(row.get("object_id", parsed.get("object_id", "")))
        coords_from_id = coord_index.get(object_id)
        # Every triplet for this candidate is centered on the SAME fixed sky
        # position (see candidate_center_lookup above), not on a centroid
        # re-measured from this particular image -- otherwise triplets from
        # different detections of the same candidate drift apart by the
        # per-image centroiding noise.
        ra = _safe_float(row.get("ra", parsed.get("ra", np.nan)))
        dec = _safe_float(row.get("dec", parsed.get("dec", np.nan)))
        coord_source = "snr_or_diff"
        if object_id in candidate_center_lookup:
            ra, dec = candidate_center_lookup[object_id]
            coord_source = candidate_center_source[object_id]
        elif coords_from_id is not None and (not np.isfinite(ra) or not np.isfinite(dec)):
            ra, dec = coords_from_id
            coord_source = "final_class_by_id"

        sci_row = {
            "filefracday": parsed["filefracday"],
            "field": parsed["field"],
            "filtercode": parsed["filter"],
            "ccdid": parsed["ccdid"],
            "qid": parsed["qid"],
            "imgtypecode": "o",
        }
        sci_url = build_sci_url(sci_row, suffix="sciimg.fits")
        sci_name = Path(sci_url).name
        sci_src = sci_index.get(sci_name)
        object_label = final_rownum_map.get(object_id, object_id)
        group_dir = ensure_dir(out_root / "stage2" / "triplets" / object_label / diff_path.stem)
        diff_dst = group_dir / "diff" / diff_path.name
        sci_dst = group_dir / "sci" / sci_name
        if full_frames:
            # Full frames are ~40 MB: read them in place instead of copying them per candidate.
            diff_dst = diff_path
            if sci_src is not None:
                sci_dst = sci_src
        else:
            diff_dst.parent.mkdir(parents=True, exist_ok=True)
            if diff_path.exists() and not diff_dst.exists():
                diff_dst.write_bytes(diff_path.read_bytes())
            sci_dst.parent.mkdir(parents=True, exist_ok=True)
        sci_status = "missing"
        if sci_src is not None and sci_src.exists():
            if not full_frames and not sci_dst.exists():
                sci_dst.write_bytes(sci_src.read_bytes())
            sci_status = "local"
        elif download_missing and not full_frames:
            try:
                downloaded = download_file(build_cutout_url(sci_url, ra, dec, size_arcsec=240), out_dir=sci_dst.parent)
                sci_dst = Path(downloaded)
                sci_status = "downloaded"
            except Exception:
                sci_status = "missing"

        # Reference frames are static per field/ccd/quadrant/filter -- no need to
        # search a pre-fetched reference-metadata table for a "best match". Build
        # the URL directly from the diff image's own field/ccd/qid/filter, same
        # as the science image above, and fetch it on demand if not already local.
        ref_row = {
            "filefracday": parsed["filefracday"],
            "field": parsed["field"],
            "filtercode": parsed["filter"],
            "ccdid": parsed["ccdid"],
            "qid": parsed["qid"],
            "imgtypecode": "o",
        }
        ref_url = build_ref_url(ref_row)
        ref_name = Path(ref_url).name
        ref_src = ref_index.get(ref_name)
        ref_dst = group_dir / "ref" / ref_name
        if full_frames and ref_src is not None:
            ref_dst = ref_src
        else:
            ref_dst.parent.mkdir(parents=True, exist_ok=True)
        ref_status = "missing"
        if ref_src is not None and ref_src.exists():
            if not full_frames and not ref_dst.exists():
                ref_dst.write_bytes(ref_src.read_bytes())
            ref_status = "local"
        elif download_missing and not full_frames:
            try:
                downloaded = download_file(build_cutout_url(ref_url, ra, dec, size_arcsec=240), out_dir=ref_dst.parent)
                ref_dst = Path(downloaded)
                ref_status = "downloaded"
            except Exception:
                ref_status = "missing"

        status = "ok" if sci_status in {"local", "downloaded"} and ref_status in {"local", "downloaded"} else "partial"

        triplet = None
        source_positions = None
        if sci_dst.exists() and ref_dst.exists() and diff_dst.exists():
            try:
                triplet, source_positions, valid_masks = build_triplet(
                    diff_dst,
                    sci_dst,
                    ref_dst,
                    ra=ra,
                    dec=dec,
                    size=triplet_size,
                    ref_flip_lr=ref_flip_lr,
                )
                status = "ok"
            except Exception as exc:
                logging.warning("Could not build triplet for %s: %s", diff_path.name, exc)
                triplet = None
                status = "triplet_error"

        manifest = _make_triplet_manifest_row(row, diff_dst, sci_dst, ref_dst, status, np.nan, triplet)
        manifest.update({"ra_used": ra, "dec_used": dec, "coord_source": coord_source})
        if triplet is not None and source_positions is not None:
            for image_key, (source_x, source_y) in source_positions.items():
                manifest[f"{image_key}_source_x_px"] = source_x
                manifest[f"{image_key}_source_y_px"] = source_y
            manifest["triplet_ready"] = True
            manifest["triplet_path"] = str(group_dir / "triplet.npy")
            np.save(group_dir / "triplet.npy", triplet)
            np.save(group_dir / "triplet_valid_mask.npy", valid_masks)
            complete = manifest.copy()
            complete["diff_path"] = str(diff_dst)
            complete["sci_path"] = str(sci_dst)
            complete["ref_path"] = str(ref_dst)
            complete_rows.append(complete)
        return manifest

    rows = snr_images.copy()
    pending_rows = rows.reset_index(drop=True)
    for start in range(0, len(pending_rows), max(1, int(chunk_size))):
        chunk = pending_rows.iloc[start : start + int(chunk_size)]
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
                futures = {executor.submit(build_one, row): idx for idx, row in chunk.iterrows()}
                for future in as_completed(futures):
                    manifest_rows.append(future.result())
        else:
            for _, row in chunk.iterrows():
                manifest_rows.append(build_one(row))

        manifest_df = pd.DataFrame(manifest_rows)
        complete_df = pd.DataFrame(complete_rows)
        write_csv(manifest_df, manifest_path)
        write_csv(complete_df, objects_path)

    manifest_df = pd.DataFrame(manifest_rows)
    complete_df = pd.DataFrame(complete_rows)
    if resume and cached_manifest is not None and not cached_manifest.empty:
        manifest_df = pd.concat([cached_manifest, manifest_df], ignore_index=True)
        if "image_path" in manifest_df.columns:
            manifest_df = manifest_df.drop_duplicates(subset=["image_path"], keep="last").reset_index(drop=True)
    if resume and cached_complete is not None and not cached_complete.empty:
        complete_df = pd.concat([cached_complete, complete_df], ignore_index=True)
        if "image_path" in complete_df.columns:
            complete_df = complete_df.drop_duplicates(subset=["image_path"], keep="last").reset_index(drop=True)
    write_csv(manifest_df, manifest_path)
    write_csv(complete_df, objects_path)
    return manifest_df, complete_df


def stage2_braai(
    manifest_df: pd.DataFrame,
    complete_df: pd.DataFrame,
    out_root: Path,
    *,
    model_path: Path,
    triplet_size: int,
    ref_flip_lr: bool,
    braai_threshold: float,
    batch_size: int = 128,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage2_dir = ensure_dir(out_root / "stage2")
    scores_path = stage2_dir / "april_stage2_braai_scores.csv"
    objects_path = stage2_dir / "april_stage2_braai_objects.csv"

    scores_df = manifest_df.copy()
    objects_df = complete_df.copy()

    if complete_df.empty:
        write_csv(scores_df, scores_path)
        write_csv(objects_df, objects_path)
        return scores_df, objects_df

    model = load_braai_model(model_path)
    work = complete_df.copy().reset_index(drop=True)
    work["braai_score"] = np.nan
    work["braai_pass"] = False

    existing_scores = load_csv_if_exists(scores_path) if resume else None
    if existing_scores is not None and "triplet_path" in existing_scores.columns:
        existing_scores = existing_scores.copy()
        existing_scores["triplet_path"] = existing_scores["triplet_path"].astype(str)
        work["triplet_path"] = work["triplet_path"].astype(str)
        scored_paths = set(existing_scores["triplet_path"].dropna().astype(str).tolist())
        work = work[~work["triplet_path"].isin(scored_paths)].copy()
        base_rows = existing_scores.to_dict("records")
    else:
        base_rows = []

    scored_rows: list[dict] = list(base_rows)
    batch_size = max(1, int(batch_size))
    for start in range(0, len(work), batch_size):
        chunk = work.iloc[start : start + batch_size].copy()
        triplets = []
        chunk_rows = []
        for _, row in chunk.iterrows():
            triplet_path = Path(str(row.get("triplet_path", "")))
            if not triplet_path.exists():
                continue
            triplets.append(np.load(triplet_path))
            chunk_rows.append(row.to_dict())

        if not triplets:
            continue

        triplet_batch = build_triplet_batch(triplets, triplet_size=triplet_size, ref_flip_lr=ref_flip_lr)
        scores = predict_braai_batch(triplet_batch, model)

        for row_dict, score in zip(chunk_rows, scores):
            row_dict = dict(row_dict)
            row_dict["braai_score"] = float(score)
            row_dict["braai_pass"] = bool(score > float(braai_threshold))
            scored_rows.append(row_dict)

        interim_scores = pd.DataFrame(scored_rows)
        if "braai_pass" in interim_scores.columns:
            interim_objects = interim_scores[interim_scores["braai_pass"]].copy()
        else:
            interim_objects = interim_scores.iloc[0:0].copy()
        write_csv(interim_scores, scores_path)
        write_csv(interim_objects, objects_path)

    scores_df = pd.DataFrame(scored_rows)
    if "braai_pass" in scores_df.columns:
        objects_df = scores_df[scores_df["braai_pass"]].copy()
    else:
        objects_df = scores_df.iloc[0:0].copy()
    if not scores_df.empty and "image_path" in scores_df.columns:
        scores_df = scores_df.sort_values([col for col in ["braai_score", "snr_empirical"] if col in scores_df.columns], ascending=[False, False] if "braai_score" in scores_df.columns else True)
    if not objects_df.empty and "image_path" in objects_df.columns:
        objects_df = objects_df.sort_values([col for col in ["braai_score", "snr_empirical"] if col in objects_df.columns], ascending=[False, False] if "braai_score" in objects_df.columns else True)
    write_csv(scores_df, scores_path)
    write_csv(objects_df, objects_path)
    return scores_df, objects_df


def _extract_zero_point(header: dict) -> float:
    keys = ("MAGZP", "MAGZERO", "ZP", "ZEROPT", "ZEROPNT")
    for key in keys:
        value = header.get(key)
        try:
            zp = float(value)
        except Exception:
            continue
        if np.isfinite(zp):
            return zp
    return np.nan


def build_detection_summary(stage2_scores: pd.DataFrame) -> pd.DataFrame:
    """Pick the best detection per object, ranked by BRAAI score then SNR."""
    if stage2_scores.empty:
        return pd.DataFrame(columns=["object_id"])

    work = stage2_scores.copy()
    work["object_id"] = work["object_id"].astype(str)
    work["braai_score"] = pd.to_numeric(work.get("braai_score"), errors="coerce")
    work["snr_empirical"] = pd.to_numeric(work.get("snr_empirical"), errors="coerce")
    sort_cols = [col for col in ["braai_score", "snr_empirical"] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    best = work.drop_duplicates(subset=["object_id"], keep="first").copy()
    keep_cols = [
        "object_id",
        "braai_score",
        "snr_empirical",
        "snr",
        "source_x_px",
        "source_y_px",
        "diff_path",
        "image_path",
        "filefracday",
        "ra",
        "dec",
    ]
    keep_cols = [col for col in keep_cols if col in best.columns]
    return best[keep_cols].reset_index(drop=True)


def _build_host_magnitude_candidates() -> list[tuple[str, str, str]]:
    """Return catalog magnitude candidates in preferred fallback order."""
    return [
        ("rmag_SDSS-DR16", "SDSS-DR16", "r"),
        ("gmag_SDSS-DR16", "SDSS-DR16", "g"),
        ("imag_SDSS-DR16", "SDSS-DR16", "i"),
        ("zmag_SDSS-DR16", "SDSS-DR16", "z"),
        ("umag_SDSS-DR16", "SDSS-DR16", "u"),
        ("Gmag_GAIA-DR3", "Gaia-DR3", "G"),
        ("BPmag_GAIA-DR3", "Gaia-DR3", "BP"),
        ("RPmag_GAIA-DR3", "Gaia-DR3", "RP"),
        ("GRVSmag_GAIA-DR3", "Gaia-DR3", "GRVS"),
        ("Bmag_GLADE+", "GLADE+", "B"),
        ("Bjmag_GLADE+", "GLADE+", "Bj"),
        ("Jmag_GLADE+", "GLADE+", "J"),
        ("Hmag_GLADE+", "GLADE+", "H"),
        ("Kmag_GLADE+", "GLADE+", "K"),
        ("W1mag_GLADE+", "GLADE+", "W1"),
        ("W2mag_GLADE+", "GLADE+", "W2"),
        ("m_J_NED-LVS", "NED-LVS", "J"),
        ("m_H_NED-LVS", "NED-LVS", "H"),
        ("m_Ks_NED-LVS", "NED-LVS", "Ks"),
        ("m_W1_NED-LVS", "NED-LVS", "W1"),
        ("m_W2_NED-LVS", "NED-LVS", "W2"),
        ("m_W3_NED-LVS", "NED-LVS", "W3"),
        ("m_W4_NED-LVS", "NED-LVS", "W4"),
        ("m_FUV_NED-LVS", "NED-LVS", "FUV"),
        ("m_NUV_NED-LVS", "NED-LVS", "NUV"),
        ("W1_CatWISE", "CatWISE", "W1"),
        ("W2_CatWISE", "CatWISE", "W2"),
        ("host_rmag", "existing", "r"),
        ("rmag", "existing", "r"),
    ]


def _legacy_flux_to_mag(flux, mw_transmission) -> float:
    """Convert Legacy Survey tractor flux to an extinction-corrected AB mag."""
    flux_value = _safe_float(flux)
    transmission = _safe_float(mw_transmission)
    if not np.isfinite(flux_value) or flux_value <= 0:
        return np.nan
    if not np.isfinite(transmission) or transmission <= 0:
        transmission = 1.0
    return float(22.5 - 2.5 * np.log10(flux_value / transmission))


def _fetch_legacy_dr9_photometry(
    ra: float,
    dec: float,
    *,
    radius_deg: float = 0.001,
    timeout: float = 30.0,
) -> dict[str, object]:
    """Return the nearest Legacy Survey DR9 tractor source near a position."""
    from io import StringIO

    import requests

    result: dict[str, object] = {
        "legacy_dr9_status": "missing",
        "legacy_dr9_sep_arcsec": np.nan,
    }
    if not np.isfinite(ra) or not np.isfinite(dec):
        result["legacy_dr9_status"] = "missing_coordinates"
        return result

    try:
        # The viewer's historical cat.json route currently returns HTML/500.
        # Data Lab exposes the same DR9 tractor catalog through anonymous TAP.
        query = f"""
            SELECT ra, dec, type, objid, flux_g, flux_r, flux_z,
                   mw_transmission_g, mw_transmission_r, mw_transmission_z
            FROM ls_dr9.tractor
            WHERE ra BETWEEN {float(ra) - radius_deg} AND {float(ra) + radius_deg}
              AND dec BETWEEN {float(dec) - radius_deg} AND {float(dec) + radius_deg}
        """
        response = requests.get(
            "https://datalab.noirlab.edu/tap/sync",
            params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
            timeout=timeout,
        )
        response.raise_for_status()
        if not response.text.lstrip().startswith(("ra,", "RA,")):
            raise RuntimeError(f"Data Lab returned non-CSV response: {response.text[:160]!r}")
        catalog = pd.read_csv(StringIO(response.text))
        catalog.columns = [str(column).lower() for column in catalog.columns]
        sources = catalog.to_dict("records")
        if catalog.empty:
            result["legacy_dr9_status"] = "no_source"
            return result

        def angular_distance(source: dict) -> float:
            source_ra = _safe_float(source.get("ra"))
            source_dec = _safe_float(source.get("dec"))
            if not np.isfinite(source_ra) or not np.isfinite(source_dec):
                return np.inf
            dra = (source_ra - ra) * np.cos(np.deg2rad(dec))
            ddec = source_dec - dec
            return float(np.hypot(dra, ddec))

        nearest = min(sources, key=angular_distance, default=None)
        if nearest is None or not np.isfinite(angular_distance(nearest)):
            result["legacy_dr9_status"] = "no_valid_source"
            return result

        result.update(
            {
                "legacy_dr9_status": "ok",
                "legacy_dr9_sep_arcsec": angular_distance(nearest) * 3600.0,
                "legacy_dr9_type": nearest.get("type"),
                "legacy_dr9_objid": nearest.get("objid"),
                "legacy_gmag": _legacy_flux_to_mag(nearest.get("flux_g"), nearest.get("mw_transmission_g")),
                "legacy_rmag": _legacy_flux_to_mag(nearest.get("flux_r"), nearest.get("mw_transmission_r")),
                "legacy_zmag": _legacy_flux_to_mag(nearest.get("flux_z"), nearest.get("mw_transmission_z")),
            }
        )
    except Exception as exc:
        logging.warning("Legacy DR9 query failed for RA=%s Dec=%s: %s", ra, dec, exc)
        result["legacy_dr9_status"] = "request_error"
    return result


def stage3_pretrigger_magnitude(
    *,
    stage2_objects: pd.DataFrame,
    diff_index: pd.DataFrame,
    diff_root: Path,
    final_class_path: Path,
    out_dir: Path,
    trigger_date,
    lookback_days: int = 10,
    posttrigger_days: int = 10,
    sigma: float = 3.0,
    maxiters: int = 5,
    min_valid_pixel: float = -5000.0,
    download_missing: bool = True,
    full_frames: bool = False,
    resume: bool = True,
):
    from ztf_downloads.ztf_search import metadata_search_batch, metadata_search_resumable
    from ztf_downloads.ztf_download import batch_download_resumable
    from ztf_downloads.ztf_download import build_cutout_url, build_sci_url
    from ztf_downloads.utils import _save_json
    from urllib.parse import urlparse


    out_dir = ensure_dir(out_dir)
    images_path = out_dir / "april_stage3_pretrigger_mag_images.csv"
    objects_path = out_dir / "april_stage3_pretrigger_mag_objects.csv"

    if resume:
        cached_images = load_csv_if_exists(images_path)
        cached_objects = load_csv_if_exists(objects_path)
        if cached_images is not None and cached_objects is not None and not cached_objects.empty:
            cache_version = 2
            has_current_cache_version = (
                "stage1_exposure_match_version" in cached_images.columns
                and "stage1_exposure_match_version" in cached_objects.columns
                and pd.to_numeric(
                    cached_images["stage1_exposure_match_version"], errors="coerce"
                ).eq(cache_version).all()
                and pd.to_numeric(
                    cached_objects["stage1_exposure_match_version"], errors="coerce"
                ).eq(cache_version).all()
                and "pretrigger_mag_pass" in cached_objects.columns
                and "n_posttrigger_detections" in cached_objects.columns
                and cached_objects["pretrigger_mag_pass"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes"})
                .all()
                and pd.to_numeric(
                    cached_objects.get("n_posttrigger_detections"), errors="coerce"
                )
                .gt(0)
                .all()
            )
            if not has_current_cache_version:
                cached_stage1 = load_csv_if_exists(
                    out_dir.parent / "stage1" / "april_stage1_quadratic_snr_images.csv"
                )
                stage1_lookup = {}
                if cached_stage1 is not None and {
                    "object_id",
                    "image_path",
                    "source_x_px",
                    "source_y_px",
                }.issubset(cached_stage1.columns):
                    for _, stage1_row in cached_stage1.iterrows():
                        if not np.isfinite(_safe_float(stage1_row.get("source_x_px"))) or not np.isfinite(
                            _safe_float(stage1_row.get("source_y_px"))
                        ):
                            continue
                        key = (
                            normalize_object_id(stage1_row.get("object_id")),
                            _norm_key_name(Path(str(stage1_row["image_path"])).name),
                        )
                        if key[0]:
                            stage1_lookup[key] = stage1_row.to_dict()

                matched_rows = 0
                stage1_fields = (
                    "snr",
                    "snr_empirical",
                    "p_value_empirical",
                    "significance_tier",
                    "detection",
                    "mag",
                    "mag_err",
                    "upper_limit",
                )
                if stage1_lookup and {"object_id", "image_path"}.issubset(cached_images.columns):
                    for image_idx, image_row in cached_images.iterrows():
                        key = (
                            normalize_object_id(image_row.get("object_id")),
                            _norm_key_name(Path(str(image_row["image_path"])).name),
                        )
                        stage1_row = stage1_lookup.get(key)
                        if stage1_row is None:
                            continue
                        for field in stage1_fields:
                            if field in stage1_row:
                                value = stage1_row[field]
                                if field == "detection":
                                    value = str(value).strip().lower() in {"true", "1", "yes"}
                                cached_images.at[image_idx, field] = value
                        if str(stage1_row.get("detection", "")).strip().lower() in {
                            "true",
                            "1",
                            "yes",
                        }:
                            cached_images.at[image_idx, "upper_limit"] = np.nan
                        matched_rows += 1

                if matched_rows:
                    for object_idx, object_row in cached_objects.iterrows():
                        object_id = normalize_object_id(object_row.get("object_id"))
                        object_images = cached_images[
                            cached_images["object_id"].astype(str).map(normalize_object_id)
                            == object_id
                        ]
                        pre = object_images[object_images["phase"].astype(str) == "pretrigger"]
                        post = object_images[object_images["phase"].astype(str) == "posttrigger"]
                        pre_detection = pre.get(
                            "detection", pd.Series(False, index=pre.index)
                        ).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
                        post_detection = post.get(
                            "detection", pd.Series(False, index=post.index)
                        ).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
                        n_pre = len(pre)
                        has_pre_detection = bool(pre_detection.any())
                        n_post_detections = int(post_detection.sum())
                        if n_pre == 0:
                            reason = "no_pretrigger_images"
                            consistent = False
                            passed = False
                        elif has_pre_detection:
                            reason = "pretrigger_detection"
                            consistent = False
                            passed = False
                        elif n_post_detections == 0:
                            reason = "no_posttrigger_detections"
                            consistent = True
                            passed = False
                        else:
                            reason = "no_pretrigger_detections"
                            consistent = True
                            passed = True
                        cached_objects.at[object_idx, "n_pretrigger_images"] = n_pre
                        cached_objects.at[object_idx, "n_posttrigger_detections"] = n_post_detections
                        cached_objects.at[object_idx, "pretrigger_mag_consistent"] = consistent
                        cached_objects.at[object_idx, "pretrigger_mag_pass"] = passed
                        cached_objects.at[object_idx, "pretrigger_mag_reason"] = reason
                    cached_objects = cached_objects[
                        cached_objects["pretrigger_mag_pass"].astype(str).str.strip().str.lower().isin(
                            {"true", "1", "yes"}
                        )
                    ].copy()
                    cached_images["stage1_exposure_match_version"] = cache_version
                    cached_objects["stage1_exposure_match_version"] = cache_version
                    write_csv(cached_images, images_path)
                    write_csv(cached_objects, objects_path)

            cached_ids = set(cached_objects["object_id"].astype(str).map(normalize_object_id)) if "object_id" in cached_objects.columns else set()
            cached_post_ids = set()
            if {"object_id", "phase", "filter"}.issubset(cached_images.columns):
                cached_post = cached_images[cached_images["phase"].astype(str) == "posttrigger"]
                cached_post_ids = set(
                    cached_post["object_id"].astype(str).map(normalize_object_id)
                )
            cache_has_posttrigger_data = bool(cached_ids) and cached_ids.issubset(
                cached_post_ids
            )
            if (
                {"stage1_centroid_ra", "stage1_centroid_dec"}.issubset(cached_objects.columns)
                and "snr_empirical" in cached_images.columns
                and "stage3_coordinate_source" in cached_images.columns
                and "pretrigger_mag_pass" in cached_objects.columns
                and "stage1_exposure_match_version" in cached_images.columns
                and "stage1_exposure_match_version" in cached_objects.columns
                and cached_objects["pretrigger_mag_pass"].any()
                and cache_has_posttrigger_data
            ):
                passed = cached_objects[cached_objects["pretrigger_mag_pass"]].copy()
                return cached_images, cached_objects, passed
            logging.info("Recomputing stage3 pre-trigger analysis because cached outputs contain no passing candidates")

    trigger = pd.to_datetime(trigger_date)
    window_start = trigger - pd.Timedelta(days=int(lookback_days))
    # Include the trigger day and the following days in every filter.
    # This covers 25-26 April as well as the remainder of the post-trigger window.
    posttrigger_window_start = trigger
    posttrigger_window_end = trigger + pd.Timedelta(days=int(posttrigger_days))
    first_two_days_end = trigger + pd.Timedelta(days=2)

    # Prepare diff_index with object_id and obs_date
    work = diff_index.copy()
    if "object_id" not in work.columns:
        work["object_id"] = work["image_path"].map(lambda p: parse_diff_name(Path(p).name)["object_id"] if parse_diff_name(Path(p).name) else "")
    work["object_id"] = work["object_id"].map(normalize_object_id)

    def _parse_obs_date(image_path):
        parsed = parse_diff_name(Path(image_path).name)
        if parsed is None:
            return pd.NaT
        return _parse_filefracday_datetime(parsed.get("filefracday"))

    work["obs_date"] = work["image_path"].map(_parse_obs_date)
    work_by_object = {
        object_id: group
        for object_id, group in work.groupby("object_id", sort=False)
    }

    # Normalize object IDs from stage2_objects and build coordinate lookup directly
    object_ids = []
    coord_lookup = {}
    # also collect source pixel centers from stage2_objects so we can carry them forward
    source_center_lookup = {}
    stage1_images_path = out_dir.parent / "stage1" / "april_stage1_quadratic_snr_images.csv"
    stage1_images = load_csv_if_exists(stage1_images_path)
    stage1_by_image: dict[str, dict] = {}
    best_stage1_sky: dict[str, tuple[float, float]] = {}
    if stage1_images is not None and "image_path" in stage1_images.columns:
        stage1_images = stage1_images.copy()
        stage1_images["_object_key"] = stage1_images["object_id"].map(normalize_object_id)
        stage1_images["_snr_rank"] = pd.to_numeric(
            stage1_images.get("snr_empirical"), errors="coerce"
        ).fillna(-np.inf)
        for _, stage1_row in stage1_images.sort_values("_snr_rank", ascending=False).drop_duplicates(
            "_object_key"
        ).iterrows():
            obj = str(stage1_row.get("_object_key", ""))
            ra = _safe_float(stage1_row.get("centroid_ra"))
            dec = _safe_float(stage1_row.get("centroid_dec"))
            if np.isfinite(ra) and np.isfinite(dec):
                best_stage1_sky[obj] = (ra, dec)
            else:
                path = Path(str(stage1_row["image_path"]))
                sx = _safe_float(stage1_row.get("source_x_px"))
                sy = _safe_float(stage1_row.get("source_y_px"))
                projected = _sky_coordinates_from_pixel(path, sx, sy)
                if projected is not None:
                    best_stage1_sky[obj] = projected
    if stage1_images is not None and {"image_path", "source_x_px", "source_y_px"}.issubset(stage1_images.columns):
        for _, stage1_row in stage1_images.iterrows():
            sx = _safe_float(stage1_row.get("source_x_px"))
            sy = _safe_float(stage1_row.get("source_y_px"))
            object_key = normalize_object_id(stage1_row.get("object_id"))
            if object_key and np.isfinite(sx) and np.isfinite(sy):
                image_key = (
                    object_key,
                    _norm_key_name(Path(str(stage1_row["image_path"])).name),
                )
                stage1_by_image[image_key] = stage1_row.to_dict()
    # sci_path is produced back in stage2 (per-image) but was previously dropped before
    # stage4. stage4 needs the diff image (not sci) to measure fwhm_px, since
    # measure_fwhm_from_image operates on the difference image at the detection position.
    diff_path_lookup = {}
    best_braai_seen = {}
    for _, row in stage2_objects.iterrows():
        obj_raw = row.get("object_id")
        if pd.isna(obj_raw):
            continue
        obj = normalize_object_id(obj_raw)
        if not obj:
            continue
        object_ids.append(obj)
        row_diff_path = row.get("diff_path")
        if row_diff_path and isinstance(row_diff_path, str):
            row_score = _safe_float(row.get("braai_score", np.nan))
            best_score = best_braai_seen.get(obj, -np.inf)
            if obj not in diff_path_lookup or (np.isfinite(row_score) and row_score > best_score):
                diff_path_lookup[obj] = row_diff_path
                best_braai_seen[obj] = row_score if np.isfinite(row_score) else best_score
        # Use exactly one sky anchor per object: the best stage-1 centroid.
        if obj in best_stage1_sky:
            coord_lookup[obj] = best_stage1_sky[obj]
        else:
            ra = _safe_float(row.get("centroid_ra", row.get("ra_used", row.get("ra", np.nan))))
            dec = _safe_float(row.get("centroid_dec", row.get("dec_used", row.get("dec", np.nan))))
            if np.isfinite(ra) and np.isfinite(dec):
                coord_lookup[obj] = (ra, dec)
            else:
                # If missing, try from work (last resort)
                obj_rows = work_by_object.get(obj)
                if obj_rows is not None and not obj_rows.empty:
                    ra = np.nanmedian(obj_rows["ra"])
                    dec = np.nanmedian(obj_rows["dec"])
                    if np.isfinite(ra) and np.isfinite(dec):
                        coord_lookup[obj] = (ra, dec)
        # source pixel centers: try common column names
        sx = _safe_float(row.get("source_x_px", row.get("source_x", row.get("x", np.nan))))
        sy = _safe_float(row.get("source_y_px", row.get("source_y", row.get("y", np.nan))))
        if not np.isfinite(sx):
            sx = np.nan
        if not np.isfinite(sy):
            sy = np.nan
        source_center_lookup[obj] = (sx, sy)

    # Remove duplicates
    object_ids = list(dict.fromkeys(object_ids))
    if not object_ids:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _measurement_center(path: Path, object_id: str):
        """Project the single best stage-1 sky anchor into this image."""
        sky = coord_lookup.get(object_id)
        if sky is not None:
            projected = _pixel_coordinates_from_sky(path, sky[0], sky[1])
            if projected is not None:
                return projected, "stage1_centroid_wcs"
        return None, "missing_stage1_centroid"

    def _reuse_stage1_significance(path: Path, object_id: str, record_dict: dict) -> dict:
        """Reuse full-frame significance for the same exposure's archived cutout."""
        image_key = (object_id, _norm_key_name(path.name))
        stage1_row = stage1_by_image.get(image_key)
        if stage1_row is None:
            return record_dict
        for field in (
            "snr",
            "snr_empirical",
            "p_value_empirical",
            "significance_tier",
            "detection",
            "mag",
            "mag_err",
            "upper_limit",
        ):
            if field in stage1_row and pd.notna(stage1_row[field]):
                value = stage1_row[field]
                if field in {"snr", "snr_empirical", "p_value_empirical"}:
                    value = _safe_float(value)
                elif field == "detection":
                    value = str(value).strip().lower() in {"true", "1", "yes"}
                record_dict[field] = value
        if str(stage1_row.get("detection", "")).strip().lower() in {"true", "1", "yes"}:
            record_dict["upper_limit"] = np.nan
        return record_dict

    # Load final-class CSV to map object ids -> row numbers for naming
    try:
        final_class_df = pd.read_csv(final_class_path, low_memory=False)
    except Exception:
        final_class_df = None
    final_rownum_map = _build_final_rownum_map(final_class_df)
    # Full-frame candidates use their best stage-1 centroid and its WCS position;
    # catalog coordinates are only appropriate for catalog-cutout searches.
    download_coord_lookup = (
        best_stage1_sky
        if full_frames
        else _build_object_coordinate_index(final_class_df)
    )

    # Identify objects that need pre-trigger downloads and post-trigger downloads.
    # Post-trigger coverage is checked independently for each g/r/i filter,
    # including the trigger day and the following two days.
    download_needed = {}
    download_post_needed = {}
    if download_missing:
        for obj in object_ids:
            obj_work = work_by_object.get(obj, work.iloc[0:0])
            pre = obj_work[(obj_work["obs_date"] < trigger) & (obj_work["obs_date"] >= window_start)]
            if len(pre) == 0:
                download_needed[obj] = True

            post = obj_work[
                (obj_work["obs_date"] >= posttrigger_window_start)
                & (obj_work["obs_date"] < first_two_days_end)
            ]
            post_filters = set(post["filter"].dropna().astype(str)) if "filter" in post.columns else set()
            if not {"zg", "zr", "zi"}.issubset(post_filters):
                download_post_needed[obj] = True

    if download_needed or download_post_needed:
        logging.info(
            "Downloading missing pre/post-trigger images for %s objects (%s pre, %s post)",
            len(set(list(download_needed.keys()) + list(download_post_needed.keys()))),
            len(download_needed),
            len(download_post_needed),
        )
        download_inputs = []
        metadata_jobs = []
        for obj in dict.fromkeys([*download_needed, *download_post_needed]):
            if obj not in download_coord_lookup:
                coordinate_source = "stage-1 centroid" if full_frames else "galaxy catalog"
                logging.warning("Could not find %s coordinates for object %s", coordinate_source, obj)
                continue
            ra, dec = download_coord_lookup[obj]

            obj_norm = normalize_object_id(obj)
            rownum = final_rownum_map.get(obj_norm, obj_norm)
            search_csv = out_dir / f"pretrigger_search_{rownum}.csv"
            search_progress = out_dir / f"pretrigger_search_{rownum}.progress.json"
            date_start = window_start.strftime("%Y-%m-%d")
            date_end = trigger.strftime("%Y-%m-%d")
            if obj in download_needed and obj in download_post_needed:
                date_end = posttrigger_window_end.strftime("%Y-%m-%d")
                search_csv = out_dir / f"allwindow_search_v3_{rownum}.csv"
                search_progress = out_dir / f"allwindow_search_v3_{rownum}.progress.json"
            elif obj in download_post_needed:
                # The archive query uses an exclusive lower date bound. Search
                # one day earlier so observations on the trigger date are not
                # discarded before the exact Python-side window check below.
                date_start = (posttrigger_window_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                date_end = posttrigger_window_end.strftime("%Y-%m-%d")
                # Use a new cache namespace because older post-trigger search
                # files were created during the r/i-only download workflow.
                search_csv = out_dir / f"posttrigger_search_all_filters_v2_{rownum}.csv"
                search_progress = out_dir / f"posttrigger_search_all_filters_v2_{rownum}.progress.json"
            metadata_jobs.append(
                (obj, obj_norm, ra, dec, search_csv, search_progress, date_start, date_end)
            )

        def _fetch_meta(job):
            obj, _, ra, dec, search_csv, search_progress, date_start, date_end = job

            try:
                return metadata_search_resumable(
                    positions=[(ra, dec)],
                    output_csv=str(search_csv),
                    progress_json=str(search_progress),
                    batch_size=50,
                    size_deg=0.01,
                    product_type="sci",
                    filtercodes=["zg", "zr", "zi"],
                    date_start=date_start,
                    date_end=date_end,
                    timeout=120,
                )
            except Exception as e1:
                logging.warning("Metadata search initial attempt failed for %s: %s", obj, e1)
            try:
                return metadata_search_resumable(
                    positions=[(ra, dec)],
                    output_csv=str(search_csv),
                    progress_json=str(search_progress),
                    batch_size=50,
                    size_deg=0.01,
                    product_type="sci",
                    filtercodes=["zg", "zr", "zi"],
                    date_start=date_start,
                    date_end=date_end,
                    timeout=300,
                    max_retries=8,
                )
            except Exception as e2:
                logging.warning("Metadata search extended attempt failed for %s: %s", obj, e2)

            try:
                parts = []
                day_start = pd.to_datetime(date_start).normalize()
                day_end_dt = pd.to_datetime(date_end).normalize()
                cur = day_start
                while cur < day_end_dt:
                    ds = cur.strftime("%Y-%m-%d")
                    de = (cur + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    try:
                        part = metadata_search_batch(
                            positions=[(ra, dec)],
                            size_deg=0.01,
                            product_type="sci",
                            ct="csv",
                            date_start=ds,
                            date_end=de,
                            filtercodes=["zg", "zr", "zi"],
                            timeout=180,
                        )
                        if part is None:
                            part = pd.DataFrame()
                        parts.append(part)
                    except Exception as e3:
                        logging.warning(
                            "Per-day metadata search failed for %s %s->%s: %s",
                            obj,
                            ds,
                            de,
                            e3,
                        )
                    cur += pd.Timedelta(days=1)
                if parts:
                    return pd.concat(parts, ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
            except Exception as e4:
                logging.warning("Per-day fallback failed for %s: %s", obj, e4)

            return pd.DataFrame()

        def _load_cached_meta(job):
            _, _, _, _, search_csv, search_progress, _, _ = job
            if not search_csv.exists() or not search_progress.exists():
                return None
            try:
                state = json.loads(search_progress.read_text(encoding="utf-8"))
                if int(state.get("total_batches", 0)) != 1:
                    return None
                if 0 not in state.get("completed_batches", []):
                    return None
                if not search_csv.stat().st_size:
                    return pd.DataFrame()
                return pd.read_csv(search_csv, low_memory=False)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                logging.warning("Ignoring unreadable metadata cache %s: %s", search_csv, exc)
                return None

        def _save_job_meta(job, df_meta):
            _, _, _, _, search_csv, search_progress, _, _ = job
            cached = df_meta.copy()
            if "in_row_id" in cached.columns:
                cached["in_row_id"] = 1
            search_csv.parent.mkdir(parents=True, exist_ok=True)
            cached.to_csv(search_csv, index=False)
            _save_json(
                search_progress,
                {
                    "completed_batches": [0],
                    "total_batches": 1,
                    "output_csv": str(search_csv),
                    "rows_last_batch": int(len(cached)),
                },
            )

        def _fetch_meta_batch(jobs):
            positions = [(job[2], job[3]) for job in jobs]
            date_start, date_end = jobs[0][6], jobs[0][7]
            result = None
            for timeout in (120, 300):
                try:
                    result = metadata_search_batch(
                        positions=positions,
                        size_deg=0.01,
                        product_type="sci",
                        ct="csv",
                        date_start=date_start,
                        date_end=date_end,
                        filtercodes=["zg", "zr", "zi"],
                        timeout=timeout,
                    )
                    break
                except Exception as exc:
                    logging.warning(
                        "Batched metadata search failed for %s candidates (timeout=%s): %s",
                        len(jobs),
                        timeout,
                        exc,
                    )

            if result is None:
                if len(jobs) == 1:
                    return [(jobs[0], _fetch_meta(jobs[0]))]
                midpoint = len(jobs) // 2
                return _fetch_meta_batch(jobs[:midpoint]) + _fetch_meta_batch(jobs[midpoint:])

            if result.empty:
                return [(job, result.copy()) for job in jobs]
            if "in_row_id" not in result.columns:
                if len(jobs) == 1:
                    return [(jobs[0], result)]
                raise ValueError("Batched archive response is missing in_row_id")

            row_ids = pd.to_numeric(result["in_row_id"], errors="coerce")
            if row_ids.isna().any() or not row_ids.between(1, len(jobs)).all():
                raise ValueError("Batched archive response has invalid in_row_id values")
            result = result.copy()
            result["_stage3_input_row"] = row_ids.astype(int)
            return [
                (
                    job,
                    result.loc[result["_stage3_input_row"] == row_number]
                    .drop(columns=["_stage3_input_row"])
                    .copy(),
                )
                for row_number, job in enumerate(jobs, start=1)
            ]

        grouped_jobs = {}
        metadata_results = []
        for job in metadata_jobs:
            cached = _load_cached_meta(job)
            if cached is not None:
                metadata_results.append((job, cached))
            else:
                grouped_jobs.setdefault((job[6], job[7]), []).append(job)

        metadata_batches = []
        for grouped in grouped_jobs.values():
            metadata_batches.extend(
                grouped[start : start + 25]
                for start in range(0, len(grouped), 25)
            )

        logging.info(
            "Fetching archive metadata for %s candidates in %s batches",
            sum(map(len, grouped_jobs.values())),
            len(metadata_batches),
        )
        worker_count = min(4, len(metadata_batches))
        with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
            futures = {
                executor.submit(_fetch_meta_batch, batch): batch
                for batch in metadata_batches
            }
            for future in as_completed(futures):
                batch_results = future.result()
                for job, df_meta in batch_results:
                    _save_job_meta(job, df_meta)
                    metadata_results.append((job, df_meta))

        for job, df_meta in metadata_results:
            obj, obj_norm, _, _, _, _, _, _ = job
            if df_meta is None or df_meta.empty:
                logging.warning("Metadata search yielded no results for %s", obj)
                continue

            # ztf_search currently does not apply filtercodes server-side.
            # Keep all returned g/r/i rows explicitly for the post-trigger set.
            if obj in download_post_needed and "filtercode" in df_meta.columns:
                df_meta = df_meta[
                    df_meta["filtercode"].astype(str).str.lower().isin(
                        {"g", "r", "i", "zg", "zr", "zi"}
                    )
                ].copy()

            for _, mrow in df_meta.iterrows():
                filefracday = mrow.get("filefracday", "")
                obs_dt = _parse_filefracday_datetime(str(filefracday))
                if pd.isna(obs_dt):
                    continue
                in_pretrigger_window = window_start <= obs_dt < trigger
                in_posttrigger_window = (
                    posttrigger_window_start
                    <= obs_dt
                    <= posttrigger_window_end
                )
                should_download_pre = obj in download_needed and in_pretrigger_window
                should_download_post = (
                    obj in download_post_needed and in_posttrigger_window
                )
                if not (should_download_pre or should_download_post):
                    continue

                in_ra = _safe_float(mrow.get("in_ra"))
                in_dec = _safe_float(mrow.get("in_dec"))
                if not np.isfinite(in_ra) or not np.isfinite(in_dec):
                    logging.warning("Skipping metadata row with invalid coordinates for %s", obj)
                    continue
                base_sci_url = build_sci_url(mrow, suffix="scimrefdiffimg.fits.fz")
                original_fname = Path(urlparse(base_sci_url).path).name
                url = build_cutout_url(
                    base_sci_url,
                    in_ra,
                    in_dec,
                    size_arcsec=240,
                )
                filefracday_str = str(int(mrow["filefracday"])).zfill(14)
                custom_label = (
                    f"RA{in_ra:.4f}_DEC{in_dec:.4f}_"
                    f"{mrow['filtercode']}_{filefracday_str}"
                )
                row_label = final_rownum_map.get(obj_norm, obj_norm)
                desired_fname = f"{str(row_label)}_{custom_label}__{original_fname}"
                download_inputs.append((url, desired_fname, str(row_label)))

        if download_inputs:
            logging.info(f"Downloading {len(download_inputs)} missing time-window cutouts")
            batch_download_resumable(
                download_inputs,
                out_dir=str(diff_root),
                progress_json=str(out_dir / "time_window_download.progress.json"),
                max_workers=6,
                chunk_size=100,
            )
            diff_index = collect_diff_index(diff_root)
            work = diff_index.copy()
            work["object_id"] = work["object_id"].map(normalize_object_id)
            work["obs_date"] = work["image_path"].map(_parse_obs_date)
            work_by_object = {
                object_id: group
                for object_id, group in work.groupby("object_id", sort=False)
            }


    # Now separate pre and post trigger
    pretrigger = work[(work["obs_date"] < trigger) & (work["obs_date"] >= window_start)].copy()
    posttrigger = work[(work["obs_date"] >= trigger) & (work["obs_date"] <= posttrigger_window_end)].copy()
    pretrigger_by_object = {
        object_id: group
        for object_id, group in pretrigger.groupby("object_id", sort=False)
    }
    posttrigger_by_object = {
        object_id: group
        for object_id, group in posttrigger.groupby("object_id", sort=False)
    }

    image_rows = []
    object_rows = []
    measurement_progress = tqdm(
        total=len(pretrigger) + len(posttrigger),
        desc="Measuring stage-3 cutouts",
        unit="image",
        dynamic_ncols=True,
    )

    def _progress_rows(frame):
        for item in frame.iterrows():
            yield item
            measurement_progress.update(1)

    for object_id in object_ids:
        group = pretrigger_by_object.get(object_id, pretrigger.iloc[0:0])
        object_post = posttrigger_by_object.get(object_id, posttrigger.iloc[0:0])

        # Measure post-trigger magnitudes
        post_mags = []
        post_mag_errs = []
        post_records = []
        for _, row in _progress_rows(object_post):
            path = Path(row["image_path"])
            if not path.exists():
                continue
            try:
                image_center, coordinate_source = _measurement_center(path, object_id)
                record = process_difference_image(
                    path,
                    sigma=sigma,
                    maxiters=maxiters,
                    min_valid_pixel=min_valid_pixel,
                    center_x=image_center[0] if image_center is not None else None,
                    center_y=image_center[1] if image_center is not None else None,
                    recenter=False,
                )
                record_dict = record.__dict__.copy()
                record_dict = _reuse_stage1_significance(path, object_id, record_dict)
                record_dict["stage3_coordinate_source"] = coordinate_source
            except Exception as exc:
                image_rows.append({"object_id": object_id, "image_path": str(path), "status": f"error: {exc}"})
                continue
            record_dict.update({"object_id": object_id, "phase": "posttrigger"})
            image_rows.append(record_dict)
            post_records.append(record_dict)
            if np.isfinite(record_dict.get("mag", np.nan)):
                post_mags.append(float(record_dict["mag"]))
                if np.isfinite(record_dict.get("mag_err", np.nan)):
                    post_mag_errs.append(float(record_dict["mag_err"]))
        post_detection_count = sum(
            str(record.get("detection", "")).strip().lower() in {"true", "1", "yes"}
            for record in post_records
        )

        # Measure pre-trigger magnitudes
        pre_records = []
        for _, row in _progress_rows(group):
            path = Path(row["image_path"])
            if not path.exists():
                continue
            try:
                image_center, coordinate_source = _measurement_center(path, object_id)
                record = process_difference_image(
                    path,
                    sigma=sigma,
                    maxiters=maxiters,
                    min_valid_pixel=min_valid_pixel,
                    center_x=image_center[0] if image_center is not None else None,
                    center_y=image_center[1] if image_center is not None else None,
                    recenter=False,
                )
                record_dict = record.__dict__.copy()
                record_dict = _reuse_stage1_significance(path, object_id, record_dict)
                record_dict["stage3_coordinate_source"] = coordinate_source
            except Exception as exc:
                image_rows.append({"object_id": object_id, "image_path": str(path), "status": f"error: {exc}"})
                continue
            record_dict.update({"object_id": object_id, "phase": "pretrigger"})
            image_rows.append(record_dict)
            pre_records.append(record_dict)

        pre_mags = [float(item["mag"]) for item in pre_records if np.isfinite(item.get("mag", np.nan))]
        pre_mag_errs = [float(item["mag_err"]) for item in pre_records if np.isfinite(item.get("mag_err", np.nan))]
        pre_upper_limits = [float(item["upper_limit"]) for item in pre_records if np.isfinite(item.get("upper_limit", np.nan))]
        pre_ref = float(np.nanmedian(pre_mags)) if pre_mags else np.nan
        pre_ref_err = float(np.nanmedian(pre_mag_errs)) if pre_mag_errs else np.nan
        # residual field name varies by photometry-record version; check the common aliases
        pre_residuals = [
            float(item[key])
            for item in pre_records
            for key in ("residual", "max_residual", "fit_residual")
            if key in item and np.isfinite(_safe_float(item.get(key)))
        ]
        pre_max_residual = float(np.nanmax(pre_residuals)) if pre_residuals else np.nan

        post_ref_mag = float(np.nanmedian(post_mags)) if post_mags else np.nan
        post_ref_err = float(np.nanmedian(post_mag_errs)) if post_mag_errs else np.nan

        # A final candidate needs pre-trigger coverage without detections and at
        # least one post-trigger detection.
        n_pre = len(group)
        if n_pre == 0:
            # No pre-trigger images -> cannot confirm transient is caused by trigger
            consistent = False
            pass_flag = False
            reason = "no_pretrigger_images"
        else:
            # Check whether any pre-trigger record had a detection
            pre_detections = [
                r for r in pre_records
                if str(r.get("detection", "")).strip().lower() in {"true", "1", "yes"}
            ]
            if pre_detections:
                # There was at least one detection before the trigger -> fail
                consistent = False
                pass_flag = False
                reason = "pretrigger_detection"
            elif post_detection_count == 0:
                consistent = True
                pass_flag = False
                reason = "no_posttrigger_detections"
            else:
                # Clean pre-trigger history and a post-trigger detection -> pass
                consistent = True
                pass_flag = True
                reason = "no_pretrigger_detections"

        object_image_records = [
            record
            for record in (*post_records, *pre_records)
            if np.isfinite(_safe_float(record.get("snr_empirical", np.nan)))
        ]
        best_image_record = max(
            object_image_records,
            key=lambda record: _safe_float(record.get("snr_empirical", -np.inf)),
            default={},
        )
        sx = _safe_float(best_image_record.get("source_x_px", np.nan))
        sy = _safe_float(best_image_record.get("source_y_px", np.nan))
        # detection RA/DEC: prefer coord_lookup, else use median from work
        dra, ddec = (np.nan, np.nan)
        if object_id in coord_lookup:
            dra, ddec = coord_lookup[object_id]
        else:
            obj_rows = work_by_object.get(object_id)
            if obj_rows is not None and not obj_rows.empty:
                dra = np.nanmedian(obj_rows["ra"]) if "ra" in obj_rows.columns else np.nan
                ddec = np.nanmedian(obj_rows["dec"]) if "dec" in obj_rows.columns else np.nan
        object_rows.append({
            "object_id": object_id,
            "n_pretrigger_images": n_pre,
            "pretrigger_mag_upper_limit": float(np.nanmax(pre_upper_limits)) if pre_upper_limits else np.nan,
            "pretrigger_mag_consistent": consistent,
            "pretrigger_mag_pass": pass_flag,
            "pretrigger_mag_reason": reason,
            "posttrigger_reference_kind": "posttrigger_median",
            "posttrigger_reference_mag": post_ref_mag,
            "posttrigger_reference_mag_err": post_ref_err,
            "n_posttrigger_detections": int(post_detection_count),
            "source_x_px": sx,
            "source_y_px": sy,
            "detection_ra": dra,
            "detection_dec": ddec,
            "stage1_centroid_ra": coord_lookup.get(object_id, (np.nan, np.nan))[0],
            "stage1_centroid_dec": coord_lookup.get(object_id, (np.nan, np.nan))[1],
            "diff_path": diff_path_lookup.get(object_id, ""),
        })

    measurement_progress.close()
    mag_df = pd.DataFrame(image_rows)
    stage3_objects_all = pd.DataFrame(object_rows)
    stage3_objects_pass = stage3_objects_all[stage3_objects_all["pretrigger_mag_pass"]].copy() if "pretrigger_mag_pass" in stage3_objects_all.columns else stage3_objects_all.iloc[0:0].copy()
    mag_df["stage1_exposure_match_version"] = 2
    stage3_objects_all["stage1_exposure_match_version"] = 2
    stage3_objects_pass["stage1_exposure_match_version"] = 2

    write_csv(mag_df, images_path)
    write_csv(stage3_objects_pass, objects_path)
    return mag_df, stage3_objects_all, stage3_objects_pass

def _compute_spike_ratio_from_image(
    diff_path: str | Path | None,
    center_x: float | None,
    center_y: float | None,
    seeing_fwhm: float | None,
) -> float:
    """Compute the peak-to-PSF-amplitude ratio for a compact source.

    This is not a universal visibility gate on its own; it is used as a diagnostic
    that can be combined with the PSF-wing score below when a stricter veto is desired.
    """
    if diff_path is None or not Path(str(diff_path)).exists():
        return np.nan
    if center_x is None or center_y is None or not np.isfinite(center_x) or not np.isfinite(center_y):
        return np.nan
    if seeing_fwhm is None or not np.isfinite(seeing_fwhm) or seeing_fwhm <= 0:
        return np.nan
    try:
        from astropy.io import fits

        data = fits.getdata(str(diff_path), memmap=False)
        arr = np.asarray(data, dtype=float)
        yy, xx = np.indices(arr.shape)
        r2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
        s = seeing_fwhm / 2.35482
        psf = np.exp(-r2 / (2.0 * s * s))
        ok = np.isfinite(arr) & np.isfinite(psf)
        if not np.any(ok):
            return np.nan
        w = psf[ok]
        d = arr[ok]
        denom = float((w * w).sum())
        if denom == 0:
            return np.nan
        amp = float((w * d).sum() / denom)
        if not np.isfinite(amp) or amp == 0:
            return np.nan
        peak = np.nanmax(np.where(r2 <= 4.0, arr, np.nan))
        if not np.isfinite(peak):
            return np.nan
        return float(peak / amp)
    except Exception:
        return np.nan


def _compute_psf_visibility_score(
    psf_spike_delta_chi2: pd.Series,
    spike_ratio: pd.Series,
    fwhm_ratio: pd.Series,
) -> pd.Series:
    """Return a composite score favouring clean PSF-like sources over spikes.

    This is calibrated on the current 14-object sample and should be treated as a
    ranking score or an optional veto, not as a universal science gate.
    """
    delta = pd.to_numeric(psf_spike_delta_chi2, errors="coerce").fillna(0.0)
    spike = pd.to_numeric(spike_ratio, errors="coerce").fillna(0.0)
    ratio = pd.to_numeric(fwhm_ratio, errors="coerce").fillna(1.0)
    feature_df = pd.DataFrame(
        {
            "log1p_delta": np.log1p(np.clip(delta.to_numpy(dtype=float), 0.0, None)),
            "log1p_spike": np.log1p(np.clip(spike.to_numpy(dtype=float), 0.0, None)),
            "fwhm_ratio": ratio.to_numpy(dtype=float),
        }
    )
    mu = feature_df.mean(axis=0)
    sigma = feature_df.std(axis=0, ddof=0).replace(0, 1.0)
    standardized = (feature_df - mu) / sigma
    weights = np.array([4.72105977, 0.4658493, -4.07055548], dtype=float)
    return standardized.dot(weights)


def stage4_host_mag(
    stage3_objects: pd.DataFrame,
    final_class_df: pd.DataFrame,
    out_root: Path,
    *,
    host_rmag_threshold: float,
    fwhm_reduced_chi2_max: float = 5.0,
    fwhm_ratio_min: float = 0.5,
    fwhm_ratio_max: float = 1.8,
    fwhm_sigma_multiplier: float = 3.0,
    psf_spike_delta_chi2_min: float = 9.0,
    psf_amplitude_snr_min: float = 3.0,
    source_delta_chi2_min: float = 9.0,
    psf_residual_rms_sigma_max: float = 1.5,
    candidate_snr_min: float = 3.0,
    candidate_snr_max: float = 5.0,
    psf_visibility_gate_enabled: bool = False,
    psf_visibility_score_min: float = 2.362838023766029,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_class_df = _ensure_object_id_column(final_class_df)

    stage4_dir = ensure_dir(out_root / "stage4")
    host_objects_path = stage4_dir / "april_stage4_hostmag_objects.csv"
    final_candidates_path = stage4_dir / "april_stage4_final_candidates.csv"
    uncertain_candidates_path = stage4_dir / "april_stage4_fwhm_uncertain_candidates.csv"
    cached_host = load_csv_if_exists(host_objects_path)
    cached_final = load_csv_if_exists(final_candidates_path)
    cached_uncertain = load_csv_if_exists(uncertain_candidates_path)
    if cached_host is not None and cached_final is not None and not cached_host.empty:
        if resume:
            cached_ids = cached_host["object_id"].astype(str).map(normalize_object_id).tolist()
            current_ids = stage3_objects["object_id"].astype(str).map(normalize_object_id).tolist()
            cached_diff_fwhm = pd.to_numeric(cached_host.get("diff_fwhm"), errors="coerce")
            diff_fwhm_cache_valid = "diff_fwhm" in cached_host.columns and cached_diff_fwhm.notna().all()
            cached_fwhm = pd.to_numeric(cached_host.get("fwhm_px"), errors="coerce")
            fwhm_cache_valid = "fwhm_px" in cached_host.columns and cached_fwhm.notna().all()
            fwhm_error_cache_valid = "fwhm_err_px" in cached_host.columns
            fwhm_fit_rms_cache_valid = "fwhm_fit_rms" in cached_host.columns
            fwhm_chi2_cache_valid = "fwhm_reduced_chi2" in cached_host.columns
            fwhm_filter_cache_valid = (
                "fwhm_filter_version" in cached_host.columns
                and pd.to_numeric(cached_host["fwhm_filter_version"], errors="coerce").eq(10).all()
                and {
                    "fwhm_ratio",
                    "fwhm_ratio_min",
                    "fwhm_ratio_max",
                    "fwhm_sigma_multiplier",
                    "psf_spike_delta_chi2",
                    "psf_amplitude_snr",
                    "psf_spike_delta_chi2_min",
                    "psf_amplitude_snr_min",
                    "source_delta_chi2",
                    "psf_residual_rms_sigma",
                    "source_delta_chi2_min",
                    "psf_residual_rms_sigma_max",
                    "candidate_snr_min",
                    "candidate_snr_max",
                }.issubset(cached_host.columns)
                and pd.to_numeric(cached_host["fwhm_ratio_min"], errors="coerce").eq(float(fwhm_ratio_min)).all()
                and pd.to_numeric(cached_host["fwhm_ratio_max"], errors="coerce").eq(float(fwhm_ratio_max)).all()
                and pd.to_numeric(cached_host["fwhm_sigma_multiplier"], errors="coerce").eq(float(fwhm_sigma_multiplier)).all()
                and pd.to_numeric(cached_host["psf_spike_delta_chi2_min"], errors="coerce").eq(float(psf_spike_delta_chi2_min)).all()
                and pd.to_numeric(cached_host["psf_amplitude_snr_min"], errors="coerce").eq(float(psf_amplitude_snr_min)).all()
                and pd.to_numeric(cached_host["source_delta_chi2_min"], errors="coerce").eq(float(source_delta_chi2_min)).all()
                and pd.to_numeric(cached_host["psf_residual_rms_sigma_max"], errors="coerce").eq(float(psf_residual_rms_sigma_max)).all()
                and pd.to_numeric(cached_host["candidate_snr_min"], errors="coerce").eq(float(candidate_snr_min)).all()
                and pd.to_numeric(cached_host["candidate_snr_max"], errors="coerce").eq(float(candidate_snr_max)).all()
            )
            magnitude_cache_valid = {"best_snr_mag", "best_snr_mag_err"}.issubset(cached_host.columns)
            snr_cache_valid = {
                "snr",
                "snr_empirical",
                "best_snr_normal",
                "best_snr_empirical",
                "snr_all",
                "snr_empirical_all",
            }.issubset(cached_host.columns)
            removed_columns_absent = not {
                "pretrigger_mag_ref",
                "pretrigger_mag_ref_err",
                "pretrigger_mag_max_residual",
            }.intersection(cached_host.columns)
            host_mag_metadata_cache_valid = {"host_rmag", "host_rmag_source", "host_rmag_band", "host_rmag_catalog", "host_w1_mag", "host_w2_mag", "host_w1_catalog", "host_w2_catalog", "host_desi_id"}.issubset(cached_host.columns)
            legacy_dr9_cache_valid = (
                {"legacy_dr9_status", "legacy_dr9_sep_arcsec", "legacy_gmag", "legacy_rmag", "legacy_zmag", "legacy_dr9_type", "legacy_dr9_objid"}.issubset(cached_host.columns)
                and not cached_host["legacy_dr9_status"].astype(str).eq("request_error").any()
            )
            fwhm_consistency_cache_valid = {"fwhm_px", "diff_fwhm", "fwhm_pass", "fwhm_reason"}.issubset(cached_host.columns)
            fwhm_status_cache_valid = "fwhm_status" in cached_host.columns
            fwhm_source_cache_valid = (
                "fwhm_source" in cached_host.columns
                and cached_host["fwhm_source"].astype(str).eq("stage1_centroid_fixed_gaussian_covariance").all()
            )
            if cached_ids == current_ids and cached_uncertain is not None and fwhm_cache_valid and fwhm_error_cache_valid and fwhm_fit_rms_cache_valid and fwhm_chi2_cache_valid and fwhm_filter_cache_valid and diff_fwhm_cache_valid and fwhm_consistency_cache_valid and fwhm_status_cache_valid and magnitude_cache_valid and snr_cache_valid and removed_columns_absent and host_mag_metadata_cache_valid and legacy_dr9_cache_valid and fwhm_source_cache_valid:
                return cached_host, cached_final
            logging.info(
                "Recomputing stage4 outputs because cached IDs or diff_fwhm values are stale: %s -> %s",
                len(cached_ids),
                len(current_ids),
            )
        else:
            logging.info("Recomputing stage4 outputs because resume=False")

    final = stage3_objects.copy()
    final = final.drop(
        columns=[
            "pretrigger_mag_ref",
            "pretrigger_mag_ref_err",
            "pretrigger_mag_max_residual",
        ],
        errors="ignore",
    )
    # propagate source pixel centers into final output for downstream use
    if "source_x_px" in final.columns:
        final["detection_x_px"] = final["source_x_px"]
    else:
        final["detection_x_px"] = np.nan
    if "source_y_px" in final.columns:
        final["detection_y_px"] = final["source_y_px"]
    else:
        final["detection_y_px"] = np.nan

    if final.empty:
        final["fwhm_px"] = np.nan
        final["fwhm_err_px"] = np.nan
        final["fwhm_fit_rms"] = np.nan
        final["fwhm_reduced_chi2"] = np.nan
        final["fwhm_filter_version"] = 10
        final["fwhm_ratio"] = np.nan
        final["fwhm_ratio_min"] = float(fwhm_ratio_min)
        final["fwhm_ratio_max"] = float(fwhm_ratio_max)
        final["fwhm_sigma_multiplier"] = float(fwhm_sigma_multiplier)
        final["psf_spike_delta_chi2_min"] = float(psf_spike_delta_chi2_min)
        final["psf_amplitude_snr_min"] = float(psf_amplitude_snr_min)
        final["candidate_snr_min"] = float(candidate_snr_min)
        final["candidate_snr_max"] = float(candidate_snr_max)
        final["diff_fwhm"] = np.nan
        final["fwhm_status"] = pd.Series(dtype="object")
        write_csv(final, host_objects_path)
        write_csv(final, final_candidates_path)
        write_csv(final, uncertain_candidates_path)
        return final, final

    final_id_norm = final["object_id"].astype(str).map(normalize_object_id)

    final["host_rmag"] = np.nan
    final["host_rmag_source"] = "missing"
    final["host_rmag_catalog"] = pd.Series([None] * len(final), dtype="object")
    final["host_rmag_band"] = pd.Series([None] * len(final), dtype="object")
    final["host_w1_mag"] = np.nan
    final["host_w2_mag"] = np.nan
    final["host_w1_catalog"] = pd.Series([None] * len(final), dtype="object")
    final["host_w2_catalog"] = pd.Series([None] * len(final), dtype="object")
    final["host_desi_id"] = pd.Series([None] * len(final), dtype="object")

    magnitude_lookups: dict[str, dict[str, float]] = {}
    for column_name, _, _ in _build_host_magnitude_candidates():
        if column_name not in final_class_df.columns:
            continue
        work = final_class_df[["object_id", column_name]].copy()
        work["object_id"] = work["object_id"].astype(str).map(normalize_object_id)
        work[column_name] = pd.to_numeric(work[column_name], errors="coerce")
        magnitude_lookups[column_name] = (
            work.dropna(subset=[column_name])
            .drop_duplicates(subset=["object_id"])
            .set_index("object_id")[column_name]
            .to_dict()
        )

    desi_lookup: dict[str, str | None] = {}
    for desi_column in ("id_DESI-DR8", "objid_DESI-DR8", "objID_DESI-DR8"):
        if desi_column not in final_class_df.columns:
            continue
        work = final_class_df[["object_id", desi_column]].copy()
        work["object_id"] = work["object_id"].astype(str).map(normalize_object_id)
        work[desi_column] = work[desi_column].map(lambda value: None if pd.isna(value) else str(value).strip())
        desi_lookup = (
            work.dropna(subset=[desi_column])
            .drop_duplicates(subset=["object_id"])
            .set_index("object_id")[desi_column]
            .to_dict()
        )
        if desi_lookup:
            break

    for idx, row in final.iterrows():
        object_id = normalize_object_id(row.get("object_id"))
        matched_value = np.nan
        matched_source = "missing"
        matched_catalog = np.nan
        matched_band = np.nan
        catwise_w1 = np.nan
        catwise_w2 = np.nan
        catwise_w1_catalog = None
        catwise_w2_catalog = None

        for column_name, catalog_name, band_name in _build_host_magnitude_candidates():
            if column_name not in final_class_df.columns:
                continue
            candidate_value = magnitude_lookups.get(column_name, {}).get(object_id, np.nan)
            if pd.notna(candidate_value):
                if catalog_name == "CatWISE":
                    if band_name == "W1" and pd.isna(catwise_w1):
                        catwise_w1 = candidate_value
                        catwise_w1_catalog = catalog_name
                    elif band_name == "W2" and pd.isna(catwise_w2):
                        catwise_w2 = candidate_value
                        catwise_w2_catalog = catalog_name
                if pd.isna(matched_value):
                    matched_value = candidate_value
                    matched_source = column_name
                    matched_catalog = catalog_name
                    matched_band = band_name

        final.at[idx, "host_rmag"] = matched_value
        final.at[idx, "host_rmag_source"] = matched_source
        final.at[idx, "host_rmag_catalog"] = matched_catalog
        final.at[idx, "host_rmag_band"] = matched_band
        final.at[idx, "host_w1_mag"] = catwise_w1
        final.at[idx, "host_w2_mag"] = catwise_w2
        final.at[idx, "host_w1_catalog"] = catwise_w1_catalog
        final.at[idx, "host_w2_catalog"] = catwise_w2_catalog
        if pd.isna(matched_value):
            final.at[idx, "host_desi_id"] = desi_lookup.get(object_id)

    host_index = final_class_df.copy()
    id_col = "object_id"
    host_index[id_col] = host_index[id_col].astype(str).map(normalize_object_id)

    lookup = host_index[[id_col, "D_L_fin"]].drop_duplicates(subset=[id_col]).set_index(id_col)["D_L_fin"]
    final["D_L_fin"] = final_id_norm.map(lookup)

    lookup = host_index[[id_col, "e_D_L_fin"]].drop_duplicates(subset=[id_col]).set_index(id_col)["e_D_L_fin"]
    final["e_D_L_fin"] = final_id_norm.map(lookup)
    for distance_column in ("D_L_low_fin", "D_L_high_fin"):
        if distance_column in host_index.columns:
            lookup = (
                host_index[[id_col, distance_column]]
                .drop_duplicates(subset=[id_col])
                .set_index(id_col)[distance_column]
            )
            final[distance_column] = final_id_norm.map(lookup)
        else:
            final[distance_column] = np.nan
    final["D_L_lower"] = pd.to_numeric(final["D_L_low_fin"], errors="coerce")
    final["D_L_upper"] = pd.to_numeric(final["D_L_high_fin"], errors="coerce")

    host_rmag_numeric = pd.to_numeric(final["host_rmag"], errors="coerce")
    final["host_rmag_pass"] = host_rmag_numeric.le(float(host_rmag_threshold)) | host_rmag_numeric.isna()
    final["host_rmag_reason"] = np.where(
        host_rmag_numeric.isna(),
        "host_rmag_missing",
        np.where(final["host_rmag_pass"], "ok", "host_too_faint_or_missing"),
    )
    # final["best_braai_score"] = pd.to_numeric(final.get("braai_score"), errors="coerce")
    stage2_dir = ensure_dir(out_root / "stage2")
    objects_path = stage2_dir / "april_stage2_braai_objects.csv"

    # 1. Read the CSV file
    lookup_df = pd.read_csv(objects_path)
    lookup_df["object_id"] = lookup_df["object_id"].astype(str).map(normalize_object_id)

    # 2. Reduce to the maximum braai_score per object_id (many rows per object possible)
    lookup_df["braai_score"] = pd.to_numeric(lookup_df.get("braai_score"), errors="coerce")
    braai_lookup = lookup_df.groupby("object_id", sort=False)["braai_score"].max()

    # 3. Map the per-object maximum BRAAI score into the final DataFrame
    final["best_braai_score"] = pd.to_numeric(final_id_norm.map(braai_lookup), errors="coerce")


    stage1_dir = ensure_dir(out_root / "stage1")
    objects_path = stage1_dir / "april_stage1_quadratic_snr_objects.csv"

    # 1. Read the CSV file
    lookup_df = pd.read_csv(objects_path)
    lookup_df["object_id"] = lookup_df["object_id"].astype(str).map(normalize_object_id)

    # 2. Reduce to the maximum best_snr per object_id (many rows per object possible)
    snr_column = "best_snr_empirical" if "best_snr_empirical" in lookup_df.columns else "best_snr"
    lookup_df[snr_column] = pd.to_numeric(lookup_df.get(snr_column), errors="coerce")
    snr_lookup = lookup_df.groupby("object_id", sort=False)[snr_column].max()

    # 3. Map the per-object maximum SNR into the final DataFrame
    final["best_snr"] = pd.to_numeric(final_id_norm.map(snr_lookup), errors="coerce")
    stage1_path = out_root / "stage1" / "april_stage1_quadratic_snr_images.csv"
    stage1 = load_csv_if_exists(stage1_path)
    if stage1 is None:
        stage1 = pd.DataFrame(columns=["object_id", "snr", "snr_empirical"])
    stage1 = stage1.copy()
    if not stage1.empty:
        stage1["object_id"] = stage1["object_id"].astype(str).map(normalize_object_id)
        stage1["snr"] = pd.to_numeric(stage1["snr"], errors="coerce")
        stage1["snr_empirical"] = pd.to_numeric(stage1["snr_empirical"], errors="coerce")
    detected_stage1 = stage1[stage1["snr_empirical"] >= 3.0].copy()
    snr_by_object = detected_stage1.groupby("object_id", sort=False)
    final["best_snr_normal"] = final_id_norm.map(snr_by_object["snr"].max())
    final["best_snr_empirical"] = final_id_norm.map(snr_by_object["snr_empirical"].max())
    final["snr_all"] = final_id_norm.map(
        snr_by_object["snr"].apply(
            lambda values: json.dumps([float(value) for value in values.dropna()])
        )
    )
    final["snr_empirical_all"] = final_id_norm.map(
        snr_by_object["snr_empirical"].apply(
            lambda values: json.dumps([float(value) for value in values.dropna()])
        )
    )
    if stage1.empty:
        final["snr"] = np.nan
        final["snr_empirical"] = np.nan
    else:
        best_detection = (
            stage1.sort_values("snr_empirical", ascending=False)
            .drop_duplicates("object_id")
            .set_index("object_id")
        )
        final["snr"] = final_id_norm.map(best_detection["snr"])
        final["snr_empirical"] = final_id_norm.map(best_detection["snr_empirical"])

    # Select the stage-3 measurement from the image with the highest SNR for
    # each object, and carry its magnitude and uncertainty into stage 4.
    stage3_images_path = out_root / "stage3" / "april_stage3_pretrigger_mag_images.csv"
    best_snr_mag_lookup = {}
    if stage3_images_path.exists():
        stage3_images = pd.read_csv(stage3_images_path, low_memory=False)
        required_columns = {"object_id", "snr_empirical", "mag", "mag_err"}
        if required_columns.issubset(stage3_images.columns):
            stage3_images = stage3_images.copy()
            stage3_images["object_id"] = stage3_images["object_id"].astype(str).map(normalize_object_id)
            stage3_images["snr_empirical"] = pd.to_numeric(stage3_images["snr_empirical"], errors="coerce")
            stage3_images["mag"] = pd.to_numeric(stage3_images["mag"], errors="coerce")
            stage3_images["mag_err"] = pd.to_numeric(stage3_images["mag_err"], errors="coerce")
            best_stage3_rows = stage3_images.sort_values("snr_empirical", ascending=False).drop_duplicates("object_id")
            best_snr_mag_lookup = best_stage3_rows.set_index("object_id")[["mag", "mag_err"]].to_dict("index")

    final["best_snr_mag"] = final_id_norm.map(lambda value: best_snr_mag_lookup.get(value, {}).get("mag", np.nan))
    final["best_snr_mag_err"] = final_id_norm.map(lambda value: best_snr_mag_lookup.get(value, {}).get("mag_err", np.nan))


    detection_ra_candidates = ["detection_ra_deg", "detection_ra", "catalog_ra_deg", "ra_used", "ra"]
    detection_dec_candidates = ["detection_dec_deg", "detection_dec", "catalog_dec_deg", "dec_used", "dec"]
    host_ra_candidates = ["RA_fin", "ra_fin", "host_ra", "catalog_ra_deg", "ra", "RA"]
    host_dec_candidates = ["DEC_fin", "dec_fin", "host_dec", "catalog_dec_deg", "dec", "DEC"]

    detection_ra = None
    detection_dec = None
    host_ra = None
    host_dec = None
    for candidate in detection_ra_candidates:
        if candidate in final.columns:
            detection_ra = pd.to_numeric(final[candidate], errors="coerce")
            break
    for candidate in detection_dec_candidates:
        if candidate in final.columns:
            detection_dec = pd.to_numeric(final[candidate], errors="coerce")
            break

    host_ra_col = next((c for c in host_ra_candidates if c in final_class_df.columns), None)
    host_dec_col = next((c for c in host_dec_candidates if c in final_class_df.columns), None)
    if host_ra_col is not None:
        ra_index = final_class_df.copy()
        ra_index["object_id"] = ra_index["object_id"].astype(str).map(normalize_object_id)
        ra_index[host_ra_col] = pd.to_numeric(ra_index[host_ra_col], errors="coerce")
        ra_lookup = ra_index[["object_id", host_ra_col]].drop_duplicates(subset=["object_id"]).set_index("object_id")[host_ra_col]
        host_ra = final_id_norm.map(ra_lookup)
    if host_dec_col is not None:
        dec_index = final_class_df.copy()
        dec_index["object_id"] = dec_index["object_id"].astype(str).map(normalize_object_id)
        dec_index[host_dec_col] = pd.to_numeric(dec_index[host_dec_col], errors="coerce")
        dec_lookup = dec_index[["object_id", host_dec_col]].drop_duplicates(subset=["object_id"]).set_index("object_id")[host_dec_col]
        host_dec = final_id_norm.map(dec_lookup)

    host_offsets = []
    for idx in final.index:
        dra = detection_ra.loc[idx] if detection_ra is not None else np.nan
        ddec = detection_dec.loc[idx] if detection_dec is not None else np.nan
        hra = host_ra.loc[idx] if host_ra is not None else np.nan
        hdec = host_dec.loc[idx] if host_dec is not None else np.nan
        host_offsets.append(compute_host_offset(dra, ddec, hra, hdec))
    final["host_offset_arcsec"] = host_offsets

    # Measure the source width from the same stage-1 detection that defines
    # best_snr. This keeps the image path and SNR centroid paired; stage-3 may
    # contain a different post-trigger image for the same object.
    stage1_detection_lookup = {}
    stage1_images_path = out_root / "stage1" / "april_stage1_quadratic_snr_images.csv"
    if stage1_images_path.exists():
        stage1_images = pd.read_csv(stage1_images_path, low_memory=False)
        required_stage1 = {"object_id", "image_path", "source_x_px", "source_y_px", "snr", "snr_empirical", "fwhm_px"}
        if required_stage1.issubset(stage1_images.columns):
            stage1_images = stage1_images.copy()
            stage1_images["object_id"] = stage1_images["object_id"].astype(str).map(normalize_object_id)
            stage1_images["snr"] = pd.to_numeric(stage1_images["snr"], errors="coerce")
            stage1_images["snr_empirical"] = pd.to_numeric(stage1_images["snr_empirical"], errors="coerce")
            stage1_images = stage1_images.sort_values("snr_empirical", ascending=False).drop_duplicates("object_id")
            stage1_detection_lookup = stage1_images.set_index("object_id").to_dict("index")

    fwhm_values = []
    fwhm_errors = []
    fwhm_fit_rms_values = []
    fwhm_chi2_values = []
    psf_spike_delta_chi2_values = []
    psf_amplitude_snr_values = []
    source_delta_chi2_values = []
    psf_residual_rms_sigma_values = []
    for _, row in final.iterrows():
        stage1_row = stage1_detection_lookup.get(normalize_object_id(row.get("object_id")), {})
        diff_path = stage1_row.get("image_path", row.get("diff_path"))
        x_px = _safe_float(stage1_row.get("source_x_px", row.get("detection_x_px", np.nan)))
        y_px = _safe_float(stage1_row.get("source_y_px", row.get("detection_y_px", np.nan)))
        final.at[row.name, "snr"] = _safe_float(stage1_row.get("snr", np.nan))
        final.at[row.name, "snr_empirical"] = _safe_float(stage1_row.get("snr_empirical", np.nan))
        if not diff_path or not Path(str(diff_path)).exists() or not np.isfinite(x_px) or not np.isfinite(y_px):
            fwhm_values.append(np.nan)
            fwhm_errors.append(np.nan)
            fwhm_fit_rms_values.append(np.nan)
            fwhm_chi2_values.append(np.nan)
            psf_spike_delta_chi2_values.append(np.nan)
            psf_amplitude_snr_values.append(np.nan)
            source_delta_chi2_values.append(np.nan)
            psf_residual_rms_sigma_values.append(np.nan)
            continue
        (
            measured_fwhm,
            measured_fwhm_error,
            fit_rms,
            reduced_chi2,
            delta_chi2,
            amplitude_snr,
            source_delta_chi2,
            residual_rms_sigma,
        ) = measure_fwhm_from_image(
            Path(str(diff_path)), x_px, y_px, box_size=21
        )
        fwhm_values.append(measured_fwhm)
        fwhm_errors.append(measured_fwhm_error)
        fwhm_fit_rms_values.append(fit_rms)
        fwhm_chi2_values.append(reduced_chi2)
        psf_spike_delta_chi2_values.append(delta_chi2)
        psf_amplitude_snr_values.append(amplitude_snr)
        source_delta_chi2_values.append(source_delta_chi2)
        psf_residual_rms_sigma_values.append(residual_rms_sigma)
    final["fwhm_px"] = fwhm_values
    final["fwhm_err_px"] = fwhm_errors
    final["fwhm_fit_rms"] = fwhm_fit_rms_values
    final["fwhm_reduced_chi2"] = fwhm_chi2_values
    final["psf_spike_delta_chi2"] = psf_spike_delta_chi2_values
    final["psf_amplitude_snr"] = psf_amplitude_snr_values
    final["source_delta_chi2"] = source_delta_chi2_values
    final["psf_residual_rms_sigma"] = psf_residual_rms_sigma_values
    final["fwhm_filter_version"] = 10
    final["fwhm_ratio_min"] = float(fwhm_ratio_min)
    final["fwhm_ratio_max"] = float(fwhm_ratio_max)
    final["fwhm_sigma_multiplier"] = float(fwhm_sigma_multiplier)
    final["psf_spike_delta_chi2_min"] = float(psf_spike_delta_chi2_min)
    final["psf_amplitude_snr_min"] = float(psf_amplitude_snr_min)
    final["source_delta_chi2_min"] = float(source_delta_chi2_min)
    final["psf_residual_rms_sigma_max"] = float(psf_residual_rms_sigma_max)
    final["candidate_snr_min"] = float(candidate_snr_min)
    final["candidate_snr_max"] = float(candidate_snr_max)
    final["fwhm_source"] = "stage1_centroid_fixed_gaussian_covariance"

    # Copy the stage-1 SEEING/FWHM for the selected difference image.  Stage 2
    # may relocate the file, so match by basename rather than full path.
    diff_fwhm_lookup = {}
    for stage1_row in stage1_detection_lookup.values():
        image_name = Path(str(stage1_row.get("image_path", ""))).name
        if image_name:
            diff_fwhm_lookup[image_name] = _safe_float(stage1_row.get("fwhm_px"))

    final["diff_fwhm"] = final["object_id"].map(
        lambda value: _safe_float(stage1_detection_lookup.get(normalize_object_id(value), {}).get("fwhm_px", np.nan))
    )

    # PSF-spike ratio and composite visibility score for ranking / optional veto.
    final["spike_ratio"] = final.apply(
        lambda row: _compute_spike_ratio_from_image(
            row.get("diff_path"),
            _safe_float(row.get("detection_x_px", np.nan)),
            _safe_float(row.get("detection_y_px", np.nan)),
            _safe_float(row.get("diff_fwhm", np.nan)),
        ),
        axis=1,
    )
    final["psf_visibility_score"] = _compute_psf_visibility_score(
        final["psf_spike_delta_chi2"],
        final["spike_ratio"],
        final["fwhm_ratio"] if "fwhm_ratio" in final.columns else pd.Series(np.nan, index=final.index),
    )

    fwhm_px_numeric = pd.to_numeric(final.get("fwhm_px"), errors="coerce")
    fwhm_err_numeric = pd.to_numeric(final.get("fwhm_err_px"), errors="coerce")
    diff_fwhm_numeric = pd.to_numeric(final.get("diff_fwhm"), errors="coerce")
    fwhm_measurement_available = fwhm_px_numeric.notna() & diff_fwhm_numeric.notna()
    fwhm_ratio = fwhm_px_numeric / diff_fwhm_numeric
    final["fwhm_ratio"] = fwhm_ratio
    final["psf_visibility_score"] = _compute_psf_visibility_score(
        final["psf_spike_delta_chi2"],
        final["spike_ratio"],
        final["fwhm_ratio"],
    )
    fwhm_abs_delta = (fwhm_px_numeric - diff_fwhm_numeric).abs()
    fwhm_tolerance = float(fwhm_sigma_multiplier) * fwhm_err_numeric
    final["fwhm_abs_delta"] = fwhm_abs_delta
    final["fwhm_tolerance"] = fwhm_tolerance
    fwhm_ratio_pass = fwhm_ratio.between(float(fwhm_ratio_min), float(fwhm_ratio_max))
    fwhm_uncertainty_pass = fwhm_err_numeric.notna() & fwhm_abs_delta.le(fwhm_tolerance)
    psf_spike_pass = (
        pd.to_numeric(final["psf_spike_delta_chi2"], errors="coerce")
        >= float(psf_spike_delta_chi2_min)
    )
    psf_amplitude_pass = (
        pd.to_numeric(final["psf_amplitude_snr"], errors="coerce")
        >= float(psf_amplitude_snr_min)
    )
    source_delta_chi2_pass = (
        pd.to_numeric(final["source_delta_chi2"], errors="coerce")
        >= float(source_delta_chi2_min)
    )
    psf_residual_pass = (
        pd.to_numeric(final["psf_residual_rms_sigma"], errors="coerce")
        <= float(psf_residual_rms_sigma_max)
    )
    final["source_model_pass"] = (
        source_delta_chi2_pass
        & psf_amplitude_pass
        & psf_residual_pass
    )
    final["psf_visibility_pass"] = True
    if psf_visibility_gate_enabled:
        final["psf_visibility_pass"] = (
            pd.to_numeric(final["psf_visibility_score"], errors="coerce")
            >= float(psf_visibility_score_min)
        )
    final["fwhm_pass"] = (
        fwhm_measurement_available
        & fwhm_ratio_pass
        & fwhm_uncertainty_pass
    )
    final["fwhm_status"] = np.select(
        [
            ~fwhm_measurement_available,
            fwhm_measurement_available & ~fwhm_ratio_pass,
            fwhm_measurement_available & fwhm_ratio_pass & ~fwhm_uncertainty_pass,
        ],
        ["missing", "fwhm_ratio_out_of_range", "fwhm_error_inconsistent"],
        default="confirmed",
    )
    final["fwhm_reason"] = np.select(
        [
            ~fwhm_measurement_available,
            fwhm_measurement_available & ~fwhm_ratio_pass,
            fwhm_measurement_available & fwhm_ratio_pass & ~fwhm_uncertainty_pass,
        ],
        ["fwhm_missing", "fwhm_ratio_out_of_range", "fwhm_error_inconsistent"],
        default="ok",
    )

    snr_numeric = pd.to_numeric(final.get("snr_empirical"), errors="coerce")
    final["candidate_snr_pass"] = snr_numeric.gt(float(candidate_snr_min)) & snr_numeric.lt(float(candidate_snr_max))
    final["candidate_snr_reason"] = np.select(
        [
            snr_numeric.isna(),
            snr_numeric.le(float(candidate_snr_min)),
            snr_numeric.ge(float(candidate_snr_max)),
        ],
        ["snr_missing", "snr_below_candidate_range", "snr_above_candidate_range"],
        default="ok",
    )
    final["final_pass"] = final["fwhm_pass"] & final["candidate_snr_pass"]
    if psf_visibility_gate_enabled:
        final["final_pass"] = final["final_pass"] & final["psf_visibility_pass"]
    passed = final[final["final_pass"]].copy()
    uncertain = final.iloc[0:0].copy()
    legacy_columns = {
        "legacy_dr9_status": "missing",
        "legacy_dr9_sep_arcsec": np.nan,
        "legacy_dr9_type": None,
        "legacy_dr9_objid": None,
        "legacy_gmag": np.nan,
        "legacy_rmag": np.nan,
        "legacy_zmag": np.nan,
    }
    for column, default in legacy_columns.items():
        final[column] = default

    # Legacy photometry is only needed for the final FWHM-selected candidates.
    # Prefer catalog coordinates, then fall back to the measured detection position.
    for idx in passed.index:
        query_ra = _safe_float(host_ra.loc[idx] if host_ra is not None else np.nan)
        query_dec = _safe_float(host_dec.loc[idx] if host_dec is not None else np.nan)
        if not np.isfinite(query_ra) or not np.isfinite(query_dec):
            query_ra = _safe_float(detection_ra.loc[idx] if detection_ra is not None else np.nan)
            query_dec = _safe_float(detection_dec.loc[idx] if detection_dec is not None else np.nan)
        legacy = _fetch_legacy_dr9_photometry(query_ra, query_dec)
        for column, value in legacy.items():
            final.at[idx, column] = value
        if pd.isna(final.at[idx, "host_rmag"]) and pd.notna(final.at[idx, "legacy_rmag"]):
            final.at[idx, "host_rmag"] = final.at[idx, "legacy_rmag"]
            final.at[idx, "host_rmag_source"] = "Legacy Survey DR9"
            final.at[idx, "host_rmag_catalog"] = "Legacy Survey DR9"
            final.at[idx, "host_rmag_band"] = "r"

    passed = final.loc[passed.index].copy()
    uncertain = final.loc[uncertain.index].copy()
    color_rows = []
    stage3_images_path = out_root / "stage3" / "april_stage3_pretrigger_mag_images.csv"
    stage3_images = load_csv_if_exists(stage3_images_path)
    if stage3_images is not None and not passed.empty:
        final_ids = {
            normalize_object_id(value)
            for value in passed["object_id"].tolist()
        }
        for object_id, group in stage3_images[
            (stage3_images.get("phase", "") == "posttrigger")
            & (stage3_images["object_id"].astype(str).map(normalize_object_id).isin(final_ids))
        ].groupby(stage3_images["object_id"].astype(str).map(normalize_object_id)):
            measurements_by_filter: dict[str, list[dict]] = {}
            for _, item in group.iterrows():
                filt = str(item.get("filter", "")).lower()
                mag = _safe_float(item.get("mag", np.nan))
                upper_limit = _safe_float(item.get("upper_limit", np.nan))
                obs_time = _parse_filefracday_datetime(item.get("filefracday"))
                if filt not in {"zg", "zr", "zi"} or pd.isna(obs_time):
                    continue
                is_detection = str(item.get("detection", "")).lower() == "true"
                if is_detection and np.isfinite(mag):
                    measurements_by_filter.setdefault(filt[-1], []).append(
                        {"record": item, "time": obs_time, "value": mag, "kind": "measurement"}
                    )
                elif not is_detection and np.isfinite(upper_limit):
                    measurements_by_filter.setdefault(filt[-1], []).append(
                        {"record": item, "time": obs_time, "value": upper_limit, "kind": "upper_limit"}
                    )
            for first_filter, second_filter in (("g", "r"), ("g", "i"), ("r", "i")):
                for first in measurements_by_filter.get(first_filter, []):
                    for second in measurements_by_filter.get(second_filter, []):
                        if first["kind"] == "upper_limit" and second["kind"] == "upper_limit":
                            # Two non-detections do not constrain their color
                            # without an additional flux or light-curve model.
                            continue
                        first_err = _safe_float(first["record"].get("mag_err", np.nan))
                        second_err = _safe_float(second["record"].get("mag_err", np.nan))
                        color_err = (
                            float(np.hypot(first_err, second_err))
                            if np.isfinite(first_err) and np.isfinite(second_err)
                            else np.nan
                        )
                        color_name = f"{first_filter}-{second_filter}"
                        color_mag = float(first["value"] - second["value"])
                        if first["kind"] == "measurement" and second["kind"] == "measurement":
                            limit_type = "measurement"
                        elif first["kind"] == "measurement":
                            # m_second >= limit_second, hence color <= bound.
                            limit_type = "upper_limit"
                        else:
                            # m_first >= limit_first, hence color >= bound.
                            limit_type = "lower_limit"
                        reference = {
                            "g-r": -0.3,
                            "r-i": -0.2,
                        }.get(color_name, np.nan)
                        color_rows.append({
                            "object_id": object_id,
                            "color": color_name,
                            "first_filter": first_filter,
                            "second_filter": second_filter,
                            "first_image_path": first["record"].get("image_path", ""),
                            "second_image_path": second["record"].get("image_path", ""),
                            "first_observation_utc": first["time"].isoformat(),
                            "second_observation_utc": second["time"].isoformat(),
                            "delta_t_days": abs(
                                (first["time"] - second["time"]).total_seconds()
                            ) / 86400.0,
                            "first_mag_or_limit": first["value"],
                            "second_mag_or_limit": second["value"],
                            "color_mag": color_mag,
                            "color_err": color_err,
                            "color_limit_type": limit_type,
                            "color_limit_value": color_mag,
                            "gw170817_reference_color": reference,
                            "color_minus_gw170817": (
                                color_mag - reference if np.isfinite(reference) else np.nan
                            ),
                        })
    colors_path = stage4_dir / "april_stage4_final_candidate_colors.csv"
    write_csv(pd.DataFrame(color_rows), colors_path)
    color_counts = pd.DataFrame(color_rows).groupby("object_id").size() if color_rows else pd.Series(dtype="int64")
    final["n_color_pairs"] = final["object_id"].astype(str).map(
        color_counts
    ).fillna(0).astype(int)
    passed["n_color_pairs"] = passed["object_id"].astype(str).map(
        color_counts
    ).fillna(0).astype(int)
    write_csv(final, host_objects_path)
    write_csv(passed, final_candidates_path)
    write_csv(uncertain, uncertain_candidates_path)
    return final, passed


def stage4_host_rmag(*, stage3_objects: pd.DataFrame, final_class_path: Path, out_dir: Path, host_mag_threshold: float = 22.5, resume: bool = True):
    final_class_df = pd.read_csv(final_class_path, low_memory=False)
    enriched_df, final_df = stage4_host_mag(stage3_objects, final_class_df, out_dir, host_rmag_threshold=host_mag_threshold, resume=resume)
    return enriched_df, final_df


def save_stage4_results(
    final_candidates: pd.DataFrame,
    final_class_df: pd.DataFrame,
    out_root: Path,
    *,
    max_triplets_per_object: int = 10,
    trigger_datetime=None,
    trigger_label: str = "Trigger",
):
    """Save normalized triplet plots, 1D/3D profiles and SNR light curves for each final candidate.

    Files are written under `out_root/stage4/results/<row_or_object_id>/`.
    This function tries to be best-effort (won't raise if plotting libs are missing).
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        logging.warning("matplotlib not available; skipping result-plot generation")
        return
    import numpy as _np
    import pandas as _pd

    results_root = ensure_dir(out_root / "stage4" / "results")
    triplet_root = out_root / "stage2" / "triplets"
    rownum_map = _build_final_rownum_map(final_class_df)

    # load stage1 images for light curves if available
    stage1_images_path = out_root / "stage1" / "april_stage1_quadratic_snr_images.csv"
    stage1_images = None
    if stage1_images_path.exists():
        try:
            stage1_images = _pd.read_csv(stage1_images_path, low_memory=False)
        except Exception:
            stage1_images = None

    stage3_images_path = out_root / "stage3" / "april_stage3_pretrigger_mag_images.csv"
    stage3_images = None
    if stage3_images_path.exists():
        try:
            stage3_images = _pd.read_csv(stage3_images_path, low_memory=False)
        except Exception as exc:
            logging.warning(
                "Could not load stage-3 photometry for light curves from %s: %s",
                stage3_images_path,
                exc,
            )

    triplet_source_positions = {}
    triplet_objects_path = out_root / "stage2" / "triplets" / "april_stage2_triplets_objects.csv"
    if triplet_objects_path.exists():
        try:
            triplet_objects = _pd.read_csv(triplet_objects_path, low_memory=False)
            required_position_columns = {"triplet_path", "diff_source_x_px", "diff_source_y_px"}
            if required_position_columns.issubset(triplet_objects.columns):
                for _, triplet_row in triplet_objects.iterrows():
                    source_x = _pd.to_numeric(triplet_row["diff_source_x_px"], errors="coerce")
                    source_y = _pd.to_numeric(triplet_row["diff_source_y_px"], errors="coerce")
                    if _np.isfinite(source_x) and _np.isfinite(source_y):
                        triplet_path_key = str(Path(str(triplet_row["triplet_path"])).resolve())
                        triplet_source_positions[triplet_path_key] = (float(source_x), float(source_y))
        except Exception:
            logging.warning("Could not load exact source positions from %s", triplet_objects_path)

    # final_candidates may be None or an empty DataFrame; handle explicitly
    if final_candidates is None or (hasattr(final_candidates, 'empty') and final_candidates.empty):
        logging.info("No final candidates to save results for")
        return

    for _, cand in final_candidates.iterrows():
        raw_oid = cand.get("object_id")
        oid = normalize_object_id(raw_oid)
        label = rownum_map.get(oid, oid)
        obj_dir = ensure_dir(results_root / str(label))

        # find triplet.npy files for this candidate
        triplet_files = []
        if max_triplets_per_object > 0:
            candidate_dir = triplet_root / str(label)
            if candidate_dir.exists():
                triplet_files = list(candidate_dir.rglob("triplet.npy"))
            if not triplet_files:
                # fallback: search anywhere for directory matching oid
                triplet_files = list(triplet_root.rglob(f"*{oid}*/triplet.npy"))

        # Each triplet lives in a directory named after its source diff image
        # (parent dir name == diff_path.stem), which encodes the epoch/filter.
        # rglob() order is filesystem-dependent and not chronological, so sort
        # by observation date here -- otherwise "_0", "_1", "_2" in the saved
        # filenames don't correspond to anything meaningful or reproducible.
        def _triplet_sort_key(tpath):
            parsed = parse_diff_name(tpath.parent.name) or {}
            obs_dt = _parse_filefracday_datetime(parsed.get("filefracday"))
            return (pd.Timestamp.max if pd.isna(obs_dt) else obs_dt, tpath.parent.name)

        triplet_files = sorted(triplet_files, key=_triplet_sort_key)

        # Record which image each saved index corresponds to, so triplet_norm_0.png
        # etc. can be traced back to a specific epoch/filter after the fact.
        triplet_index_rows = []
        for i, tpath in enumerate(triplet_files[:max_triplets_per_object]):
            parsed = parse_diff_name(tpath.parent.name) or {}
            obs_dt = _parse_filefracday_datetime(parsed.get("filefracday"))
            source_position = triplet_source_positions.get(str(tpath.resolve()))
            triplet_center = None
            if source_position is not None:
                source_x, source_y = source_position
                triplet_shape = _np.load(tpath, mmap_mode="r").shape[0]
                triplet_center = (
                    triplet_shape // 2 + source_x - round(source_x),
                    triplet_shape // 2 + source_y - round(source_y),
                )
            triplet_index_rows.append({
                "index": i,
                "diff_image": tpath.parent.name,
                "filter": parsed.get("filter", ""),
                "filefracday": parsed.get("filefracday", ""),
                "obs_datetime": obs_dt,
                "triplet_path": str(tpath),
                "stage1_centroid_x_px": triplet_center[0] if triplet_center is not None else _np.nan,
                "stage1_centroid_y_px": triplet_center[1] if triplet_center is not None else _np.nan,
            })
        if triplet_index_rows:
            _pd.DataFrame(triplet_index_rows).to_csv(obj_dir / "triplet_index.csv", index=False)

        for i, tpath in enumerate(triplet_files[:max_triplets_per_object]):
            try:
                trip = _np.load(tpath)
                mask_path = tpath.with_name("triplet_valid_mask.npy")
                if mask_path.exists():
                    valid_masks = _np.asarray(_np.load(mask_path), dtype=bool)
                    if valid_masks.shape != trip.shape:
                        raise ValueError(f"Invalid mask shape {valid_masks.shape} for {tpath}")
                else:
                    # Compatibility for manually supplied old triplets. The
                    # sentinel used by _crop and non-finite values are invalid.
                    valid_masks = _np.isfinite(trip) & (trip != -5000.0)
            except Exception:
                continue

            # normalize each plane (median/std)
            planes = []
            for k in range(trip.shape[-1]):
                p = _np.asarray(trip[..., k], dtype=float)
                p[~valid_masks[..., k]] = _np.nan
                if not _np.any(_np.isfinite(p)):
                    raise ValueError(f"Triplet plane {k} is fully masked")
                m = _np.nanmedian(p)
                s = _np.nanstd(p)
                if not _np.isfinite(s) or s == 0:
                    s = 1.0
                planes.append((p-m)/s)

            source_position = triplet_source_positions.get(str(tpath.resolve()))
            if source_position is None:
                logging.warning("No stage-1 source position for centered profiles of %s", tpath)
                profile_cx = profile_cy = _np.nan
            else:
                source_x, source_y = source_position
                profile_cx = trip.shape[1] // 2 + source_x - round(source_x)
                profile_cy = trip.shape[0] // 2 + source_y - round(source_y)

            # 3-panel normalized image with flipped ref and smoothed row below
            try:
                def _smooth(img, sigma=1.0):
                    try:
                        from scipy.ndimage import gaussian_filter

                        return gaussian_filter(img, sigma=sigma)
                    except Exception:
                        # fallback simple 3x3 uniform smoothing
                        kernel = _np.ones((3, 3), dtype=float) / 9.0
                        padded = _np.pad(img, 1, mode="reflect")
                        out = _np.empty_like(img)
                        for yy in range(img.shape[0]):
                            for xx in range(img.shape[1]):
                                out[yy, xx] = (padded[yy:yy+3, xx:xx+3] * kernel).sum()
                        return out

                # flip the reference horizontally for display
                display_planes = [p.copy() for p in planes]
                if len(display_planes) > 1:
                    display_planes[1] = _np.fliplr(display_planes[1])

                # make smoothed versions
                smoothed = [_smooth(p) for p in display_planes]

                fig, axes = plt.subplots(2, 3, figsize=(10, 8))
                im = None

                from astropy.visualization import ZScaleInterval

                zscale = ZScaleInterval(contrast=0.2)
                limits = []
                for plane in display_planes:
                    finite_plane = plane[_np.isfinite(plane)]
                    try:
                        vmin, vmax = zscale.get_limits(finite_plane)
                    except Exception:
                        vmin, vmax = -3.0, 3.0
                    if not _np.isfinite(vmin) or not _np.isfinite(vmax) or vmin >= vmax:
                        vmin, vmax = -3.0, 3.0
                    limits.append((float(vmin), float(vmax)))

                titles = ("sci", "ref", "diff")
                for col in range(3):
                    ax = axes[0, col]
                    vmin, vmax = limits[col]
                    im = ax.imshow(display_planes[col], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
                    ax.set_title(titles[col])
                    ax.axis("off")
                    ax2 = axes[1, col]
                    ax2.imshow(smoothed[col], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
                    ax2.set_title(f"{titles[col]} (smoothed)")
                    ax2.axis("off")

                    # WCS-based cropping centers the SNR source in the diff panels.
                    if col == 0 or col == 2:
                        center = display_planes[col].shape[0] // 2
                        gap = 5
                        arm_len = 12
                        for crosshair_ax in (ax, ax2):
                            crosshair_ax.plot([center, center], [center - arm_len, center - gap], color="red", linewidth=1.8, solid_capstyle="butt")
                            crosshair_ax.plot([center, center], [center + gap, center + arm_len], color="red", linewidth=1.8, solid_capstyle="butt")
                            crosshair_ax.plot([center - arm_len, center - gap], [center, center], color="red", linewidth=1.8, solid_capstyle="butt")
                            crosshair_ax.plot([center + gap, center + arm_len], [center, center], color="red", linewidth=1.8, solid_capstyle="butt")

                fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical", location="right", fraction=0.05)
                fig.savefig(obj_dir / f"triplet_norm_{i}.png", dpi=400, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                logging.debug("failed to save triplet image for %s: %s", oid, tpath)

            # 1D radial profile of diff plane
            try:
                if not _np.isfinite(profile_cx) or not _np.isfinite(profile_cy):
                    raise ValueError("Stage-1 centroid position is unavailable")
                diff = planes[-1]
                yy, xx = _np.indices(diff.shape)
                dx = xx - profile_cx
                dy = yy - profile_cy
                r = _np.sqrt(dx ** 2 + dy ** 2)
                within_radius = r <= 15.0
                signed_radius = _np.sign(dx) * _np.floor(r).astype(int)
                profile_valid = within_radius & valid_masks[..., -1] & _np.isfinite(diff)
                signed_radius = signed_radius[profile_valid].astype(int)
                vals = diff[profile_valid]
                bins = _np.arange(-15, 16)
                profile = _np.array([
                    _np.nanmean(vals[signed_radius == radius])
                    if _np.any(signed_radius == radius)
                    else _np.nan
                    for radius in bins
                ])
                fig = plt.figure(figsize=(4, 3))
                plt.plot(bins, profile, marker=".")
                fit_mask = _np.isfinite(profile)
                if _np.count_nonzero(fit_mask) >= 5:
                    try:
                        fitted_fwhm = float(cand.get("fwhm_px", _np.nan))
                        if not _np.isfinite(fitted_fwhm) or fitted_fwhm <= 0:
                            raise ValueError("Stage-4 Gaussian FWHM is unavailable")
                        fit_sigma = fitted_fwhm / 2.35482
                        fit_baseline = float(_np.nanmedian(profile[fit_mask][-5:]))
                        central_profile = profile[bins == 0]
                        if central_profile.size == 0 or not _np.isfinite(central_profile[0]):
                            raise ValueError("radial profile center is unavailable")
                        fit_amplitude = float(central_profile[0] - fit_baseline)
                        fit_fwhm = 2.35482 * fit_sigma
                        dense_bins = _np.linspace(-15, 15, 400)
                        plt.plot(
                            dense_bins,
                            fit_baseline + fit_amplitude * _np.exp(
                                -0.5 * (dense_bins / fit_sigma) ** 2
                            ),
                            color="tab:red",
                            linewidth=1.5,
                            label=f"Stage-4 Gaussian fit (FWHM={fit_fwhm:.2f} px)",
                        )
                        plt.legend(fontsize=7)
                    except Exception:
                        logging.debug("failed Gaussian overlay for radial profile %s", oid)
                plt.xlim(-15, 15)
                plt.axvline(0, color="black", linewidth=0.8, alpha=0.6)
                plt.xlabel("signed radius (px; left / right of center)")
                plt.ylabel("mean normalized diff")
                plt.savefig(obj_dir / f"radial_profile_{i}.png", bbox_inches="tight", dpi=400)
                plt.close(fig)
            except Exception:
                logging.debug("failed radial profile for %s", oid)

            # 3D surface plot of diff
            try:
                if not _np.isfinite(profile_cx) or not _np.isfinite(profile_cy):
                    raise ValueError("Stage-1 centroid position is unavailable")
                fig = plt.figure(figsize=(5, 4))
                ax = fig.add_subplot(111, projection="3d")
                xs = _np.arange(diff.shape[1])
                ys = _np.arange(diff.shape[0])
                X, Y = _np.meshgrid(xs, ys)
                X_offset = X - profile_cx
                Y_offset = Y - profile_cy
                surface_radius = _np.sqrt(X_offset ** 2 + Y_offset ** 2)
                surface_mask = surface_radius <= 15.0
                surface_mask &= valid_masks[..., -1] & _np.isfinite(diff)
                surface_values = diff[surface_mask]
                z_min, z_max = _np.nanpercentile(surface_values, [5, 95])
                z_span = max(float(z_max - z_min), 1e-6)
                clipped_surface = _np.clip(diff, z_min, z_max)
                surface = _np.ma.masked_where(~surface_mask, clipped_surface)
                marker_z = float(z_max) + 0.2 * z_span
                center_y = int(round(profile_cy))
                center_x = int(round(profile_cx))
                if (
                    not (0 <= center_y < diff.shape[0] and 0 <= center_x < diff.shape[1])
                    or not valid_masks[center_y, center_x, -1]
                    or not _np.isfinite(diff[center_y, center_x])
                ):
                    raise ValueError("Diff center is masked or invalid")
                center_z = float(diff[center_y, center_x])
                ax.plot_surface(
                    X_offset, Y_offset, surface, cmap="viridis", linewidth=0,
                    antialiased=False, alpha=0.72,
                )
                ax.plot([0, 0], [0, 0], [center_z, marker_z], color="red", linestyle="--", linewidth=2)
                ax.scatter(
                    [0], [0], [marker_z], color="red", edgecolors="white",
                    marker="*", s=180, depthshade=False, label="stage1 centroid",
                )
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)
                ax.set_zlim(float(z_min), marker_z + 0.08 * z_span)
                ax.set_xlabel("dx (px)", labelpad=2, fontsize=9)
                ax.set_ylabel("dy (px)", labelpad=2, fontsize=9)
                ax.set_zlabel("normalized diff (5-95% clipped)", labelpad=2, fontsize=9)
                ax.tick_params(axis="both", which="major", labelsize=8)
                ax.tick_params(axis="z", which="major", labelsize=8)
                ax.set_box_aspect((1, 1, 0.7))
                ax.view_init(elev=30, azim=-55)
                ax.legend(loc="upper left", fontsize=8)
                plt.savefig(obj_dir / f"surface_{i}.png", bbox_inches="tight", dpi=400)
                plt.close(fig)
            except Exception:
                logging.debug("failed 3d surface for %s", oid)

        # light curve: magnitude vs time from stage3 pretrigger mag images (preferred)
        try:
            stage1_detections_without_mag = 0
            if stage3_images is not None:
                ssel = stage3_images[
                    stage3_images["object_id"].astype(str).map(normalize_object_id) == oid
                ].copy()
            else:
                ssel = _pd.DataFrame()

            if ssel.empty and stage1_images is not None:
                ssel = stage1_images[stage1_images["object_id"].astype(str).map(normalize_object_id) == oid].copy()

            if stage1_images is not None and not stage1_images.empty:
                stage1_for_object = stage1_images[
                    stage1_images["object_id"].astype(str).map(normalize_object_id) == oid
                ].copy()
                stage1_for_object["mag"] = _pd.to_numeric(
                    stage1_for_object.get("mag"), errors="coerce"
                )
                stage1_for_object["mag_err"] = _pd.to_numeric(
                    stage1_for_object.get("mag_err"), errors="coerce"
                )
                detected = stage1_for_object.get("detection", _pd.Series(False, index=stage1_for_object.index))
                stage1_for_object = stage1_for_object[
                    detected.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
                ].copy()
                if not stage1_for_object.empty:
                    stage1_detections_without_mag = int(stage1_for_object["mag"].isna().sum())
                    def _plot_observation_key(row):
                        def _normalize(value):
                            if _pd.isna(value):
                                return ""
                            text = str(value).strip()
                            return text[:-2] if text.endswith(".0") else text

                        parsed = parse_diff_name(Path(str(row.get("image_path", ""))).name) or {}
                        return (
                            _normalize(row.get("filefracday", parsed.get("filefracday"))),
                            _normalize(row.get("filter", parsed.get("filter"))),
                            _normalize(row.get("field", parsed.get("field"))),
                            _normalize(row.get("ccdid", parsed.get("ccdid"))),
                            _normalize(row.get("qid", parsed.get("qid"))),
                        )

                    stage1_for_object["upper_limit"] = _np.nan
                    detection_keys = set(
                        stage1_for_object.apply(_plot_observation_key, axis=1)
                    )
                    if not ssel.empty:
                        ssel = ssel[
                            ~ssel.apply(_plot_observation_key, axis=1).isin(detection_keys)
                        ].copy()
                    stage1_plot_detections = stage1_for_object[
                        stage1_for_object["mag"].notna()
                    ].copy()
                    ssel = _pd.concat(
                        [ssel, stage1_plot_detections], ignore_index=True, sort=False
                    )

            if not ssel.empty:
                # prefer magnitudes if available, otherwise use SNR as fallback
                ssel["mag"] = _pd.to_numeric(ssel.get("mag"), errors="coerce")
                ssel["mag_err"] = _pd.to_numeric(ssel.get("mag_err"), errors="coerce")
                ssel["upper_limit"] = _pd.to_numeric(ssel.get("upper_limit"), errors="coerce")
                ssel["snr_empirical"] = _pd.to_numeric(ssel.get("snr_empirical"), errors="coerce")

                def _get_filefracday(row):
                    value = row.get("filefracday")
                    if _pd.notna(value):
                        text = str(value).strip()
                        if text.endswith(".0"):
                            text = text[:-2]
                        if len(text) >= 8 and text[:8].isdigit():
                            return text
                    parsed = parse_diff_name(Path(str(row.get("image_path", ""))).name)
                    return parsed.get("filefracday") if parsed else None

                # Extract filefracday-derived fractional day for plotting (day-of-year + fraction)
                def _extract_t_frac(row):
                    try:
                        filefracday = _get_filefracday(row)
                        if filefracday:
                            dt = _parse_filefracday_datetime(filefracday)
                            if pd.notna(dt):
                                return float(dt.dayofyear) + (dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6) / 86400.0
                    except Exception:
                        pass
                    return _np.nan

                ssel["t_frac"] = ssel.apply(_extract_t_frac, axis=1)

                # Extract human date and filter label for xticks and color grouping
                def _extract_date_filter_label(row):
                    try:
                        ffd = _get_filefracday(row)
                        if ffd:
                            if len(ffd) >= 8:
                                d = ffd[:8]
                                filt = row.get("filter")
                                if _pd.isna(filt):
                                    parsed = parse_diff_name(Path(str(row.get("image_path", ""))).name)
                                    filt = parsed.get("filter", "?") if parsed else "?"
                                return f"{d[0:4]}-{d[4:6]}-{d[6:8]} ({filt})"
                    except Exception:
                        pass
                    return Path(str(row.get("image_path", ""))).name

                ssel["date_label"] = ssel.apply(_extract_date_filter_label, axis=1)

                # Prefer a precise datetime parsed from filefracday when available
                def _extract_obs_datetime(row):
                    try:
                        filefracday = _get_filefracday(row)
                        if filefracday:
                            return _parse_filefracday_datetime(filefracday)
                    except Exception:
                        pass
                    return _pd.NaT

                ssel["obs_datetime"] = ssel.apply(_extract_obs_datetime, axis=1)

                # For any rows missing precise datetime, fall back to the date parsed from the label
                mask_na = ssel["obs_datetime"].isna()
                if mask_na.any():
                    ssel.loc[mask_na, "obs_datetime"] = ssel.loc[mask_na, "date_label"].apply(lambda dl: _pd.to_datetime(dl[:10], errors="coerce") if isinstance(dl, str) else _pd.NaT)

                # Extract filter from filename (more robust than regex on date_label)
                def _extract_filter(row):
                    value = row.get("filter")
                    if _pd.notna(value):
                        return str(value).lower()
                    try:
                        parsed = parse_diff_name(Path(str(row.get("image_path", ""))).name)
                        if parsed:
                            return parsed.get("filter")
                    except Exception:
                        pass
                    return None

                ssel["filt"] = ssel.apply(_extract_filter, axis=1)

                # plotting using real datetimes on X axis and coloring by filter
                filter_colors = {"zg": "green", "zr": "red", "zi": "orange"}
                ssel_plot = ssel[ssel["obs_datetime"].notna()].copy()
                if trigger_datetime is not None and pd.notna(trigger_datetime):
                    trigger = pd.to_datetime(trigger_datetime)
                    cutoff = (trigger - pd.Timedelta(days=1)).normalize()
                    ssel_plot = ssel_plot[ssel_plot["obs_datetime"] >= cutoff].copy()
                if not ssel_plot.empty:
                    import matplotlib.dates as mdates
                    from matplotlib.dates import DateFormatter

                    fig, ax = plt.subplots(figsize=(9, 4))
                    for filt, color in filter_colors.items():
                        mask_f = ssel_plot["filt"] == filt
                        if mask_f.any():
                            det = ssel_plot[mask_f & ssel_plot["mag"].notna() & (ssel_plot["snr_empirical"] >= 3)]
                            if not det.empty:
                                ax.errorbar(det["obs_datetime"].values, det["mag"].values, yerr=det.get("mag_err"), fmt="o", color=color, label=f"Обнаружение ({filt})")
                            ul = ssel_plot[mask_f & ssel_plot["upper_limit"].notna()]
                            if not ul.empty:
                                ax.scatter(ul["obs_datetime"].values, ul["upper_limit"].values, marker="v", color=color, s=60, alpha=0.6, label=f"Верхний предел ({filt})")

                    # xticks: use a date locator/formatter so labels do not overlap
                    import matplotlib.dates as mdates
                    from matplotlib.dates import DateFormatter

                    locator = mdates.AutoDateLocator()
                    # Keep explicit year-month-day labels and suppress time text.
                    formatter = mdates.DateFormatter("%Y-%m-%d")
                    ax.xaxis.set_major_locator(locator)
                    ax.xaxis.set_major_formatter(formatter)
                    fig.autofmt_xdate(rotation=45, ha='right')

                    if trigger_datetime is not None and pd.notna(trigger_datetime):
                        trigger = pd.to_datetime(trigger_datetime)
                        ax.axvline(trigger, color="black", linestyle="--", linewidth=1.2, alpha=0.8, label=trigger_label)

                    ax.invert_yaxis()
                    plotted_mags = _pd.concat(
                        [
                            ssel_plot.loc[ssel_plot["mag"].notna(), "mag"],
                            ssel_plot.loc[ssel_plot["upper_limit"].notna(), "upper_limit"],
                        ],
                        ignore_index=True,
                    )
                    plotted_mags = _pd.to_numeric(plotted_mags, errors="coerce").dropna()
                    if not plotted_mags.empty:
                        mag_min = float(plotted_mags.min())
                        mag_max = float(plotted_mags.max())
                        margin = max(0.25, 0.08 * (mag_max - mag_min))
                        ax.set_ylim(mag_max + margin, mag_min - margin)

                    ax.set_ylabel("Звёздная величина")
                    ax.set_title(f"Кривая блеска {oid}")
                    ax.legend()
                    if stage1_detections_without_mag:
                        ax.text(
                            0.01,
                            0.02,
                            f"Срабатывания этапа 1 без измеряемой величины: {stage1_detections_without_mag}",
                            transform=ax.transAxes,
                            fontsize=8,
                            va="bottom",
                        )
                    plt.tight_layout()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(obj_dir / "lightcurve_mag.png", bbox_inches="tight", dpi=400)
                    plt.close(fig)
        except Exception:
            logging.debug("failed to save light curve for %s", oid)




def stage5_minor_planet_check(
    final_candidates: pd.DataFrame,
    out_root: Path,
    *,
    radius_arcmin: float = 5.0,
    limiting_magnitude: float = 24.0,
    observatory_code: str = "644",
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-match final candidates against known solar-system objects in SkyBoT."""
    stage5_dir = ensure_dir(out_root / "stage5")
    candidates_path = stage5_dir / "april_stage5_minor_planet_candidates.csv"
    matches_path = stage5_dir / "april_stage5_minor_planet_matches.csv"
    cached_candidates = load_csv_if_exists(candidates_path) if resume else None
    cached_matches = load_csv_if_exists(matches_path) if resume else None

    object_ids = (
        final_candidates["object_id"].astype(str).map(normalize_object_id).tolist()
        if "object_id" in final_candidates.columns
        else []
    )
    if (
        cached_candidates is not None
        and cached_matches is not None
        and {"object_id", "minor_planet_status", "minor_planet_radius_arcmin", "minor_planet_limit_mag", "minor_planet_observatory_code"}.issubset(cached_candidates.columns)
    ):
        cached_ids = cached_candidates["object_id"].astype(str).map(normalize_object_id).tolist()
        settings_match = (
            pd.to_numeric(cached_candidates["minor_planet_radius_arcmin"], errors="coerce").eq(float(radius_arcmin)).all()
            and pd.to_numeric(cached_candidates["minor_planet_limit_mag"], errors="coerce").eq(float(limiting_magnitude)).all()
            and cached_candidates["minor_planet_observatory_code"].astype(str).eq(str(observatory_code)).all()
        )
        retry_needed = cached_candidates["minor_planet_status"].astype(str).eq("query_error").any()
        if cached_ids == object_ids and settings_match and not retry_needed:
            return cached_candidates, cached_matches

    summary_columns = [
        "minor_planet_service",
        "minor_planet_status",
        "minor_planet_match_count",
        "minor_planet_bright_match_count",
        "minor_planet_observation_utc",
        "minor_planet_search_radius_arcmin",
        "minor_planet_radius_arcmin",
        "minor_planet_limit_mag",
        "minor_planet_observatory_code",
        "minor_planet_nearest_number",
        "minor_planet_nearest_name",
        "minor_planet_nearest_type",
        "minor_planet_nearest_vmag",
        "minor_planet_nearest_sep_arcsec",
        "minor_planet_nearest_posunc_arcsec",
        "minor_planet_nearest_ra_rate_arcsec_per_hour",
        "minor_planet_nearest_dec_rate_arcsec_per_hour",
        "minor_planet_nearest_link",
        "minor_planet_query_error",
    ]
    match_columns = [
        "object_id",
        "candidate_image_path",
        "observation_utc",
        "number",
        "name",
        "type",
        "vmag",
        "separation_arcsec",
        "posunc_arcsec",
        "ra_rate_arcsec_per_hour",
        "dec_rate_arcsec_per_hour",
        "geocentric_distance_au",
        "heliocentric_distance_au",
        "link",
    ]
    if final_candidates.empty:
        empty_candidates = final_candidates.copy()
        for column in summary_columns:
            empty_candidates[column] = pd.Series(dtype="object")
        empty_matches = pd.DataFrame(columns=match_columns)
        write_csv(empty_candidates, candidates_path)
        write_csv(empty_matches, matches_path)
        return empty_candidates, empty_matches

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astroquery.imcce import Skybot

    stage1_path = out_root / "stage1" / "april_stage1_quadratic_snr_images.csv"
    stage1 = load_csv_if_exists(stage1_path)
    best_stage1: dict[str, dict] = {}
    if stage1 is not None and not stage1.empty and {"object_id", "snr", "snr_empirical"}.issubset(stage1.columns):
        stage1 = stage1.copy()
        stage1["object_id"] = stage1["object_id"].astype(str).map(normalize_object_id)
        stage1["snr"] = pd.to_numeric(stage1["snr"], errors="coerce")
        stage1["snr_empirical"] = pd.to_numeric(stage1["snr_empirical"], errors="coerce")
        best_rows = stage1.sort_values("snr_empirical", ascending=False).drop_duplicates("object_id")
        best_stage1 = best_rows.set_index("object_id").to_dict("index")

    def _plain_value(table, column: str, index: int, unit=None):
        if column not in table.colnames:
            return np.nan
        value = table[column][index]
        if np.ma.is_masked(value):
            return np.nan
        if unit is not None and hasattr(value, "to_value"):
            return float(value.to_value(unit))
        value = getattr(value, "value", value)
        if isinstance(value, np.generic):
            value = value.item()
        return value

    summary_rows = []
    match_rows = []
    for _, candidate in final_candidates.iterrows():
        object_id = normalize_object_id(candidate.get("object_id"))
        stage1_row = best_stage1.get(object_id, {})
        image_value = stage1_row.get("image_path", candidate.get("diff_path", ""))
        image_path = Path(str(image_value)) if image_value else None
        parsed = parse_diff_name(image_path.name) if image_path is not None else None
        filefracday = stage1_row.get("filefracday")
        if filefracday is None or pd.isna(filefracday) or not str(filefracday).strip():
            filefracday = parsed.get("filefracday") if parsed else None
        obs_datetime = (
            _parse_filefracday_datetime(str(filefracday))
            if filefracday is not None
            else pd.NaT
        )

        ra = _safe_float(stage1_row.get("centroid_ra", candidate.get("detection_ra", np.nan)))
        dec = _safe_float(stage1_row.get("centroid_dec", candidate.get("detection_dec", np.nan)))
        if (not np.isfinite(ra) or not np.isfinite(dec)) and parsed is not None:
            ra = _safe_float(parsed.get("ra"))
            dec = _safe_float(parsed.get("dec"))

        record = candidate.to_dict()
        record.update(
            {
                "minor_planet_service": "IMCCE SkyBoT",
                "minor_planet_status": "missing_astrometry",
                "minor_planet_match_count": 0,
                "minor_planet_bright_match_count": 0,
                "minor_planet_observation_utc": obs_datetime.isoformat() if pd.notna(obs_datetime) else "",
                "minor_planet_search_radius_arcmin": float(radius_arcmin),
                "minor_planet_radius_arcmin": float(radius_arcmin),
                "minor_planet_limit_mag": float(limiting_magnitude),
                "minor_planet_observatory_code": str(observatory_code),
                "minor_planet_nearest_number": "",
                "minor_planet_nearest_name": "",
                "minor_planet_nearest_type": "",
                "minor_planet_nearest_vmag": np.nan,
                "minor_planet_nearest_sep_arcsec": np.nan,
                "minor_planet_nearest_posunc_arcsec": np.nan,
                "minor_planet_nearest_ra_rate_arcsec_per_hour": np.nan,
                "minor_planet_nearest_dec_rate_arcsec_per_hour": np.nan,
                "minor_planet_nearest_link": "",
                "minor_planet_query_error": "",
            }
        )

        if np.isfinite(ra) and np.isfinite(dec) and pd.notna(obs_datetime):
            try:
                result = Skybot.cone_search(
                    SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"),
                    float(radius_arcmin) * u.arcmin,
                    Time(obs_datetime.to_pydatetime(), scale="utc"),
                    location=str(observatory_code),
                    position_error=120.0 * u.arcsec,
                    find_planets=False,
                    find_asteroids=True,
                    find_comets=True,
                    cache=resume,
                )
                record["minor_planet_match_count"] = len(result)
                record["minor_planet_status"] = "match" if len(result) else "no_match"
                candidate_matches = []
                for match_index in range(len(result)):
                    match = {
                        "object_id": object_id,
                        "candidate_image_path": str(image_path) if image_path is not None else "",
                        "observation_utc": record["minor_planet_observation_utc"],
                        "number": _plain_value(result, "Number", match_index),
                        "name": _plain_value(result, "Name", match_index),
                        "type": _plain_value(result, "Type", match_index),
                        "vmag": _plain_value(result, "V", match_index),
                        "separation_arcsec": _plain_value(result, "centerdist", match_index, u.arcsec),
                        "posunc_arcsec": _plain_value(result, "posunc", match_index, u.arcsec),
                        "ra_rate_arcsec_per_hour": _plain_value(result, "RA_rate", match_index, u.arcsec / u.hour),
                        "dec_rate_arcsec_per_hour": _plain_value(result, "DEC_rate", match_index, u.arcsec / u.hour),
                        "geocentric_distance_au": _plain_value(result, "geodist", match_index, u.au),
                        "heliocentric_distance_au": _plain_value(result, "heliodist", match_index, u.au),
                        "link": _plain_value(result, "externallink", match_index),
                    }
                    vmag = _safe_float(match["vmag"])
                    if not np.isfinite(vmag) or vmag <= float(limiting_magnitude):
                        record["minor_planet_bright_match_count"] += 1
                    candidate_matches.append(match)
                    match_rows.append(match)

                if candidate_matches:
                    separations = np.asarray(
                        [ _safe_float(match["separation_arcsec"]) for match in candidate_matches ],
                        dtype=float,
                    )
                    finite_separations = np.isfinite(separations)
                    nearest_index = (
                        int(np.nanargmin(separations))
                        if finite_separations.any()
                        else 0
                    )
                    nearest = candidate_matches[nearest_index]
                    record.update(
                        {
                            "minor_planet_nearest_number": nearest["number"],
                            "minor_planet_nearest_name": nearest["name"],
                            "minor_planet_nearest_type": nearest["type"],
                            "minor_planet_nearest_vmag": nearest["vmag"],
                            "minor_planet_nearest_sep_arcsec": nearest["separation_arcsec"],
                            "minor_planet_nearest_posunc_arcsec": nearest["posunc_arcsec"],
                            "minor_planet_nearest_ra_rate_arcsec_per_hour": nearest["ra_rate_arcsec_per_hour"],
                            "minor_planet_nearest_dec_rate_arcsec_per_hour": nearest["dec_rate_arcsec_per_hour"],
                            "minor_planet_nearest_link": nearest["link"],
                        }
                    )
            except Exception as exc:
                record["minor_planet_status"] = "query_error"
                record["minor_planet_query_error"] = str(exc)
                logging.warning("SkyBoT query failed for object %s: %s", object_id, exc)

        summary_rows.append(record)

    candidates_df = pd.DataFrame(summary_rows)
    matches_df = pd.DataFrame(match_rows, columns=match_columns)
    write_csv(candidates_df, candidates_path)
    write_csv(matches_df, matches_path)
    return candidates_df, matches_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the April candidate pipeline")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config YAML")
    parser.add_argument(
        "--stage",
        choices=("all", "stage0", "stage1", "stage2-triplets", "stage2-braai", "stage3", "stage4", "stage5-minor-planets"),
        default="all",
        help="Run only this stage; prerequisite stage CSV outputs must already exist",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse stage outputs if they already exist")
    parser.add_argument("--force", action="store_true", help="Ignore cached stage outputs and recompute")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    resume = bool(args.resume or config.get("resume", True)) and not bool(args.force)

    trigger_date_text = str(config.get("trigger_date", "2019-04-25"))
    trigger_time_text = str(config.get("trigger_time", "")).strip()
    trigger_datetime_text = f"{trigger_date_text} {trigger_time_text}".strip() if trigger_time_text else trigger_date_text
    try:
        trigger_datetime = pd.to_datetime(trigger_datetime_text)
    except Exception:
        logging.warning("Invalid trigger datetime in config: %s", trigger_datetime_text)
        trigger_datetime = pd.NaT

    out_root = ensure_dir(resolve_cli_path(config.get("out_root", "april_candidate_pipeline_out")))
    log_path = out_root / "pipeline.log"
    logging.basicConfig(level=getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s", filename=log_path, filemode="w")

    logging.info("Starting pipeline")
    diff_root = resolve_cli_path(config.get("diff_root", "ztf_diff_cutout_imgs_catalog2"))
    sci_root = resolve_cli_path(config.get("sci_root", "ztf_sci_cutout_imgs"))
    ref_root = resolve_cli_path(config.get("ref_root", "ztf_ref_cutout_imgs"))
    final_class_csv = resolve_cli_path(config.get("final_class_csv", "final_class_table_new_(only_in_dist_bounds).csv"))
    model_path = resolve_cli_path(config.get("model_path", "braai/models/braai_d6_m9.h5"))

    def read_stage_csv(relative_path: str, producer: str) -> pd.DataFrame:
        path = out_root / relative_path
        cached = load_csv_if_exists(path)
        if cached is None:
            raise FileNotFoundError(
                f"Missing prerequisite output for --stage {args.stage}: {path}. "
                f"Run --stage {producer} first."
            )
        return cached

    def load_final_class() -> pd.DataFrame:
        return pd.read_csv(final_class_csv, low_memory=False)

    if args.stage != "all":
        logging.info("Running selected stage: %s", args.stage)
        if args.stage == "stage0":
            result = stage0_diff_index(
                diff_root,
                out_root,
                limit_images=int(config.get("limit_images", 0)),
                object_ids=config.get("object_ids", []),
                resume=resume,
                full_frames=bool(config.get("blind_search", False)),
            )
            logging.info("Stage0 complete: %s indexed images", len(result))
            return 0

        if args.stage == "stage1":
            diff_index = read_stage_csv("stage0/april_stage0_diff_index.csv", "stage0")
            stage1_images, _ = run_stage1(diff_index, out_root, config, load_final_class(), resume)
            logging.info("Stage1 complete: %s measured images", len(stage1_images))
            return 0

        if args.stage == "stage2-triplets":
            stage1_images = read_stage_csv(
                "stage1/april_stage1_quadratic_snr_images.csv", "stage1"
            )
            snr_threshold = float(config.get("snr_threshold", 3.0))
            stage1_pass = stage1_images[
                pd.to_numeric(stage1_images.get("snr_empirical"), errors="coerce")
                >= snr_threshold
            ].copy()
            manifest_df, complete_df = stage2_triplets(
                stage1_pass,
                out_root,
                sci_root=sci_root,
                ref_root=ref_root,
                final_class_df=load_final_class(),
                download_missing=bool(config.get("download_missing", True)),
                triplet_size=int(config.get("triplet_size", 63)),
                ref_flip_lr=bool(config.get("ref_flip_lr", False)),
                max_workers=int(config.get("max_workers", 4)),
                chunk_size=int(config.get("chunk_size", 200)),
                full_frames=bool(config.get("blind_search", False)),
                resume=resume,
            )
            logging.info(
                "Stage2 triplets complete: %s manifest rows, %s complete triplets",
                len(manifest_df),
                len(complete_df),
            )
            return 0

        if args.stage == "stage2-braai":
            manifest_df = read_stage_csv(
                "stage2/triplets/april_stage2_triplets_manifest.csv", "stage2-triplets"
            )
            complete_df = read_stage_csv(
                "stage2/triplets/april_stage2_triplets_objects.csv", "stage2-triplets"
            )
            scores_df, braai_pass_df = stage2_braai(
                manifest_df,
                complete_df,
                out_root,
                model_path=model_path,
                triplet_size=int(config.get("triplet_size", 63)),
                ref_flip_lr=bool(config.get("ref_flip_lr", False)),
                braai_threshold=float(config.get("braai_threshold", 0.25)),
                batch_size=int(config.get("chunk_size", 128)),
                resume=resume,
            )
            logging.info(
                "Stage2 BRAAI complete: %s scored triplets, %s passing rows",
                len(scores_df),
                len(braai_pass_df),
            )
            return 0

        if args.stage == "stage3":
            diff_index = read_stage_csv("stage0/april_stage0_diff_index.csv", "stage0")
            braai_pass_df = read_stage_csv("stage2/april_stage2_braai_objects.csv", "stage2-braai")
            _, _, stage3_objects_pass = stage3_pretrigger_magnitude(
                stage2_objects=braai_pass_df,
                diff_index=diff_index,
                diff_root=diff_root,
                final_class_path=final_class_csv,
                out_dir=out_root / "stage3",
                trigger_date=trigger_date_text,
                lookback_days=int(config.get("lookback_days", 10)),
                posttrigger_days=int(config.get("posttrigger_days", 10)),
                sigma=float(config.get("sigma", 3.0)),
                maxiters=int(config.get("maxiters", 5)),
                min_valid_pixel=float(config.get("min_valid_pixel", -5000.0)),
                download_missing=bool(config.get("download_missing", True)),
                full_frames=bool(config.get("blind_search", False)),
                resume=resume,
            )
            logging.info("Stage3 complete: %s passing objects", len(stage3_objects_pass))
            return 0

        if args.stage == "stage5-minor-planets":
            final_candidates = read_stage_csv(
                "stage4/april_stage4_final_candidates.csv", "stage4"
            )
            checked, matches = stage5_minor_planet_check(
                final_candidates,
                out_root,
                radius_arcmin=float(config.get("minor_planet_radius_arcmin", 5.0)),
                limiting_magnitude=float(config.get("minor_planet_limit_mag", 24.0)),
                observatory_code=str(config.get("minor_planet_observatory_code", "644")),
                resume=resume,
            )
            logging.info(
                "Stage5 minor-planet check complete: %s candidates checked, %s matches",
                len(checked),
                len(matches),
            )
            return 0

        stage3_objects_pass = read_stage_csv(
            "stage3/april_stage3_pretrigger_mag_objects.csv", "stage3"
        )
        final_class_df = load_final_class()
        stage4_objects, final_candidates = stage4_host_mag(
            stage3_objects_pass,
            final_class_df,
            out_root,
            host_rmag_threshold=float(config.get("host_rmag_threshold", 22.5)),
            fwhm_reduced_chi2_max=float(config.get("fwhm_reduced_chi2_max", 5.0)),
            fwhm_ratio_min=float(config.get("fwhm_ratio_min", 0.5)),
            fwhm_ratio_max=float(config.get("fwhm_ratio_max", 1.8)),
            fwhm_sigma_multiplier=float(config.get("fwhm_sigma_multiplier", 3.0)),
            psf_spike_delta_chi2_min=float(config.get("psf_spike_delta_chi2_min", 9.0)),
            psf_amplitude_snr_min=float(config.get("psf_amplitude_snr_min", 3.0)),
            source_delta_chi2_min=float(config.get("source_delta_chi2_min", 9.0)),
            psf_residual_rms_sigma_max=float(config.get("psf_residual_rms_sigma_max", 1.5)),
            candidate_snr_min=float(config.get("candidate_snr_min", 3.0)),
            candidate_snr_max=float(config.get("candidate_snr_max", 5.0)),
            psf_visibility_gate_enabled=bool(config.get("psf_visibility_gate_enabled", False)),
            psf_visibility_score_min=float(config.get("psf_visibility_score_min", 2.362838023766029)),
            resume=resume,
        )
        try:
            save_stage4_results(
                final_candidates,
                final_class_df,
                out_root,
                trigger_datetime=trigger_datetime,
                trigger_label=str(config.get("trigger_label", "Trigger")),
            )
        except Exception:
            logging.exception("Failed to save stage4 results")
        logging.info("Stage4 complete: %s final candidates", len(final_candidates))
        return 0

    final_class_df = pd.read_csv(final_class_csv, low_memory=False)

    diff_index = stage0_diff_index(diff_root, out_root, limit_images=int(config.get("limit_images", 0)), object_ids=config.get("object_ids", []), resume=resume, full_frames=bool(config.get("blind_search", False)))
    if diff_index.empty:
        logging.warning("No difference images found")
        return 0

    stage1_images, stage1_objects = run_stage1(diff_index, out_root, config, final_class_df, resume)
    stage1_pass = stage1_images[pd.to_numeric(stage1_images.get("snr_empirical"), errors="coerce") >= float(config.get("snr_threshold", 3.0))].copy()
    if stage1_pass.empty:
        logging.warning("No candidates passed the SNR threshold")
        return 0

    manifest_df, complete_df = stage2_triplets(
        stage1_pass,
        out_root,
        sci_root=sci_root,
        ref_root=ref_root,
        final_class_df=final_class_df,
        download_missing=bool(config.get("download_missing", True)),
        triplet_size=int(config.get("triplet_size", 63)),
        ref_flip_lr=bool(config.get("ref_flip_lr", False)),
        max_workers=int(config.get("max_workers", 4)),
        chunk_size=int(config.get("chunk_size", 200)),
        full_frames=bool(config.get("blind_search", False)),
        resume=resume,
    )
    if complete_df.empty:
        logging.warning("No complete triplets were built")
        return 0

    scores_df, braai_pass_df = stage2_braai(
        manifest_df,
        complete_df,
        out_root,
        model_path=model_path,
        triplet_size=int(config.get("triplet_size", 63)),
        ref_flip_lr=bool(config.get("ref_flip_lr", False)),
        braai_threshold=float(config.get("braai_threshold", 0.25)),
        batch_size=int(config.get("chunk_size", 128)),
        resume=resume,
    )

    stage3_images, stage3_objects_all, stage3_objects_pass = stage3_pretrigger_magnitude(
        stage2_objects=braai_pass_df,
        diff_index=diff_index,
        diff_root=diff_root,
        final_class_path=final_class_csv,
        out_dir=out_root / "stage3",
        trigger_date=trigger_date_text,
        lookback_days=int(config.get("lookback_days", 10)),
        posttrigger_days=int(config.get("posttrigger_days", 10)),
        sigma=float(config.get("sigma", 3.0)),
        maxiters=int(config.get("maxiters", 5)),
        min_valid_pixel=float(config.get("min_valid_pixel", -5000.0)),
        download_missing=bool(config.get("download_missing", True)),
        full_frames=bool(config.get("blind_search", False)),
        resume=resume,
    )

    stage4_objects, final_candidates = stage4_host_mag(
        stage3_objects_pass,
        final_class_df,
        out_root,
        host_rmag_threshold=float(config.get("host_rmag_threshold", 22.5)),
        fwhm_reduced_chi2_max=float(config.get("fwhm_reduced_chi2_max", 5.0)),
        fwhm_ratio_min=float(config.get("fwhm_ratio_min", 0.5)),
        fwhm_ratio_max=float(config.get("fwhm_ratio_max", 1.8)),
        fwhm_sigma_multiplier=float(config.get("fwhm_sigma_multiplier", 3.0)),
        psf_spike_delta_chi2_min=float(config.get("psf_spike_delta_chi2_min", 9.0)),
        psf_amplitude_snr_min=float(config.get("psf_amplitude_snr_min", 3.0)),
        source_delta_chi2_min=float(config.get("source_delta_chi2_min", 9.0)),
        psf_residual_rms_sigma_max=float(config.get("psf_residual_rms_sigma_max", 1.5)),
        candidate_snr_min=float(config.get("candidate_snr_min", 3.0)),
        candidate_snr_max=float(config.get("candidate_snr_max", 5.0)),
        psf_visibility_gate_enabled=bool(config.get("psf_visibility_gate_enabled", False)),
        psf_visibility_score_min=float(config.get("psf_visibility_score_min", 2.362838023766029)),
        resume=resume,
    )

    # Save per-candidate results (triplets, profiles, light curves)
    try:
        save_stage4_results(
            final_candidates,
            final_class_df,
            out_root,
            trigger_datetime=trigger_datetime,
            trigger_label=str(config.get("trigger_label", "Trigger")),
        )
    except Exception:
        logging.exception("Failed to save stage4 results")

    try:
        minor_planet_candidates, minor_planet_matches = stage5_minor_planet_check(
            final_candidates,
            out_root,
            radius_arcmin=float(config.get("minor_planet_radius_arcmin", 5.0)),
            limiting_magnitude=float(config.get("minor_planet_limit_mag", 24.0)),
            observatory_code=str(config.get("minor_planet_observatory_code", "644")),
            resume=resume,
        )
        logging.info(
            "Stage5 minor-planet check complete: %s candidates checked, %s matches",
            len(minor_planet_candidates),
            len(minor_planet_matches),
        )
    except Exception:
        logging.exception("Failed to run stage5 minor-planet check")

    logging.info("Pipeline complete: %s final candidates", len(final_candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())