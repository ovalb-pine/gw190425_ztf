"""End-to-end April candidate pipeline for ZTF cutouts."""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yaml
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("PyYAML is required to read config.yaml") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))

from braai_batch import build_triplet_batch, load_braai_model, predict_braai_batch
from ztf_downloads.fetch_sci_ref_for_snr import _build_exact_coord_index, choose_ref_row, compute_host_offset
from ztf_downloads.snr_photometry import aperture_from_min_apcor, first_2d_data_and_header, parse_image_name, run_aperture_photometry_photutils
from ztf_downloads.snr_photometry_quadratic import process_difference_image, process_difference_images
from ztf_downloads.ztf_download import build_cutout_url, build_sci_url, build_ref_url, download_file

def _lookup_object_coords(final_class_df, object_id):
    object_id = normalize_object_id(object_id)
    rows = final_class_df[final_class_df["objID_SDSS-DR16"].map(normalize_object_id) == object_id]
    if not rows.empty:
        row = rows.iloc[0]
        ra = _safe_float(row.get("ra_fin", row.get("ra_SDSS-DR16", np.nan)))
        dec = _safe_float(row.get("dec_fin", row.get("dec_SDSS-DR16", np.nan)))
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    return np.nan, np.nan

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


def _norm_key_name(name: str) -> str:
    if "__ztf_" in name:
        return name.split("__", 1)[1]
    return name


def parse_diff_name(name: str):
    pattern = re.compile(
        r"(?P<object_id>\d+)_RA(?P<ra>[-+\d\.]+)_DEC(?P<dec>[-+\d\.]+)_"
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
    return pd.DataFrame(rows)


def _load_local_image(path: Path) -> np.ndarray:
    from astropy.io import fits

    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if isinstance(data, np.ndarray) and data.ndim == 2:
                return np.asarray(data, dtype=float)
    raise ValueError(f"No 2D image plane found in {path}")


def _normalize_image(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    arr = arr.copy()
    arr[~finite] = np.nanmedian(arr[finite])
    lo, hi = np.nanpercentile(arr, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        centered = arr - np.nanmedian(arr)
        scale = np.nanstd(centered)
        return centered / scale if np.isfinite(scale) and scale > 0 else centered
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo)


def _extract_zero_point(header: dict) -> float:
    for key in ("MAGZP", "MAGZERO", "ZP", "ZEROPT", "ZEROPNT"):
        try:
            value = float(header.get(key))
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    return np.nan


def _measure_diff_magnitude(image_path: Path, *, sigma: float, maxiters: int, min_valid_pixel: float) -> dict:
    from astropy.stats import sigma_clipped_stats

    data, header = first_2d_data_and_header(image_path)
    info = parse_image_name(Path(image_path).name)
    ny, nx = data.shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    try:
        ap_r, apcor_key, apcor_val, ap_diam_px = aperture_from_min_apcor(header)
    except Exception:
        ap_r = 2.0
        apcor_key = ""
        apcor_val = np.nan
        ap_diam_px = np.nan

    phot = run_aperture_photometry_photutils(data, cx, cy, ap_r)
    zero_point = _extract_zero_point(header)
    net_flux = float(phot.get("net_flux", np.nan))
    flux_err = float(phot.get("flux_err", np.nan))

    if np.isfinite(zero_point) and np.isfinite(net_flux) and net_flux > 0:
        mag = float(zero_point - 2.5 * np.log10(net_flux))
        mag_err = float(1.0857362047581294 * flux_err / net_flux) if np.isfinite(flux_err) and flux_err > 0 else np.nan
        upper_limit = np.nan
        detection = True
    elif np.isfinite(zero_point) and np.isfinite(flux_err) and flux_err > 0:
        upper_flux = 3.0 * flux_err
        mag = np.nan
        mag_err = np.nan
        upper_limit = float(zero_point - 2.5 * np.log10(max(upper_flux, 1e-12)))
        detection = False
    else:
        mag = np.nan
        mag_err = np.nan
        upper_limit = np.nan
        detection = False

    return {
        "image_path": str(image_path),
        "object_id": info["object_id"] if info else "",
        "ra": info["ra"] if info else np.nan,
        "dec": info["dec"] if info else np.nan,
        "mag": mag,
        "mag_err": mag_err,
        "upper_limit": upper_limit,
        "detection": detection,
        "zero_point": zero_point,
        "aperture_radius_px": ap_r,
        "apcor_min_key": apcor_key,
        "apcor_min_value": apcor_val,
        "apcor_min_diameter_px": ap_diam_px,
        **phot,
    }


def _center_crop(data: np.ndarray, size: int) -> np.ndarray:
    size = int(size)
    size = size if size % 2 == 1 else size + 1
    size = max(size, 5)
    half = size // 2
    cy = data.shape[0] // 2
    cx = data.shape[1] // 2
    y1 = max(0, cy - half)
    y2 = min(data.shape[0], cy + half + 1)
    x1 = max(0, cx - half)
    x2 = min(data.shape[1], cx + half + 1)
    crop = np.asarray(data[y1:y2, x1:x2], dtype=float)
    if crop.shape != (size, size):
        pad_y = size - crop.shape[0]
        pad_x = size - crop.shape[1]
        crop = np.pad(crop, ((0, pad_y), (0, pad_x)), mode="edge")
    return crop


from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


def _load_image_and_wcs(path):
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float32), WCS(hdu.header)
    raise ValueError(f"No 2D image found in {path}")


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


def build_triplet(
    diff_path,
    sci_path,
    ref_path,
    *,
    center_x,
    center_y,
    size=63,
    ref_flip_lr=False,
):
    diff, _ = _load_image_and_wcs(diff_path)
    sci, _ = _load_image_and_wcs(sci_path)
    ref, _ = _load_image_and_wcs(ref_path)

    # If center coordinates are not finite (missing), fall back to image center
    ny, nx = diff.shape
    if center_x is None or not np.isfinite(center_x):
        center_x_use = (nx - 1) / 2.0
    else:
        center_x_use = float(center_x)
    if center_y is None or not np.isfinite(center_y):
        center_y_use = (ny - 1) / 2.0
    else:
        center_y_use = float(center_y)

    diff = _crop(diff, center_x_use, center_y_use, size)
    sci = _crop(sci, center_x_use, center_y_use, size)
    ref = _crop(ref, center_x_use, center_y_use, size)

    return np.stack([sci, ref, diff], axis=-1)


def stage0_diff_index(diff_root: Path, out_root: Path, limit_images: int = 0, object_ids: Iterable[str] | None = None, *, resume: bool = True) -> pd.DataFrame:
    out_path = ensure_dir(out_root / "stage0") / "april_stage0_diff_index.csv"
    if resume:
        cached = load_csv_if_exists(out_path)
        if cached is not None:
            return cached

    df = collect_diff_index(diff_root)
    if object_ids:
        wanted = {str(_extract_object_id(v)) for v in object_ids if str(v).strip()}
        df = df[df["object_id"].astype(str).isin(wanted)].copy()
    if limit_images and limit_images > 0:
        df = df.head(int(limit_images)).copy()
    write_csv(df, out_path)
    return df


def stage1_quadratic_snr(
    diff_index: pd.DataFrame,
    out_root: Path,
    *,
    snr_threshold: float,
    sigma: float,
    maxiters: int,
    min_valid_pixel: float,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage1_dir = ensure_dir(out_root / "stage1")
    images_path = stage1_dir / "april_stage1_quadratic_snr_images.csv"
    objects_path = stage1_dir / "april_stage1_quadratic_snr_objects.csv"

    if resume:
        cached_images = load_csv_if_exists(images_path)
        cached_objects = load_csv_if_exists(objects_path)
        if cached_images is not None and cached_objects is not None:
            return cached_images, cached_objects

    if diff_index.empty:
        empty = diff_index.copy()
        write_csv(empty, images_path)
        write_csv(empty, objects_path)
        return empty, empty

    rows = process_difference_images(diff_index["image_path"].tolist(), sigma=sigma, maxiters=maxiters, min_valid_pixel=min_valid_pixel)
    if "image_path" not in rows.columns:
        rows["image_path"] = diff_index["image_path"].values[: len(rows)]
    merged = diff_index.merge(rows, on="image_path", how="left", suffixes=("", "_phot"))
    merged["snr"] = pd.to_numeric(merged.get("snr"), errors="coerce")
    pass_df = merged[merged["snr"] >= float(snr_threshold)].copy()

    summary = (
        pass_df.groupby("object_id", dropna=False)
        .agg(best_snr=("snr", "max"), n_pass_images=("image_path", "count"), first_obs_date=("filefracday", "min"), last_obs_date=("filefracday", "max"))
        .reset_index()
    )

    write_csv(merged, images_path)
    write_csv(summary, objects_path)
    return merged, summary


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
    ref_meta: pd.DataFrame,
    final_class_df: pd.DataFrame,
    max_ref_sep_arcsec: float,
    download_missing: bool,
    triplet_size: int,
    ref_flip_lr: bool,
    max_workers: int,
    chunk_size: int = 200,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sci_index = build_file_index(sci_root)
    ref_index = build_file_index(ref_root)
    coord_index = _build_exact_coord_index(final_class_df)

    # Map object ids to final-class row numbers for file/dir naming
    final_rownum_map = _build_final_rownum_map(final_class_df)

    stage2_dir = ensure_dir(out_root / "stage2")
    triplet_dir = ensure_dir(stage2_dir / "triplets")
    manifest_path = triplet_dir / "april_stage2_triplets_manifest.csv"
    objects_path = triplet_dir / "april_stage2_triplets_objects.csv"

    manifest_rows: list[dict] = []
    complete_rows: list[dict] = []

    cached_manifest = None
    cached_complete = None
    if resume:
        cached_manifest = load_csv_if_exists(manifest_path)
        cached_complete = load_csv_if_exists(objects_path)
        if cached_manifest is not None:
            manifest_rows.extend(cached_manifest.to_dict("records"))
        if cached_complete is not None:
            complete_rows.extend(cached_complete.to_dict("records"))

    completed_ids = set()
    if cached_complete is not None and not cached_complete.empty and "image_path" in cached_complete.columns:
        completed_ids = {str(value) for value in cached_complete["image_path"].astype(str).tolist() if str(value)}
    elif cached_manifest is not None and not cached_manifest.empty and "triplet_ready" in cached_manifest.columns:
        ready_rows = cached_manifest[cached_manifest["triplet_ready"].fillna(False).astype(bool)]
        if "image_path" in ready_rows.columns:
            completed_ids = {str(value) for value in ready_rows["image_path"].astype(str).tolist() if str(value)}

    if completed_ids:
        snr_images = snr_images[~snr_images["image_path"].astype(str).isin(completed_ids)].copy()

    def build_one(row: pd.Series) -> dict:
        diff_path = Path(row["image_path"])
        parsed = parse_diff_name(diff_path.name)
        if parsed is None:
            return _make_triplet_manifest_row(row, diff_path, Path(""), Path(""), "bad_diff_name", np.nan)

        object_id = _extract_object_id(row.get("object_id", parsed["object_id"]))
        coords_from_id = coord_index.get(object_id)
        if coords_from_id is not None:
            ra, dec = coords_from_id
        else:
            ra = _safe_float(row.get("ra", parsed["ra"]))
            dec = _safe_float(row.get("dec", parsed["dec"]))

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
        diff_dst.parent.mkdir(parents=True, exist_ok=True)
        if diff_path.exists() and not diff_dst.exists():
            diff_dst.write_bytes(diff_path.read_bytes())

        sci_dst = group_dir / "sci" / sci_name
        sci_dst.parent.mkdir(parents=True, exist_ok=True)
        sci_status = "missing"
        if sci_src is not None and sci_src.exists():
            if not sci_dst.exists():
                sci_dst.write_bytes(sci_src.read_bytes())
            sci_status = "local"
        elif download_missing:
            try:
                downloaded = download_file(build_cutout_url(sci_url, ra, dec, size_arcsec=240), out_dir=sci_dst.parent)
                sci_dst = Path(downloaded)
                sci_status = "downloaded"
            except Exception:
                sci_status = "missing"

        row_for_ref = {"filter": parsed["filter"], "field": parsed["field"], "ccdid": parsed["ccdid"], "qid": parsed["qid"], "ra": ra, "dec": dec}
        ref_choice = choose_ref_row(ref_meta, row_for_ref, max_ref_sep_arcsec)
        ref_dst = group_dir / "ref" / ""
        ref_status = "no_ref_match"
        ref_sep = np.nan
        if ref_choice is not None:
            ref_sep = float(ref_choice["match_sep_arcsec"])
            ref_row = {
                "filefracday": str(ref_choice["filefracday"]),
                "field": _safe_int(ref_choice["field"]),
                "filtercode": str(ref_choice["filtercode"]),
                "ccdid": _safe_int(ref_choice["ccdid"]),
                "qid": _safe_int(ref_choice["qid"]),
                "imgtypecode": str(ref_choice.get("imgtypecode", "o")),
            }
            ref_url = build_ref_url(ref_row)
            ref_name = Path(ref_url).name
            ref_src = ref_index.get(ref_name)
            ref_dst = group_dir / "ref" / ref_name
            ref_dst.parent.mkdir(parents=True, exist_ok=True)
            if ref_src is not None and ref_src.exists():
                if not ref_dst.exists():
                    ref_dst.write_bytes(ref_src.read_bytes())
                ref_status = "local"
            elif download_missing:
                try:
                    downloaded = download_file(build_cutout_url(ref_url, ra, dec, size_arcsec=240), out_dir=ref_dst.parent)
                    ref_dst = Path(downloaded)
                    ref_status = "downloaded"
                except Exception:
                    ref_status = "missing"

        status = "ok" if sci_status in {"local", "downloaded"} and ref_status in {"local", "downloaded"} else "partial"

        triplet = None
        if status == "ok" and sci_dst.exists() and ref_dst.exists() and diff_dst.exists():
            try:
                triplet = build_triplet(
                    diff_dst,
                    sci_dst,
                    ref_dst,
                    center_x=float(row["source_x_px"]),
                    center_y=float(row["source_y_px"]),
                    size=triplet_size,
                    ref_flip_lr=ref_flip_lr,
                )
            except Exception:
                triplet = None

        manifest = _make_triplet_manifest_row(row, diff_dst, sci_dst, ref_dst, status, ref_sep, triplet)
        manifest.update({"ra_used": ra, "dec_used": dec, "coord_source": "final_class_by_id" if coords_from_id is not None else "snr_or_diff"})
        if triplet is not None:
            manifest["triplet_ready"] = True
            manifest["triplet_path"] = str(group_dir / "triplet.npy")
            np.save(group_dir / "triplet.npy", triplet)
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
        scores_df = scores_df.sort_values([col for col in ["braai_score", "snr"] if col in scores_df.columns], ascending=[False, False] if "braai_score" in scores_df.columns else True)
    if not objects_df.empty and "image_path" in objects_df.columns:
        objects_df = objects_df.sort_values([col for col in ["braai_score", "snr"] if col in objects_df.columns], ascending=[False, False] if "braai_score" in objects_df.columns else True)
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
    work["snr"] = pd.to_numeric(work.get("snr"), errors="coerce")
    sort_cols = [col for col in ["braai_score", "snr"] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    best = work.drop_duplicates(subset=["object_id"], keep="first").copy()
    keep_cols = [
        "object_id",
        "braai_score",
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

def stage3_pretrigger_magnitude(
    *,
    stage2_objects: pd.DataFrame,
    diff_index: pd.DataFrame,
    diff_root: Path,
    final_class_path: Path,
    out_dir: Path,
    trigger_date,
    lookback_days: int = 10,
    sigma: float = 3.0,
    maxiters: int = 5,
    min_valid_pixel: float = -5000.0,
    download_missing: bool = True,
    resume: bool = True,
):
    from ztf_downloads.ztf_search import metadata_search_resumable
    from ztf_downloads.ztf_download import batch_download_resumable
    from ztf_downloads.ztf_download import build_cutout_url, build_sci_url
    from urllib.parse import urlparse


    out_dir = ensure_dir(out_dir)
    images_path = out_dir / "april_stage3_pretrigger_mag_images.csv"
    objects_path = out_dir / "april_stage3_pretrigger_mag_objects.csv"

    if resume:
        cached_images = load_csv_if_exists(images_path)
        cached_objects = load_csv_if_exists(objects_path)
        if cached_images is not None and cached_objects is not None and not cached_objects.empty:
            if "pretrigger_mag_pass" in cached_objects.columns and cached_objects["pretrigger_mag_pass"].any():
                passed = cached_objects[cached_objects["pretrigger_mag_pass"]].copy()
                return cached_images, cached_objects, passed
            logging.info("Recomputing stage3 pre-trigger analysis because cached outputs contain no passing candidates")

    trigger = pd.to_datetime(trigger_date)
    window_start = trigger - pd.Timedelta(days=int(lookback_days))

    # Prepare diff_index with object_id and obs_date
    work = diff_index.copy()
    if "object_id" not in work.columns:
        work["object_id"] = work["image_path"].map(lambda p: parse_diff_name(Path(p).name)["object_id"] if parse_diff_name(Path(p).name) else "")
    work["object_id"] = work["object_id"].map(normalize_object_id)

    def _parse_obs_date(image_path):
        parsed = parse_diff_name(Path(image_path).name)
        if parsed is None:
            return pd.NaT
        filefracday = str(parsed.get("filefracday", ""))
        if len(filefracday) >= 14:
            ts = pd.to_datetime(filefracday[:14], errors="coerce", format="%Y%m%d%H%M%S")
            if pd.notna(ts):
                return ts
        if len(filefracday) >= 8:
            return pd.to_datetime(filefracday[:8], errors="coerce", format="%Y%m%d")
        return pd.NaT

    work["obs_date"] = work["image_path"].map(_parse_obs_date)

    # Normalize object IDs from stage2_objects and build coordinate lookup directly
    object_ids = []
    coord_lookup = {}
    # also collect source pixel centers from stage2_objects so we can carry them forward
    source_center_lookup = {}
    for _, row in stage2_objects.iterrows():
        obj_raw = row.get("object_id")
        if pd.isna(obj_raw):
            continue
        obj = normalize_object_id(obj_raw)
        if not obj:
            continue
        object_ids.append(obj)
        # Use ra_used/dec_used if available, else fallback to ra/dec
        ra = _safe_float(row.get("ra_used", row.get("ra", np.nan)))
        dec = _safe_float(row.get("dec_used", row.get("dec", np.nan)))
        if np.isfinite(ra) and np.isfinite(dec):
            coord_lookup[obj] = (ra, dec)
        else:
            # If missing, try from work (last resort)
            obj_rows = work[work["object_id"] == obj]
            if not obj_rows.empty:
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

    # Load final-class CSV to map object ids -> row numbers for naming
    try:
        final_class_df = pd.read_csv(final_class_path, low_memory=False)
    except Exception:
        final_class_df = None
    final_rownum_map = _build_final_rownum_map(final_class_df)

    # Identify objects that need pre-trigger downloads
    download_needed = {}
    if download_missing:
        for obj in object_ids:
            obj_work = work[work["object_id"] == obj]
            pre = obj_work[(obj_work["obs_date"] < trigger) & (obj_work["obs_date"] >= window_start)]
            if len(pre) == 0:
                download_needed[obj] = True

    if download_needed:
        logging.info(f"Downloading pre-trigger images for {len(download_needed)} objects")
        download_inputs = []
        for obj in download_needed:
            if obj not in coord_lookup:
                logging.warning(f"Could not find coordinates for object {obj}")
                continue
            ra, dec = coord_lookup[obj]

            # Search metadata – only r‑band
            # prefer row number for filenames if available
            obj_norm = normalize_object_id(obj)
            rownum = final_rownum_map.get(obj_norm, obj_norm)
            search_csv = out_dir / f"pretrigger_search_{rownum}.csv"
            search_progress = out_dir / f"pretrigger_search_{rownum}.progress.json"
            # Robust metadata fetch: try normal batch, then longer timeout/retries,
            # then fall back to per-day queries if IRSA is timing out or returning
            # transient errors.
            from ztf_downloads.ztf_search import metadata_search_batch

            def _fetch_meta():
                # primary attempt: short timeout
                try:
                    return metadata_search_resumable(
                        positions=[(ra, dec)],
                        output_csv=str(search_csv),
                        progress_json=str(search_progress),
                        batch_size=50,
                        size_deg=0.01,
                        product_type="sci",
                        filtercodes=["zr"],
                        date_start=window_start.strftime("%Y-%m-%d"),
                        date_end=trigger.strftime("%Y-%m-%d"),
                        timeout=120,
                    )
                except Exception as e1:
                    logging.warning(f"Metadata search initial attempt failed for {obj}: {e1}")
                # Second attempt: longer timeout and more retries
                try:
                    return metadata_search_resumable(
                        positions=[(ra, dec)],
                        output_csv=str(search_csv),
                        progress_json=str(search_progress),
                        batch_size=50,
                        size_deg=0.01,
                        product_type="sci",
                        filtercodes=["zr"],
                        date_start=window_start.strftime("%Y-%m-%d"),
                        date_end=trigger.strftime("%Y-%m-%d"),
                        timeout=300,
                        max_retries=8,
                    )
                except Exception as e2:
                    logging.warning(f"Metadata search extended attempt failed for {obj}: {e2}")

                # Final fallback: split date range into per-day POST queries to reduce server load
                try:
                    parts = []
                    # iterate day-by-day
                    day_start = pd.to_datetime(window_start).normalize()
                    day_end = pd.to_datetime(trigger).normalize()
                    cur = day_start
                    while cur < day_end:
                        ds = cur.strftime("%Y-%m-%d")
                        de = (cur + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        try:
                            part = metadata_search_batch(positions=[(ra, dec)], size_deg=0.01, product_type="sci", ct="csv", date_start=ds, date_end=de, filtercodes=["zr"], timeout=180)
                            if part is None:
                                part = pd.DataFrame()
                            parts.append(part)
                        except Exception as e3:
                            logging.warning(f"Per-day metadata search failed for {obj} {ds}->{de}: {e3}")
                        cur += pd.Timedelta(days=1)
                    if parts:
                        return pd.concat(parts, ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
                except Exception as e4:
                    logging.warning(f"Per-day fallback failed for {obj}: {e4}")

                # all attempts failed
                return pd.DataFrame()

            df_meta = _fetch_meta()
            if df_meta is None or df_meta.empty:
                logging.warning(f"Metadata search yielded no results for {obj}")
                continue

            if df_meta is None or df_meta.empty:
                continue

            for _, mrow in df_meta.iterrows():
                field = mrow.get("field", 0)
                ccdid = mrow.get("ccdid", 0)
                qid = mrow.get("qid", 0)
                filtercode = mrow.get("filtercode", "")
                filefracday = mrow.get("filefracday", "")
                obj_norm = normalize_object_id(obj)

                # 1. Get the correct science image URL using the existing function
                base_sci_url = build_sci_url(mrow, suffix="scimrefdiffimg.fits.fz")
                
                # 2. Extract the original filename from that URL
                original_fname = Path(urlparse(base_sci_url).path).name
                
                # 3. Create cutout URL (adds center, size, gzip)
                url = build_cutout_url(base_sci_url, float(mrow["in_ra"]), float(mrow["in_dec"]), size_arcsec=240)
                
                # 4. Get the full 14‑digit filefracday as string (ensure no truncation)
                filefracday_str = str(int(mrow["filefracday"])).zfill(14)
                
                # 5. Your custom label
                custom_label = f"RA{mrow['in_ra']:.4f}_DEC{mrow['in_dec']:.4f}_{mrow['filtercode']}_{filefracday_str}"

                row_label = final_rownum_map.get(obj_norm, obj_norm)
                
                # 6. Desired final filename: gal_folder + custom_label + "__" + original_fname
                desired_fname = f"{str(row_label)}_{custom_label}__{original_fname}"
                
                download_inputs.append((url, desired_fname, str(row_label)))

                # file_label = (
                #     f"{row_label}_RA{ra:.4f}_DEC{dec:.4f}_{filtercode}_{filefracday}__ztf_{int(field):06d}_{filtercode}_c{int(ccdid)}_o_q{int(qid)}_scimrefdiffimg"
                # )
                # url = build_cutout_url(
                #     build_sci_url(mrow, suffix="scimrefdiffimg.fits.fz"),
                #     float(mrow['in_ra']),
                #     float(mrow['in_dec']),
                #     size_arcsec=240,
                # )
                # download_inputs.append((url, file_label, str(row_label)))

        if download_inputs:
            logging.info(f"Downloading {len(download_inputs)} pre-trigger cutouts")
            batch_download_resumable(
                download_inputs,
                out_dir=str(diff_root),
                progress_json=str(out_dir / "pretrigger_download.progress.json"),
                max_workers=6,
                chunk_size=100,
            )
            # Re-index diff_root to include newly downloaded files
            diff_index = collect_diff_index(diff_root)
            work = diff_index.copy()
            # IMPORTANT: normalize object_id to avoid scientific notation
            work["object_id"] = work["object_id"].map(normalize_object_id)
            work["obs_date"] = work["image_path"].map(_parse_obs_date)


    # Now separate pre and post trigger
    pretrigger = work[(work["obs_date"] < trigger) & (work["obs_date"] >= window_start)].copy()
    posttrigger = work[work["obs_date"] >= trigger].copy()

    image_rows = []
    object_rows = []
    for object_id in object_ids:
        group = pretrigger[pretrigger["object_id"] == object_id].copy()
        object_post = posttrigger[posttrigger["object_id"] == object_id].copy()

        # Measure post-trigger magnitudes
        post_mags = []
        post_mag_errs = []
        for _, row in object_post.iterrows():
            path = Path(row["image_path"])
            if not path.exists():
                continue
            try:
                record = _measure_diff_magnitude(path, sigma=sigma, maxiters=maxiters, min_valid_pixel=min_valid_pixel)
            except Exception as exc:
                image_rows.append({"object_id": object_id, "image_path": str(path), "status": f"error: {exc}"})
                continue
            sx, sy = source_center_lookup.get(object_id, (np.nan, np.nan))
            record.update({"object_id": object_id, "phase": "posttrigger", "source_x_px": sx, "source_y_px": sy})
            image_rows.append(record)
            if np.isfinite(record.get("mag", np.nan)):
                post_mags.append(float(record["mag"]))
                if np.isfinite(record.get("mag_err", np.nan)):
                    post_mag_errs.append(float(record["mag_err"]))

        # Measure pre-trigger magnitudes
        pre_records = []
        for _, row in group.iterrows():
            path = Path(row["image_path"])
            if not path.exists():
                continue
            try:
                record = _measure_diff_magnitude(path, sigma=sigma, maxiters=maxiters, min_valid_pixel=min_valid_pixel)
            except Exception as exc:
                image_rows.append({"object_id": object_id, "image_path": str(path), "status": f"error: {exc}"})
                continue
            sx, sy = source_center_lookup.get(object_id, (np.nan, np.nan))
            record.update({"object_id": object_id, "phase": "pretrigger", "source_x_px": sx, "source_y_px": sy})
            image_rows.append(record)
            pre_records.append(record)

        pre_mags = [float(item["mag"]) for item in pre_records if np.isfinite(item.get("mag", np.nan))]
        pre_upper_limits = [float(item["upper_limit"]) for item in pre_records if np.isfinite(item.get("upper_limit", np.nan))]

        post_ref_mag = float(np.nanmedian(post_mags)) if post_mags else np.nan
        post_ref_err = float(np.nanmedian(post_mag_errs)) if post_mag_errs else np.nan

        # Determine consistency and pass
        n_pre = len(group)
        if n_pre == 0:
            consistent = False
            pass_flag = True
            reason = "no_pretrigger_images"
        elif pre_mags:
            pre_ref = float(np.nanmedian(pre_mags))
            if np.isfinite(post_ref_mag):
                combined_err = np.sqrt(max(post_ref_err, 0.0) ** 2 + np.nanstd(pre_mags) ** 2)
                if not np.isfinite(combined_err) or combined_err <= 0:
                    combined_err = 0.5
                brightening = pre_ref - post_ref_mag
                consistent = brightening <= max(0.5, 2.0 * combined_err)
            else:
                consistent = False
            pass_flag = not consistent
            reason = "consistent" if consistent else "pretrigger_detection"
        else:
            if post_mags and pre_upper_limits:
                tightest_ul = float(np.nanmax(pre_upper_limits))
                pass_flag = bool(np.isfinite(post_ref_mag) and tightest_ul >= post_ref_mag)
                reason = "upper_limits_allow_transient" if pass_flag else "upper_limit_too_bright"
                consistent = pass_flag
            else:
                consistent = False
                pass_flag = False
                reason = "no_upper_limits"

        sx, sy = source_center_lookup.get(object_id, (np.nan, np.nan))
        # detection RA/DEC: prefer coord_lookup, else use median from work
        dra, ddec = (np.nan, np.nan)
        if object_id in coord_lookup:
            dra, ddec = coord_lookup[object_id]
        else:
            obj_rows = work[work["object_id"] == object_id]
            if not obj_rows.empty:
                dra = np.nanmedian(obj_rows["ra"]) if "ra" in obj_rows.columns else np.nan
                ddec = np.nanmedian(obj_rows["dec"]) if "dec" in obj_rows.columns else np.nan
        object_rows.append({
            "object_id": object_id,
            "n_pretrigger_images": n_pre,
            "pretrigger_mag_ref": pre_ref if pre_mags else np.nan,
            "pretrigger_mag_ref_err": np.nan,
            "pretrigger_mag_max_residual": np.nan,
            "pretrigger_mag_upper_limit": float(np.nanmax(pre_upper_limits)) if pre_upper_limits else np.nan,
            "pretrigger_mag_consistent": consistent,
            "pretrigger_mag_pass": pass_flag,
            "pretrigger_mag_reason": reason,
            "posttrigger_reference_kind": "posttrigger_median",
            "posttrigger_reference_mag": post_ref_mag,
            "posttrigger_reference_mag_err": post_ref_err,
            "n_posttrigger_detections": int(len(post_mags)),
            "source_x_px": sx,
            "source_y_px": sy,
            "detection_ra": dra,
            "detection_dec": ddec,
        })

    mag_df = pd.DataFrame(image_rows)
    stage3_objects_all = pd.DataFrame(object_rows)
    stage3_objects_pass = stage3_objects_all[stage3_objects_all["pretrigger_mag_pass"]].copy() if "pretrigger_mag_pass" in stage3_objects_all.columns else stage3_objects_all.iloc[0:0].copy()

    write_csv(mag_df, images_path)
    write_csv(stage3_objects_pass, objects_path)
    return mag_df, stage3_objects_all, stage3_objects_pass

def stage4_host_mag(stage3_objects: pd.DataFrame, final_class_df: pd.DataFrame, out_root: Path, *, host_rmag_threshold: float, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage4_dir = ensure_dir(out_root / "stage4")
    host_objects_path = stage4_dir / "april_stage4_hostmag_objects.csv"
    final_candidates_path = stage4_dir / "april_stage4_final_candidates.csv"
    cached_host = load_csv_if_exists(host_objects_path)
    cached_final = load_csv_if_exists(final_candidates_path)
    if cached_host is not None and cached_final is not None and not cached_host.empty:
        if resume:
            cached_ids = cached_host["object_id"].astype(str).map(normalize_object_id).tolist()
            current_ids = stage3_objects["object_id"].astype(str).map(normalize_object_id).tolist()
            if cached_ids == current_ids:
                return cached_host, cached_final
            logging.info(
                "Recomputing stage4 outputs because stage3 object IDs changed: %s -> %s",
                len(cached_ids),
                len(current_ids),
            )
        else:
            logging.info("Recomputing stage4 outputs because resume=False")

    final = stage3_objects.copy()
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
        write_csv(final, host_objects_path)
        write_csv(final, final_candidates_path)
        return final, final

    final_id_norm = final["object_id"].astype(str).map(normalize_object_id)

    host_col = None
    for candidate in ("rmag_SDSS-DR16", "host_rmag", "rmag"):
        if candidate in final_class_df.columns:
            host_col = candidate
            break
    if host_col is None:
        final["host_rmag"] = np.nan
        final["host_rmag_source"] = "missing"
    else:
        host_index = final_class_df.copy()
        id_col = "object_id"
        host_index[id_col] = host_index[id_col].astype(str).map(normalize_object_id)
        host_index[host_col] = pd.to_numeric(host_index[host_col], errors="coerce")
        lookup = host_index[[id_col, host_col]].drop_duplicates(subset=[id_col]).set_index(id_col)[host_col]
        final["host_rmag"] = final_id_norm.map(lookup)
        final["host_rmag_source"] = np.where(final["host_rmag"].notna(), host_col, "missing")

        lookup = host_index[[id_col, "D_L_fin"]].drop_duplicates(subset=[id_col]).set_index(id_col)["D_L_fin"]
        final["D_L_fin"] = final_id_norm.map(lookup)

        lookup = host_index[[id_col, "e_D_L_fin"]].drop_duplicates(subset=[id_col]).set_index(id_col)["e_D_L_fin"]
        final["e_D_L_fin"] = final_id_norm.map(lookup)

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
    lookup_df["best_snr"] = pd.to_numeric(lookup_df.get("best_snr"), errors="coerce")
    snr_lookup = lookup_df.groupby("object_id", sort=False)["best_snr"].max()

    # 3. Map the per-object maximum SNR into the final DataFrame
    final["best_snr"] = pd.to_numeric(final_id_norm.map(snr_lookup), errors="coerce")


    detection_ra_candidates = ["detection_ra_deg", "detection_ra", "catalog_ra_deg", "ra_used", "ra"]
    detection_dec_candidates = ["detection_dec_deg", "detection_dec", "catalog_dec_deg", "dec_used", "dec"]
    host_ra_candidates = ["host_ra", "catalog_ra_deg"]
    host_dec_candidates = ["host_dec", "catalog_dec_deg"]

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
    for candidate in host_ra_candidates:
        if candidate in final.columns:
            host_ra = pd.to_numeric(final[candidate], errors="coerce")
            break
    for candidate in host_dec_candidates:
        if candidate in final.columns:
            host_dec = pd.to_numeric(final[candidate], errors="coerce")
            break

    host_offsets = []
    for idx in final.index:
        dra = detection_ra.loc[idx] if detection_ra is not None else np.nan
        ddec = detection_dec.loc[idx] if detection_dec is not None else np.nan
        hra = host_ra.loc[idx] if host_ra is not None else np.nan
        hdec = host_dec.loc[idx] if host_dec is not None else np.nan
        host_offsets.append(compute_host_offset(dra, ddec, hra, hdec))
    final["host_offset_arcsec"] = host_offsets

    passed = final[final["host_rmag_pass"]].copy()
    write_csv(final, host_objects_path)
    write_csv(passed, final_candidates_path)
    return final, passed


def stage4_host_rmag(*, stage3_objects: pd.DataFrame, final_class_path: Path, out_dir: Path, host_mag_threshold: float = 22.5, resume: bool = True):
    final_class_df = pd.read_csv(final_class_path, low_memory=False)
    enriched_df, final_df = stage4_host_mag(stage3_objects, final_class_df, out_dir, host_rmag_threshold=host_mag_threshold, resume=resume)
    return enriched_df, final_df


def save_stage4_results(final_candidates: pd.DataFrame, final_class_df: pd.DataFrame, out_root: Path, *, max_triplets_per_object: int = 10):
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
        candidate_dir = triplet_root / str(label)
        if candidate_dir.exists():
            triplet_files = list(candidate_dir.rglob("triplet.npy"))
        if not triplet_files:
            # fallback: search anywhere for directory matching oid
            triplet_files = list(triplet_root.rglob(f"*{oid}*/triplet.npy"))

        for i, tpath in enumerate(triplet_files[:max_triplets_per_object]):
            try:
                trip = _np.load(tpath)
            except Exception:
                continue

            # normalize each plane (median/std)
            planes = []
            for k in range(trip.shape[-1]):
                p = _np.asarray(trip[..., k], dtype=float)
                m = _np.nanmedian(p)
                s = _np.nanstd(p)
                if not _np.isfinite(s) or s == 0:
                    s = 1.0
                planes.append((p - m) / s)

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
                vmax = 5
                titles = ("sci", "ref", "diff")
                for col in range(3):
                    ax = axes[0, col]
                    im = ax.imshow(display_planes[col], origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax)
                    ax.set_title(titles[col])
                    ax.axis("off")
                    ax2 = axes[1, col]
                    ax2.imshow(smoothed[col], origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax)
                    ax2.set_title(f"{titles[col]} (smoothed)")
                    ax2.axis("off")

                fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05)
                fig.savefig(obj_dir / f"triplet_norm_{i}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                logging.debug("failed to save triplet image for %s: %s", oid, tpath)

            # 1D radial profile of diff plane
            try:
                diff = planes[-1]
                cy, cx = _np.array(diff.shape) // 2
                yy, xx = _np.indices(diff.shape)
                r = _np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                r_int = r.astype(int).ravel()
                vals = diff.ravel()
                tbin = _np.bincount(r_int, weights=vals)
                cnt = _np.bincount(r_int)
                radial = tbin / _np.maximum(cnt, 1)
                fig = plt.figure(figsize=(4, 3))
                plt.plot(radial, marker=".")
                plt.xlabel("radius (px)")
                plt.ylabel("mean normalized diff")
                plt.savefig(obj_dir / f"radial_profile_{i}.png", bbox_inches="tight")
                plt.close(fig)
            except Exception:
                logging.debug("failed radial profile for %s", oid)

            # 3D surface plot of diff
            try:
                fig = plt.figure(figsize=(5, 4))
                ax = fig.add_subplot(111, projection="3d")
                xs = _np.arange(diff.shape[1])
                ys = _np.arange(diff.shape[0])
                X, Y = _np.meshgrid(xs, ys)
                ax.plot_surface(X, Y, diff, cmap="viridis", linewidth=0, antialiased=False)
                ax.set_zlim(_np.nanpercentile(diff, 5), _np.nanpercentile(diff, 95))
                plt.savefig(obj_dir / f"surface_{i}.png", bbox_inches="tight")
                plt.close(fig)
            except Exception:
                logging.debug("failed 3d surface for %s", oid)

            # light curve: magnitude vs time from stage3 pretrigger mag images (preferred)
            try:
                stage3_images_path = out_root / "stage3" / "april_stage3_pretrigger_mag_images.csv"
                s3 = None
                if stage3_images_path.exists():
                    try:
                        s3 = _pd.read_csv(stage3_images_path, low_memory=False)
                    except Exception:
                        s3 = None

                if s3 is not None:
                    ssel = s3[s3["object_id"].astype(str).map(normalize_object_id) == oid].copy()
                else:
                    ssel = _pd.DataFrame()

                if ssel.empty and stage1_images is not None:
                    # fallback to stage1 SNR-driven table (no mags available)
                    ssel = stage1_images[stage1_images["object_id"].astype(str).map(normalize_object_id) == oid].copy()

                if not ssel.empty:
                    # prefer magnitudes if available, otherwise use SNR as fallback
                    ssel["mag"] = _pd.to_numeric(ssel.get("mag"), errors="coerce")
                    ssel["mag_err"] = _pd.to_numeric(ssel.get("mag_err"), errors="coerce")
                    ssel["upper_limit"] = _pd.to_numeric(ssel.get("upper_limit"), errors="coerce")
                    ssel["snr"] = _pd.to_numeric(ssel.get("snr"), errors="coerce")

                    # Extract filefracday-derived fractional day for plotting (day-of-year + fraction)
                    def _extract_t_frac(pth):
                        try:
                            name = Path(str(pth)).name
                            parsed = parse_diff_name(name)
                            if parsed and parsed.get("filefracday"):
                                ffd = str(parsed["filefracday"]).strip()
                                if len(ffd) >= 14:
                                    dt = _pd.to_datetime(ffd[:14], format="%Y%m%d%H%M%S", errors="coerce")
                                    if pd.notna(dt):
                                        return float(dt.dayofyear) + (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
                                if len(ffd) >= 8:
                                    date = _pd.to_datetime(ffd[:8], format="%Y%m%d", errors="coerce")
                                    if pd.notna(date):
                                        frac = 0.0
                                        if len(ffd) > 8:
                                            try:
                                                frac = float(f"0.{ffd[8:]}")
                                            except Exception:
                                                frac = 0.0
                                        return float(date.dayofyear) + frac
                        except Exception:
                            pass
                        return _np.nan

                    ssel["t_frac"] = ssel.get("image_path").apply(_extract_t_frac)

                    # Extract human date and filter label for xticks and color grouping
                    def _extract_date_filter_label(pth):
                        try:
                            name = Path(str(pth)).name
                            parsed = parse_diff_name(name)
                            if parsed and parsed.get("filefracday"):
                                ffd = str(parsed["filefracday"]).strip()
                                if len(ffd) >= 8:
                                    d = ffd[:8]
                                    return f"{d[0:4]}-{d[4:6]}-{d[6:8]} ({parsed.get('filter', '?')})"
                        except Exception:
                            pass
                        return Path(str(pth)).name

                    ssel["date_label"] = ssel.get("image_path").apply(_extract_date_filter_label)

                    # Prefer a precise datetime parsed from filefracday when available
                    def _extract_obs_datetime(pth):
                        try:
                            name = Path(str(pth)).name
                            parsed = parse_diff_name(name)
                            if parsed and parsed.get("filefracday"):
                                ffd = str(parsed["filefracday"]).strip()
                                if len(ffd) >= 14:
                                    dt = _pd.to_datetime(ffd[:14], format="%Y%m%d%H%M%S", errors="coerce")
                                    if pd.notna(dt):
                                        return dt
                                if len(ffd) >= 8:
                                    base = _pd.to_datetime(ffd[:8], format="%Y%m%d", errors="coerce")
                                    if pd.isna(base):
                                        return _pd.NaT
                                    frac = 0.0
                                    if len(ffd) > 8:
                                        try:
                                            frac = float(f"0.{ffd[8:]}")
                                        except Exception:
                                            frac = 0.0
                                    return base + _pd.to_timedelta(frac * 86400.0, unit="s")
                        except Exception:
                            pass
                        return _pd.NaT

                    ssel["obs_datetime"] = ssel.get("image_path").apply(_extract_obs_datetime)

                    # For any rows missing precise datetime, fall back to the date parsed from the label
                    mask_na = ssel["obs_datetime"].isna()
                    if mask_na.any():
                        ssel.loc[mask_na, "obs_datetime"] = ssel.loc[mask_na, "date_label"].apply(lambda dl: _pd.to_datetime(dl[:10], errors="coerce") if isinstance(dl, str) else _pd.NaT)

                    # Extract filter from filename (more robust than regex on date_label)
                    def _extract_filter(pth):
                        try:
                            parsed = parse_diff_name(Path(str(pth)).name)
                            if parsed:
                                return parsed.get("filter")
                        except Exception:
                            pass
                        return None

                    ssel["filt"] = ssel.get("image_path").apply(_extract_filter)

                    # plotting using real datetimes on X axis and coloring by filter
                    filter_colors = {"zg": "green", "zr": "red", "zi": "orange"}
                    ssel_plot = ssel[ssel["obs_datetime"].notna()].copy()
                    if not ssel_plot.empty:
                        import matplotlib.dates as mdates
                        from matplotlib.dates import DateFormatter

                        fig, ax = plt.subplots(figsize=(9, 4))
                        for filt, color in filter_colors.items():
                            mask_f = ssel_plot["filt"] == filt
                            if mask_f.any() and ssel_plot.loc[mask_f, "mag"].notna().any():
                                det = ssel_plot[mask_f & ssel_plot["mag"].notna()]
                                ax.errorbar(det["obs_datetime"].values, det["mag"].values, yerr=det.get("mag_err"), fmt="o", color=color, label=f"Det ({filt})")
                            if mask_f.any() and ssel_plot.loc[mask_f, "upper_limit"].notna().any():
                                ul = ssel_plot[mask_f & ssel_plot["upper_limit"].notna()]
                                ax.scatter(ul["obs_datetime"].values, ul["upper_limit"].values, marker="v", color=color, s=60, alpha=0.6, label=f"UL ({filt})")

                        # xticks: use a date locator/formatter so labels do not overlap
                        import matplotlib.dates as mdates
                        from matplotlib.dates import DateFormatter

                        locator = mdates.AutoDateLocator()
                        # choose formatter depending on whether times are non-midnight
                        unique_dt = sorted(ssel_plot["obs_datetime"].dropna().unique())
                        times_present = any((dt.time().hour != 0 or dt.time().minute != 0 or dt.time().second != 0) for dt in unique_dt)
                        fmt = DateFormatter('%Y-%m-%d %H:%M:%S') if times_present else DateFormatter('%Y-%m-%d')
                        ax.xaxis.set_major_locator(locator)
                        ax.xaxis.set_major_formatter(fmt)
                        fig.autofmt_xdate(rotation=45, ha='right')

                        ax.invert_yaxis()
                        ax.set_ylabel("mag")
                        ax.set_title(f"Light curve {oid}")
                        ax.legend()
                        plt.tight_layout()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(obj_dir / "lightcurve_mag.png", bbox_inches="tight")
                        plt.close(fig)
            except Exception:
                logging.debug("failed to save light curve for %s", oid)




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the April candidate pipeline")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config YAML")
    parser.add_argument("--resume", action="store_true", help="Reuse stage outputs if they already exist")
    parser.add_argument("--force", action="store_true", help="Ignore cached stage outputs and recompute")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    resume = bool(args.resume or config.get("resume", True)) and not bool(args.force)
    out_root = ensure_dir(resolve_cli_path(config.get("out_root", "april_candidate_pipeline_out")))
    log_path = out_root / "pipeline.log"
    logging.basicConfig(level=getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s", filename=log_path, filemode="w")

    logging.info("Starting pipeline")
    diff_root = resolve_cli_path(config.get("diff_root", "ztf_diff_cutout_imgs_catalog2"))
    sci_root = resolve_cli_path(config.get("sci_root", "ztf_sci_cutout_imgs"))
    ref_root = resolve_cli_path(config.get("ref_root", "ztf_ref_cutout_imgs"))
    ref_meta_path = resolve_cli_path(config.get("ref_meta", "metadata_ref_best.csv"))
    final_class_csv = resolve_cli_path(config.get("final_class_csv", "final_class_table_new_(only_in_dist_bounds).csv"))
    model_path = resolve_cli_path(config.get("model_path", "braai/models/braai_d6_m9.h5"))

    ref_meta = pd.read_csv(ref_meta_path, low_memory=False)
    final_class_df = pd.read_csv(final_class_csv, low_memory=False)

    diff_index = stage0_diff_index(diff_root, out_root, limit_images=int(config.get("limit_images", 0)), object_ids=config.get("object_ids", []), resume=resume)
    if diff_index.empty:
        logging.warning("No difference images found")
        return 0

    stage1_images, stage1_objects = stage1_quadratic_snr(
        diff_index,
        out_root,
        snr_threshold=float(config.get("snr_threshold", 3.0)),
        sigma=float(config.get("sigma", 3.0)),
        maxiters=int(config.get("maxiters", 5)),
        min_valid_pixel=float(config.get("min_valid_pixel", -5000.0)),
        resume=resume,
    )
    stage1_pass = stage1_images[pd.to_numeric(stage1_images.get("snr"), errors="coerce") >= float(config.get("snr_threshold", 3.0))].copy()
    if stage1_pass.empty:
        logging.warning("No candidates passed the SNR threshold")
        return 0

    manifest_df, complete_df = stage2_triplets(
        stage1_pass,
        out_root,
        sci_root=sci_root,
        ref_root=ref_root,
        ref_meta=ref_meta,
        final_class_df=final_class_df,
        max_ref_sep_arcsec=float(config.get("max_ref_sep_arcsec", 3.0)),
        download_missing=bool(config.get("download_missing", True)),
        triplet_size=int(config.get("triplet_size", 63)),
        ref_flip_lr=bool(config.get("ref_flip_lr", False)),
        max_workers=int(config.get("max_workers", 4)),
        chunk_size=int(config.get("chunk_size", 200)),
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
        trigger_date=str(config.get("trigger_date", "2019-04-25")),
        lookback_days=int(config.get("lookback_days", 10)),
        sigma=float(config.get("sigma", 3.0)),
        maxiters=int(config.get("maxiters", 5)),
        min_valid_pixel=float(config.get("min_valid_pixel", -5000.0)),
        download_missing=bool(config.get("download_missing", True)),
        resume=resume,
    )

    stage4_objects, final_candidates = stage4_host_mag(
        stage3_objects_pass,
        final_class_df,
        out_root,
        host_rmag_threshold=float(config.get("host_rmag_threshold", 22.5)),
        resume=resume,
    )

    # Save per-candidate results (triplets, profiles, light curves)
    try:
        save_stage4_results(final_candidates, final_class_df, out_root)
    except Exception:
        logging.exception("Failed to save stage4 results")

    logging.info("Pipeline complete: %s final candidates", len(final_candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())