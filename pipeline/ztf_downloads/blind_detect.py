"""Blind source detection on full-frame ZTF difference images.

Replaces the galaxy-position SNR step (stage 1) when working with full images:
instead of measuring one peak near a catalogue position, every peak with
matched-filter SNR >= ``snr_min`` is found anywhere in the image, measured, and
detections of the same sky position are linked across images into object ids.

Output tables keep the column names of the old stage-1 images/objects tables
(``object_id``, ``image_path``, ``source_x_px``, ``source_y_px``, ``snr``,
``snr_empirical``, ``fwhm_px``, ``centroid_ra``, ``centroid_dec`` ...), so stage 2
(triplets), BRAAI and the later stages can read them unchanged.

Significance
------------
``snr_empirical`` is the peak of the matched-filter SNR map: the difference image
is convolved with a Gaussian of the header FWHM and divided by the local rms of
the *filtered* image (so correlated noise is accounted for). The old "off-source
null apertures" correct for choosing the brightest peak near one position; in a
blind search the equivalent check is the sign-flipped image, so the same peak
finder is run on ``-SNR`` and ``n_neg_ge_snr`` reports how many negative peaks at
least this bright the image produced (an empirical false-alarm count).
"""

from __future__ import annotations

import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground, StdBackgroundRMS
from photutils.detection import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm

try:  # package import (run from pipeline/)
    from .snr_photometry_quadratic import (
        _conservative_centroid,
        _extract_fwhm_px,
        _extract_zero_point,
        _odd_at_least,
        first_2d_data_and_header,
    )
except ImportError:  # pragma: no cover - flat-module fallback
    from snr_photometry_quadratic import (  # type: ignore
        _conservative_centroid,
        _extract_fwhm_px,
        _extract_zero_point,
        _odd_at_least,
        first_2d_data_and_header,
    )

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493

# Full-frame names, e.g. ztf_20190426386979_000534_zr_c12_o_q1_scimrefdiffimg.fits.fz
_FULL_NAME = re.compile(
    r"ztf_(?P<filefracday>\d{14})_(?P<field>\d{6})_(?P<filter>z[gri])"
    r"_c(?P<ccdid>\d+)_o_q(?P<qid>\d)_scimrefdiffimg"
)


# --------------------------------------------------------------------------- #
# Stage 0: index of full difference images
# --------------------------------------------------------------------------- #
def parse_full_diff_name(name: str) -> dict | None:
    match = _FULL_NAME.search(name)
    if not match:
        return None
    return {
        "filefracday": match.group("filefracday"),
        "field": int(match.group("field")),
        "filter": match.group("filter"),
        "ccdid": int(match.group("ccdid")),
        "qid": int(match.group("qid")),
    }


def collect_full_diff_index(diff_root: Path) -> pd.DataFrame:
    """One row per full difference image found under ``diff_root``."""
    rows = []
    for path in sorted(Path(diff_root).rglob("*scimrefdiffimg.fits*")):
        parsed = parse_full_diff_name(path.name)
        if parsed is None:
            rows.append({"image_path": str(path), "image_name": path.name, "parse_status": "unparsed"})
            continue
        rows.append({"image_path": str(path), "image_name": path.name, "parse_status": "ok", **parsed})
    df = pd.DataFrame(rows)
    if df.empty or "parse_status" not in df.columns:
        return df
    ok = df[df["parse_status"] == "ok"].sort_values("image_path")
    ok = ok.drop_duplicates(subset=["field", "ccdid", "qid", "filefracday", "filter"], keep="first")
    return pd.concat([ok, df[df["parse_status"] != "ok"]], ignore_index=True)


# --------------------------------------------------------------------------- #
# Matched filter + peak finding
# --------------------------------------------------------------------------- #
def _background_and_rms(arr: np.ndarray, finite: np.ndarray, box: int):
    """Smooth background and rms maps (sigma-clipped, masked where not finite)."""
    filled = np.where(finite, arr, 0.0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        bkg = Background2D(
            filled,
            (box, box),
            mask=~finite,
            exclude_percentile=50.0,
            filter_size=(3, 3),
            sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
            bkg_estimator=MedianBackground(),
            bkgrms_estimator=StdBackgroundRMS(),
        )
    return bkg.background, bkg.background_rms


def matched_filter_snr(data: np.ndarray, valid: np.ndarray, fwhm_px: float, box: int = 64):
    """Per-pixel matched-filter SNR map (Gaussian PSF of the header FWHM).

    Returns ``(snr_map, ok)``; ``snr_map`` is NaN where fewer than half of the
    kernel footprint is valid.
    """
    sigma = float(fwhm_px) * FWHM_TO_SIGMA
    weight = gaussian_filter(valid.astype(np.float32), sigma, mode="constant")
    flux = gaussian_filter(np.where(valid, data, 0.0).astype(np.float32), sigma, mode="constant")
    ok = weight > 0.5
    filt = np.full(data.shape, np.nan, dtype=np.float32)
    filt[ok] = flux[ok] / weight[ok]
    bkg, rms = _background_and_rms(filt, ok, box)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = (filt - bkg) / rms
    snr[~ok | ~np.isfinite(snr)] = np.nan
    return snr, ok


def _peaks(snr_map: np.ndarray, threshold: float, box_size: int) -> np.ndarray:
    """(x, y, snr) of local maxima above ``threshold`` as an (N, 3) array."""
    finite = np.isfinite(snr_map)
    tbl = find_peaks(np.where(finite, snr_map, 0.0), threshold, box_size=box_size, mask=~finite)
    if tbl is None or len(tbl) == 0:
        return np.empty((0, 3))
    return np.column_stack([tbl["x_peak"], tbl["y_peak"], tbl["peak_value"]]).astype(float)


# --------------------------------------------------------------------------- #
# One image
# --------------------------------------------------------------------------- #
def detect_in_diff_image(
    image_path,
    *,
    snr_min: float = 3.0,
    bkg_box: int = 64,
    min_valid_pixel: float = -4000.0,
    min_valid_frac: float = 0.9,
    max_peaks: int = 50000,
    refine_centroid: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Find and measure all peaks with matched-filter SNR >= ``snr_min``."""
    p = Path(image_path)
    data, header = first_2d_data_and_header(p)
    fwhm_px, fwhm_key = _extract_fwhm_px(header)
    ap_r = 2.0 * fwhm_px
    valid = np.isfinite(data) & (data > float(min_valid_pixel))

    snr_map, _ = matched_filter_snr(data, valid, fwhm_px, box=bkg_box)
    peak_box = _odd_at_least(int(round(2.0 * fwhm_px)), 5)

    pos = _peaks(snr_map, snr_min, peak_box)
    neg = _peaks(-snr_map, min(snr_min, 2.0), peak_box)   # sign-flipped image = empirical null
    neg_snr = np.sort(neg[:, 2]) if len(neg) else np.empty(0)

    stats = {
        "image_path": str(p),
        "status": "ok",
        "fwhm_px": float(fwhm_px),
        "n_pos_peaks": int(len(pos)),
        "n_neg_peaks_ge_snr_min": int((neg_snr >= snr_min).sum()),
        "valid_fraction": float(valid.mean()),
    }
    if len(pos) == 0:
        return pd.DataFrame(), stats
    if len(pos) > max_peaks:
        pos = pos[np.argsort(pos[:, 2])[::-1][:max_peaks]]
        stats["status"] = f"truncated_to_{max_peaks}_peaks"

    # centroid on the difference image (same routine as the old stage 1)
    xy = pos[:, :2].copy()
    if refine_centroid:
        for i, (x, y) in enumerate(xy):
            xy[i] = _conservative_centroid(data, x, y, fwhm_px, max_shift_fwhm=1.0)

    # aperture photometry for every peak in one call
    aper = CircularAperture(xy, r=ap_r)
    area = float(aper.area)
    ap_sum = np.asarray(aperture_photometry(np.where(valid, data, 0.0), aper)["aperture_sum"], dtype=float)
    valid_frac = np.asarray(aperture_photometry(valid.astype(float), aper)["aperture_sum"], dtype=float) / area

    bkg_pix, rms_pix = _background_and_rms(data, valid, bkg_box)
    ix = np.clip(np.round(xy[:, 0]).astype(int), 0, data.shape[1] - 1)
    iy = np.clip(np.round(xy[:, 1]).astype(int), 0, data.shape[0] - 1)
    bkg_at = bkg_pix[iy, ix]
    rms_at = rms_pix[iy, ix]
    net = ap_sum - bkg_at * area
    flux_err = np.sqrt(area) * rms_at
    with np.errstate(invalid="ignore", divide="ignore"):
        snr_ap = net / flux_err

    zp = _extract_zero_point(header)
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.where(np.isfinite(zp) & (net > 0), zp - 2.5 * np.log10(net), np.nan)
        mag_err = np.where(net > 0, 1.0857362047581294 * flux_err / net, np.nan)

    wcs = WCS(header)
    wcs = wcs.celestial if getattr(wcs, "naxis", 2) > 2 else wcs
    ra, dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)

    n_neg_ge = len(neg_snr) - np.searchsorted(neg_snr, pos[:, 2], side="left")
    parsed = parse_full_diff_name(p.name) or {}
    snr_peak = pos[:, 2]

    det = pd.DataFrame(
        {
            "image_path": str(p),
            "filefracday": parsed.get("filefracday", ""),
            "filter": parsed.get("filter", ""),
            "field": parsed.get("field", -1),
            "ccdid": parsed.get("ccdid", -1),
            "qid": parsed.get("qid", -1),
            "source_x_px": xy[:, 0],
            "source_y_px": xy[:, 1],
            "centroid_ra": ra,
            "centroid_dec": dec,
            "ra": ra,
            "dec": dec,
            "fwhm_px": fwhm_px,
            "fwhm_header_key": fwhm_key,
            "aperture_radius_px": ap_r,
            "aperture_area_px": area,
            "aperture_sum": ap_sum,
            "background_median": bkg_at,
            "background_std": rms_at,
            "net_flux": net,
            "flux_err": flux_err,
            "snr": snr_ap,                       # aperture SNR, diagnostic only
            "snr_empirical": snr_peak,           # matched-filter peak SNR (decides detection)
            "n_neg_ge_snr": n_neg_ge.astype(int),
            "significance_tier": np.where(snr_peak >= 5.0, "secure", "marginal"),
            "mag": mag,
            "mag_err": mag_err,
            "zero_point": zp,
            "valid_aperture_frac": valid_frac,
            "detection": True,
            "status": "ok",
        }
    )
    det = det[det["valid_aperture_frac"] >= float(min_valid_frac)].reset_index(drop=True)
    return det, stats


def _detect_task(task: tuple):
    path, kwargs = task
    try:
        return detect_in_diff_image(path, **kwargs)
    except Exception as exc:  # keep going; the error is recorded per image
        return pd.DataFrame(), {"image_path": str(path), "status": f"error: {exc}"}


def detect_all(index_df: pd.DataFrame, *, workers: int = 1, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run :func:`detect_in_diff_image` on every image of ``index_df``."""
    tasks = [(row["image_path"], kwargs) for _, row in index_df.iterrows()]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            results = list(tqdm(pool.map(_detect_task, tasks), total=len(tasks), desc="Blind detection", unit="image"))
    else:
        results = [_detect_task(t) for t in tqdm(tasks, desc="Blind detection", unit="image")]
    dets = [d for d, _ in results if d is not None and not d.empty]
    detections = pd.concat(dets, ignore_index=True) if dets else pd.DataFrame()
    image_stats = pd.DataFrame([s for _, s in results])
    return detections, image_stats


# --------------------------------------------------------------------------- #
# Link detections across images -> object ids
# --------------------------------------------------------------------------- #
def link_detections(
    det: pd.DataFrame,
    *,
    radius_arcsec: float = 2.0,
    min_detections: int = 1,
    min_epochs: int = 1,
    id_prefix: str = "B",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group detections of the same sky position (friends-of-friends) into objects.

    An *epoch* is a distinct ``filefracday`` (one exposure), so the same source
    seen in two overlapping quadrants/fields of one exposure counts once.
    Returns ``(detections_with_object_id, objects)``.
    """
    if det.empty:
        return det.assign(object_id=pd.Series(dtype=str)), pd.DataFrame()

    ra = np.deg2rad(det["centroid_ra"].to_numpy(float))
    dec = np.deg2rad(det["centroid_dec"].to_numpy(float))
    xyz = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    chord = 2.0 * np.sin(np.deg2rad(radius_arcsec / 3600.0) / 2.0)
    pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    n = len(det)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, labels = connected_components(graph, directed=False)

    det = det.copy()
    det["_label"] = labels
    grp = det.groupby("_label")
    summary = grp.agg(
        n_det=("image_path", "count"),
        n_epochs=("filefracday", "nunique"),
        best_snr_empirical=("snr_empirical", "max"),
        best_snr_naive=("snr", "max"),
        first_obs_date=("filefracday", "min"),
        last_obs_date=("filefracday", "max"),
    )
    keep = summary[(summary["n_det"] >= min_detections) & (summary["n_epochs"] >= min_epochs)].copy()
    keep = keep.sort_values("best_snr_empirical", ascending=False)
    keep["object_id"] = [f"{id_prefix}{i:07d}" for i in range(1, len(keep) + 1)]

    det = det[det["_label"].isin(keep.index)].copy()
    det["object_id"] = det["_label"].map(keep["object_id"])
    # SNR-weighted mean position per object (unit vectors, so RA wrap is safe)
    w = np.clip(det["snr_empirical"].to_numpy(float), 1e-6, None)
    r, d = np.deg2rad(det["centroid_ra"].to_numpy(float)), np.deg2rad(det["centroid_dec"].to_numpy(float))
    vec = pd.DataFrame({"x": w * np.cos(d) * np.cos(r), "y": w * np.cos(d) * np.sin(r), "z": w * np.sin(d)})
    vec["_label"] = det["_label"].to_numpy()
    mean = vec.groupby("_label").sum()
    keep["ra"] = np.rad2deg(np.arctan2(mean["y"], mean["x"])).reindex(keep.index) % 360.0
    keep["dec"] = np.rad2deg(np.arctan2(mean["z"], np.hypot(mean["x"], mean["y"]))).reindex(keep.index)

    objects = keep.reset_index(drop=True)
    objects["n_pass_images"] = objects["n_det"]
    objects["best_snr"] = objects["best_snr_empirical"]
    det = det.drop(columns="_label").sort_values(["object_id", "filefracday"]).reset_index(drop=True)
    return det, objects[["object_id", "ra", "dec", "n_det", "n_epochs", "best_snr", "best_snr_empirical",
                         "best_snr_naive", "n_pass_images", "first_obs_date", "last_obs_date"]]


# --------------------------------------------------------------------------- #
# Drop-in replacement for stage1_quadratic_snr
# --------------------------------------------------------------------------- #
def stage1_blind_search(
    diff_index: pd.DataFrame,
    out_root: Path,
    *,
    snr_threshold: float = 3.0,
    min_valid_pixel: float = -4000.0,
    min_detections: int = 1,
    min_epochs: int = 1,
    link_radius_arcsec: float = 2.0,
    max_workers: int = 1,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blind stage 1: writes the same two CSVs the old stage 1 wrote."""
    stage1_dir = Path(out_root) / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    images_path = stage1_dir / "april_stage1_quadratic_snr_images.csv"
    objects_path = stage1_dir / "april_stage1_quadratic_snr_objects.csv"
    stats_path = stage1_dir / "blind_image_stats.csv"

    if resume and images_path.exists() and objects_path.exists():
        cached = pd.read_csv(images_path, low_memory=False)
        if {"object_id", "centroid_ra", "centroid_dec", "snr_empirical"}.issubset(cached.columns):
            return cached, pd.read_csv(objects_path, low_memory=False)

    det, image_stats = detect_all(diff_index, workers=max_workers, snr_min=snr_threshold, min_valid_pixel=min_valid_pixel)
    image_stats.to_csv(stats_path, index=False)
    if det.empty:
        det.to_csv(images_path, index=False)
        pd.DataFrame().to_csv(objects_path, index=False)
        return det, pd.DataFrame()

    det, objects = link_detections(
        det, radius_arcsec=link_radius_arcsec, min_detections=min_detections, min_epochs=min_epochs
    )
    det.to_csv(images_path, index=False)
    objects.to_csv(objects_path, index=False)
    return det, objects
