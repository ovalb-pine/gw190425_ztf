"""Stage-3 images from full-frame ZTF difference images, cut out locally.

The per-candidate route asks IRSA (a) which exposures cover every candidate and
(b) for one cutout per candidate and exposure: thousands of small requests.  Here
the network work is done per *area* and per *exposure* instead:

1. one grid of query points over the candidates' bounding box -> the metadata of all
   exposures in the pre/post-trigger windows (a handful of requests);
2. a local point-in-footprint test (ra1..ra4 / dec1..dec4) decides which exposure
   covers which candidate;
3. every exposure that covers at least one candidate is downloaded once (frames already
   on disk, e.g. in ``diff_root``, are reused);
4. each frame is opened once and all its candidates are cut out with Cutout2D.

The cutouts get exactly the names the IRSA cutout route produced, so everything
downstream (parse_diff_name, _norm_key_name, the stage-1 significance reuse) is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from tqdm import tqdm

FRAME_SUFFIX = "scimrefdiffimg.fits.fz"
FRAME_KEY = ["filefracday", "field", "ccdid", "qid", "filtercode"]
_CORNER_COLS = [f"ra{i}" for i in range(1, 5)] + [f"dec{i}" for i in range(1, 5)]
_EST_MB_PER_FRAME = 40.0  # upper bound (full-frame sciimg is ~40 MB); diff images are usually smaller


@dataclass
class FullFrameConfig:
    frame_cache: Path                      # where missing full frames are downloaded
    cutout_root: Path                      # where the cutouts are written (<row_label>/<name>)
    local_roots: list[Path] = field(default_factory=list)  # existing full frames to reuse
    size_arcsec: float = 240.0             # same as the IRSA cutouts
    grid_step_deg: float = 0.05            # spacing of the metadata query points
    max_grid_positions: int = 4000
    max_frames: int = 3000                 # safety stop: raise it on purpose
    download_workers: int = 4
    cutout_workers: int = 4
    metadata_timeout: int = 300
    metadata_batch_size: int = 100


# --------------------------------------------------------------------------- #
# 1. Metadata for the area (not per candidate)
# --------------------------------------------------------------------------- #
def build_query_grid(ra, dec, step_deg: float, max_positions: int = 4000) -> list[tuple[float, float]]:
    """Grid of (ra, dec) query points covering the bounding box of the targets.

    Every quadrant that overlaps the box contains at least one point, apart from
    slivers thinner than ``step_deg`` (candidates there sit within ``step_deg`` of
    the frame edge anyway).
    """
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    ra, dec = ra[ok], dec[ok]
    if ra.size == 0:
        return []
    ra_ref = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(ra))))) % 360.0)  # wrap-safe centre
    dra = (ra - ra_ref + 180.0) % 360.0 - 180.0
    ny = int(np.ceil((dec.max() - dec.min()) / step_deg)) + 1
    ys = np.linspace(dec.min(), dec.max(), ny)
    # RA spacing is step/cos(dec) per row, so the on-sky spacing stays <= step and the RA edges are exact.
    nx_row = [int(np.ceil((dra.max() - dra.min()) * np.cos(np.radians(d)) / step_deg)) + 1 for d in ys]
    if sum(nx_row) > max_positions:
        raise ValueError(
            f"Query grid would have {sum(nx_row)} points (> {max_positions}); the candidates span too large an area. "
            "Raise full_frame_grid_step_deg or max_grid_positions."
        )
    g_ra, g_dec = [], []
    for d, n in zip(ys, nx_row):
        g_ra.extend(((ra_ref + np.linspace(dra.min(), dra.max(), n)) % 360.0).tolist())
        g_dec.extend([float(d)] * n)
    return list(zip(g_ra, g_dec))


def fetch_region_metadata(
    grid,
    work_dir: Path,
    date_start: str,
    date_end: str,
    cfg: FullFrameConfig,
    search_fn: Callable | None = None,
) -> pd.DataFrame:
    """Exposure metadata for the grid; one row per exposure/quadrant.

    The on-disk cache is only reused for identical settings (grid, dates): a changed grid
    or window starts a fresh cache instead of silently mixing results.
    """
    if search_fn is None:
        from ztf_downloads.ztf_search import metadata_search_resumable as search_fn  # noqa: N813

    work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = work_dir / "full_frame_region_meta.csv"
    progress = work_dir / "full_frame_region_meta.progress.json"
    sig_path = work_dir / "full_frame_region_meta.settings.json"
    settings = {
        "n_positions": len(grid),
        "grid_hash": hashlib.sha1(np.round(np.asarray(grid, float), 6).tobytes()).hexdigest(),
        "date_start": date_start,
        "date_end": date_end,
        "batch_size": int(cfg.metadata_batch_size),
    }
    if sig_path.exists() and json.loads(sig_path.read_text()) != settings:
        logging.info("Region metadata settings changed: discarding the old metadata cache")
        for p in (csv_path, progress):
            p.unlink(missing_ok=True)
    sig_path.write_text(json.dumps(settings, indent=1))

    df = search_fn(
        positions=grid,
        output_csv=str(csv_path),
        progress_json=str(progress),
        batch_size=int(cfg.metadata_batch_size),
        size_deg=0.01,
        date_start=date_start,
        date_end=date_end,
        timeout=int(cfg.metadata_timeout),
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    if "imgtypecode" in df.columns:
        df = df[df["imgtypecode"].astype(str).str.lower() == "o"]
    missing = [c for c in FRAME_KEY + _CORNER_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Region metadata lacks columns {missing}; cannot decide which frame covers a candidate")
    df["filefracday"] = pd.to_numeric(df["filefracday"], errors="coerce")
    df = df.dropna(subset=["filefracday"])
    df["filefracday"] = df["filefracday"].astype("int64")
    return df.drop_duplicates(subset=FRAME_KEY).reset_index(drop=True)


def frame_name(row) -> str:
    from ztf_downloads.ztf_download import build_sci_url

    return Path(urlparse(build_sci_url(row, suffix=FRAME_SUFFIX)).path).name


# --------------------------------------------------------------------------- #
# 2. Which frame covers which candidate (local)
# --------------------------------------------------------------------------- #
def quad_contains(corner_ra, corner_dec, ra, dec) -> np.ndarray:
    """Points inside the (convex) quadrilateral given by 4 corners; vectorised."""
    corner_ra = np.asarray(corner_ra, float)
    corner_dec = np.asarray(corner_dec, float)
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    ra0, dec0 = corner_ra[0], corner_dec.mean()
    cosd = np.cos(np.radians(dec0))
    cx = ((corner_ra - ra0 + 180.0) % 360.0 - 180.0) * cosd
    cy = corner_dec - dec0
    px = ((ra - ra0 + 180.0) % 360.0 - 180.0) * cosd
    py = dec - dec0
    order = np.argsort(np.arctan2(cy - cy.mean(), cx - cx.mean()))      # counter-clockwise
    cx, cy = cx[order], cy[order]
    inside = np.ones(px.shape, bool)
    for i in range(4):
        j = (i + 1) % 4
        cross = (cx[j] - cx[i]) * (py - cy[i]) - (cy[j] - cy[i]) * (px - cx[i])
        inside &= cross >= 0
    return inside & np.isfinite(px) & np.isfinite(py)


def plan_pairs(
    targets: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_window,
    post_window,
    post_filters: Iterable[str],
    parse_filefracday: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(target, frame) pairs to cut, using the same window rules as the per-candidate route.

    targets: object_id, row_label, ra, dec, need_pre, need_post
    pre_window = (start, trigger)  -> start <= t < trigger   (needs need_pre)
    post_window = (trigger, end)   -> trigger <= t <= end, filter in post_filters (needs need_post)
    Returns (pairs, frames_used); pairs has one row per cutout.
    """
    post_filters = {str(f).lower() for f in post_filters}
    f = frames.copy()
    f["obs_dt"] = f["filefracday"].map(lambda v: parse_filefracday(str(int(v)).zfill(14)))
    f = f.dropna(subset=["obs_dt"]).reset_index(drop=True)
    f["is_pre"] = (f["obs_dt"] >= pre_window[0]) & (f["obs_dt"] < pre_window[1])
    f["is_post"] = (
        (f["obs_dt"] >= post_window[0])
        & (f["obs_dt"] <= post_window[1])
        & f["filtercode"].astype(str).str.lower().isin(post_filters)
    )
    t_ra = targets["ra"].to_numpy(float)
    t_dec = targets["dec"].to_numpy(float)
    need_pre = targets["need_pre"].to_numpy(bool)
    need_post = targets["need_post"].to_numpy(bool)

    chunks = []
    for k, fr in f.iterrows():
        eligible = (need_pre & bool(fr["is_pre"])) | (need_post & bool(fr["is_post"]))
        if not eligible.any():
            continue
        idx = np.flatnonzero(eligible)
        inside = quad_contains(
            [fr[f"ra{i}"] for i in range(1, 5)], [fr[f"dec{i}"] for i in range(1, 5)], t_ra[idx], t_dec[idx]
        )
        hit = idx[inside]
        if hit.size:
            part = targets.iloc[hit][["object_id", "row_label", "ra", "dec"]].copy()
            part["frame_idx"] = k
            chunks.append(part)
    if not chunks:
        return pd.DataFrame(columns=["object_id", "row_label", "ra", "dec", "frame_idx"]), f.iloc[0:0]
    pairs = pd.concat(chunks, ignore_index=True)
    return pairs, f.loc[sorted(pairs["frame_idx"].unique())]


# --------------------------------------------------------------------------- #
# 3. Frames: reuse what is on disk, download the rest once
# --------------------------------------------------------------------------- #
def locate_local_frames(roots: Iterable[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("ztf_*scimrefdiffimg.fits*"):
            if p.is_file() and "__" not in p.name and p.stat().st_size > 0:
                found.setdefault(p.name, p)
    return found


def ensure_frames(frames: pd.DataFrame, cfg: FullFrameConfig, download_fn: Callable | None = None) -> dict[str, Path]:
    """Return {frame_name: path} for every frame that exists locally after downloading the missing ones."""
    from ztf_downloads.ztf_download import batch_download_resumable, build_sci_url

    if download_fn is None:
        download_fn = batch_download_resumable
    names = {frame_name(r): r for _, r in frames.iterrows()}
    local = locate_local_frames([cfg.frame_cache, *cfg.local_roots])
    todo = {n: r for n, r in names.items() if n not in local}
    logging.info(
        "Full frames: %s needed, %s already on disk, %s to download (<= ~%.1f GB)",
        len(names), len(names) - len(todo), len(todo), len(todo) * _EST_MB_PER_FRAME / 1024.0,
    )
    if len(names) > cfg.max_frames:
        raise RuntimeError(
            f"{len(names)} full frames needed (> full_frame_max_frames={cfg.max_frames}). "
            "Check the candidate list / windows, or raise the limit on purpose."
        )
    if todo:
        items = [
            (build_sci_url(r, suffix=FRAME_SUFFIX), n, str(int(r["filefracday"]))[:8]) for n, r in todo.items()
        ]
        # The progress file is tied to this exact item list: chunk indices of a different list would be skipped.
        tag = hashlib.sha1("|".join(sorted(n for n, _, _ in items)).encode()).hexdigest()[:12]
        download_fn(
            items,
            out_dir=str(cfg.frame_cache),
            progress_json=str(cfg.frame_cache / f"full_frames_{tag}.progress.json"),
            max_workers=int(cfg.download_workers),
            chunk_size=50,
        )
        local = locate_local_frames([cfg.frame_cache, *cfg.local_roots])
    missing = [n for n in names if n not in local]
    if missing:
        logging.warning("%s frames are not available (no difference image at IRSA?); their cutouts are skipped", len(missing))
    return {n: local[n] for n in names if n in local}


# --------------------------------------------------------------------------- #
# 4. Cut out
# --------------------------------------------------------------------------- #
_WCS_KEY = re.compile(r"^(A|B|AP|BP)_(ORDER|\d+_\d+)$|^(PV|CD|PC|CDELT|CRPIX|CRVAL|CTYPE|CUNIT)\d*(_\d+)?$")
_STRUCT_KEYS = ("SIMPLE", "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "PCOUNT", "GCOUNT",
                "BZERO", "BSCALE", "CHECKSUM", "DATASUM", "LONPOLE", "LATPOLE")


def cutout_path(cfg: FullFrameConfig, row_label: str, ra: float, dec: float, filtercode: str, filefracday: int, fname: str) -> Path:
    name = f"{row_label}_RA{ra:.4f}_DEC{dec:.4f}_{filtercode}_{int(filefracday):014d}__{fname}"
    return Path(cfg.cutout_root) / str(row_label) / name


def _first_2d(path: Path):
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if isinstance(data, np.ndarray) and data.ndim == 2:
                return np.asarray(data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"No 2D image in {path}")


def _cutout_header(header, cut: Cutout2D, src_name: str, ra: float, dec: float, size_arcsec: float):
    hdr = header.copy()
    for key in list(hdr.keys()):
        if key in _STRUCT_KEYS or _WCS_KEY.match(key) or key in ("RADESYS", "EQUINOX", "WCSAXES"):
            hdr.remove(key, ignore_missing=True, remove_all=True)
    hdr.update(cut.wcs.to_header(relax=True))
    hdr["CUTSRC"] = (src_name[:68], "full frame this cutout was taken from")
    hdr["CUTRA"] = (float(ra), "[deg] requested cutout centre")
    hdr["CUTDEC"] = (float(dec), "[deg] requested cutout centre")
    hdr["CUTSIZE"] = (float(size_arcsec), "[arcsec] requested cutout size")
    return hdr


def _cut_one_frame(task):
    """Open one frame once; write all its cutouts. Returns a list of (status, out_path)."""
    frame_path, src_name, jobs, size_arcsec = task
    out: list[tuple[str, str]] = []
    pending = [(ra, dec, op) for ra, dec, op in jobs if not (Path(op).exists() and Path(op).stat().st_size > 0)]
    out += [("exists", op) for _, _, op in jobs if (Path(op).exists() and Path(op).stat().st_size > 0)]
    if not pending:
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            data, header = _first_2d(Path(frame_path))
            wcs = WCS(header)
            wcs = wcs.celestial if getattr(wcs, "naxis", 2) > 2 else wcs
            ny, nx = data.shape
            for ra, dec, op in pending:
                x, y = wcs.all_world2pix(float(ra), float(dec), 0)
                if not (np.isfinite(x) and np.isfinite(y) and -0.5 <= x < nx - 0.5 and -0.5 <= y < ny - 0.5):
                    out.append(("off_frame", op))
                    continue
                cut = Cutout2D(
                    data,
                    SkyCoord(float(ra) * u.deg, float(dec) * u.deg),
                    size=(size_arcsec * u.arcsec, size_arcsec * u.arcsec),
                    wcs=wcs,
                    mode="trim",
                )
                hdr = _cutout_header(header, cut, src_name, ra, dec, size_arcsec)
                op = Path(op)
                op.parent.mkdir(parents=True, exist_ok=True)
                tmp = op.with_name(op.name + ".part")
                fits.PrimaryHDU(data=np.ascontiguousarray(cut.data, dtype=np.float32), header=hdr).writeto(tmp, overwrite=True)
                tmp.replace(op)   # plain FITS under the original ".fits.fz" name, like the IRSA cutout service delivered
                out.append(("written", str(op)))
    except Exception as exc:  # one broken frame must not stop the others
        out += [("error:" + str(exc)[:120], op) for _, _, op in pending]
    return out


def cut_out_all(pairs: pd.DataFrame, frames: pd.DataFrame, paths: dict[str, Path], cfg: FullFrameConfig) -> dict:
    tasks = []
    for k, grp in pairs.groupby("frame_idx", sort=False):
        fr = frames.loc[k]
        fname = frame_name(fr)
        if fname not in paths:
            continue
        jobs = [
            (float(r.ra), float(r.dec),
             str(cutout_path(cfg, r.row_label, r.ra, r.dec, str(fr["filtercode"]), int(fr["filefracday"]), fname)))
            for r in grp.itertuples()
        ]
        tasks.append((str(paths[fname]), fname, jobs, float(cfg.size_arcsec)))

    counts: dict[str, int] = {}
    errors: list[str] = []

    def _tally(results):
        for status, op in results:
            key = "error" if status.startswith("error") else status
            counts[key] = counts.get(key, 0) + 1
            if key == "error":
                errors.append(f"{Path(op).name}: {status}")

    if cfg.cutout_workers <= 1 or len(tasks) <= 1:
        for t in tqdm(tasks, desc="Cutting out of full frames", unit="frame", dynamic_ncols=True):
            _tally(_cut_one_frame(t))
    else:
        with ProcessPoolExecutor(max_workers=int(cfg.cutout_workers)) as ex:
            futs = [ex.submit(_cut_one_frame, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Cutting out of full frames", unit="frame", dynamic_ncols=True):
                _tally(fut.result())
    if errors:
        logging.warning("%s cutouts failed, e.g. %s", len(errors), errors[:3])
    return counts


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_full_frame_cutouts(
    targets: pd.DataFrame,
    *,
    pre_window,
    post_window,
    post_filters=("zg", "zr", "zi"),
    cfg: FullFrameConfig,
    work_dir: Path,
    parse_filefracday: Callable,
    search_fn: Callable | None = None,
    download_fn: Callable | None = None,
) -> dict:
    """Make every missing pre/post-trigger cutout of ``targets`` from full frames.

    targets: object_id, row_label, ra, dec, need_pre, need_post (bool).
    pre_window = (start, trigger), post_window = (trigger, end): pandas Timestamps.
    """
    targets = targets.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    targets = targets[targets["need_pre"] | targets["need_post"]].reset_index(drop=True)
    if targets.empty:
        return {"targets": 0}

    starts, ends = [], []
    if targets["need_pre"].any():
        starts.append(pre_window[0]); ends.append(pre_window[1])
    if targets["need_post"].any():
        starts.append(post_window[0]); ends.append(post_window[1])
    # IRSA's lower date bound is exclusive: start one day early; the exact cut is done below.
    date_start = (min(starts) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    date_end = (max(ends) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    grid = build_query_grid(targets["ra"], targets["dec"], cfg.grid_step_deg, cfg.max_grid_positions)
    logging.info("Full-frame route: %s targets, %s metadata query points, %s..%s", len(targets), len(grid), date_start, date_end)
    meta = fetch_region_metadata(grid, Path(work_dir), date_start, date_end, cfg, search_fn=search_fn)
    if meta.empty:
        logging.warning("No exposures found for the area/window")
        return {"targets": len(targets), "frames_in_area": 0, "cutouts": 0}

    pairs, frames = plan_pairs(
        targets, meta, pre_window=pre_window, post_window=post_window,
        post_filters=post_filters, parse_filefracday=parse_filefracday,
    )
    logging.info("%s exposures in the area -> %s (target, exposure) pairs on %s frames", len(meta), len(pairs), len(frames))
    if pairs.empty:
        return {"targets": len(targets), "frames_in_area": int(len(meta)), "cutouts": 0}

    paths = ensure_frames(frames, cfg, download_fn=download_fn)
    counts = cut_out_all(pairs, frames, paths, cfg)
    logging.info("Cutouts: %s", counts)
    return {"targets": len(targets), "frames_in_area": int(len(meta)), "pairs": int(len(pairs)),
            "frames_used": int(len(frames)), "frames_available": len(paths), **counts}
