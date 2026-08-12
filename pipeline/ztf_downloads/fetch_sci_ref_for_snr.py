import argparse
from pathlib import Path
import shutil
import sys
import re
import time

import numpy as np
import pandas as pd

# Allow running from braai/ while importing workspace modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ztf_downloads.ztf_download import build_sci_url, build_cutout_url, download_file


def compute_host_offset(detection_ra, detection_dec, host_ra, host_dec):
    """Return angular separation between a detection and host galaxy center in arcsec."""
    values = np.asarray([detection_ra, detection_dec, host_ra, host_dec], dtype=float)
    if not np.isfinite(values).all():
        return np.nan

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    detection = SkyCoord(ra=float(detection_ra) * u.deg, dec=float(detection_dec) * u.deg)
    host = SkyCoord(ra=float(host_ra) * u.deg, dec=float(host_dec) * u.deg)
    return float(detection.separation(host).arcsec)


def _safe_int(v):
    try:
        return int(v)
    except Exception:
        return None


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def _norm_key_name(name: str) -> str:
    # Many files are stored as "label__ztf_..._sciimg.fits".
    # Keep only the ztf_* tail when available for robust matching.
    if "__ztf_" in name:
        return name.split("__", 1)[1]
    return name


def _extract_object_id(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    m = re.search(r"(\d{8,})", text)
    return m.group(1) if m else text


def _build_exact_coord_index(final_class_df: pd.DataFrame):
    id_col = None
    for candidate in ("objID_SDSS-DR16", "objID", "object_id"):
        if candidate in final_class_df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError("Final class table must contain one of: objID_SDSS-DR16, objID, object_id")

    for candidate in ("ra_fin", "ra"):
        if candidate in final_class_df.columns:
            ra_col = candidate
            break
    else:
        raise ValueError("Final class table must contain ra_fin (or ra)")

    for candidate in ("dec_fin", "dec"):
        if candidate in final_class_df.columns:
            dec_col = candidate
            break
    else:
        raise ValueError("Final class table must contain dec_fin (or dec)")

    work = final_class_df[[id_col, ra_col, dec_col]].copy()
    work[id_col] = work[id_col].map(_extract_object_id)
    work[ra_col] = pd.to_numeric(work[ra_col], errors="coerce")
    work[dec_col] = pd.to_numeric(work[dec_col], errors="coerce")
    work = work[work[id_col].astype(bool)].copy()
    work = work[np.isfinite(work[ra_col]) & np.isfinite(work[dec_col])].copy()
    work = work.drop_duplicates(subset=[id_col], keep="first")

    return {
        str(row[id_col]): (float(row[ra_col]), float(row[dec_col]))
        for _, row in work.iterrows()
    }


def parse_diff_name(name: str):
    pat = re.compile(
        r"(?P<object_id>\d+)_RA(?P<ra>[-+\d\.]+)_DEC(?P<dec>[-+\d\.]+)_"
        r"(?P<filter>z[gri])_(?P<filefracday>\d+)__ztf_\d+_"
        r"(?P<field>\d{6})_(?P<filter2>z[gri])_c(?P<ccdid>\d+)_o_q(?P<qid>\d)_scimrefdiffimg"
    )
    m = pat.search(name)
    if not m:
        return None
    return {
        "ra": float(m.group("ra")),
        "dec": float(m.group("dec")),
        "filter": m.group("filter"),
        "filefracday": m.group("filefracday"),
        "field": int(m.group("field")),
        "ccdid": int(m.group("ccdid")),
        "qid": int(m.group("qid")),
    }


def build_file_index(root: Path):
    idx = {}
    if not root.exists():
        return idx
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        nm = p.name
        key = _norm_key_name(nm)
        idx.setdefault(key, p)
    return idx


def choose_ref_row(ref_meta: pd.DataFrame, row: pd.Series, max_sep_arcsec: float):
    f = str(row["filter"]) if "filter" in row else str(row["filtercode"])
    fld = _safe_int(row["field"])
    ccd = _safe_int(row["ccdid"])
    qid = _safe_int(row["qid"])
    ra = _safe_float(row["ra"])
    dec = _safe_float(row["dec"])

    # Try strict match (filter + field + ccdid + qid). If no rows found,
    # progressively relax to (filter + field) and then (filter) only.
    sub_strict = ref_meta[
        (ref_meta["filtercode"].astype(str) == f)
        & (pd.to_numeric(ref_meta["field"], errors="coerce") == fld)
        & (pd.to_numeric(ref_meta["ccdid"], errors="coerce") == ccd)
        & (pd.to_numeric(ref_meta["qid"], errors="coerce") == qid)
    ].copy()

    if len(sub_strict) > 0:
        sub = sub_strict
    else:
        sub_field = ref_meta[
            (ref_meta["filtercode"].astype(str) == f)
            & (pd.to_numeric(ref_meta["field"], errors="coerce") == fld)
        ].copy()
        if len(sub_field) > 0:
            sub = sub_field
        else:
            sub_filter = ref_meta[(ref_meta["filtercode"].astype(str) == f)].copy()
            if len(sub_filter) > 0:
                sub = sub_filter
            else:
                return None

    dra = (pd.to_numeric(sub["in_ra"], errors="coerce").to_numpy() - ra) * np.cos(np.deg2rad(dec))
    ddec = pd.to_numeric(sub["in_dec"], errors="coerce").to_numpy() - dec
    sep_arcsec = np.sqrt(dra ** 2 + ddec ** 2) * 3600.0

    i = int(np.nanargmin(sep_arcsec))
    if np.isfinite(sep_arcsec[i]) and sep_arcsec[i] <= max_sep_arcsec:
        out = sub.iloc[i].copy()
        out["match_sep_arcsec"] = float(sep_arcsec[i])
        return out

    # If no good match found in the restricted subset, try a global nearest
    # neighbor across all reference metadata and accept it if it's reasonably
    # close (allow up to 2x the configured max separation).
    global_dra = (pd.to_numeric(ref_meta["in_ra"], errors="coerce").to_numpy() - ra) * np.cos(np.deg2rad(dec))
    global_ddec = pd.to_numeric(ref_meta["in_dec"], errors="coerce").to_numpy() - dec
    global_sep_arcsec = np.sqrt(global_dra ** 2 + global_ddec ** 2) * 3600.0
    j = int(np.nanargmin(global_sep_arcsec))
    if np.isfinite(global_sep_arcsec[j]) and global_sep_arcsec[j] <= (max_sep_arcsec * 2.0):
        out = ref_meta.iloc[j].copy()
        out["match_sep_arcsec"] = float(global_sep_arcsec[j])
        return out

    return None


def copy_if_exists(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def fmt_seconds(value: float) -> str:
    value = max(0.0, float(value))
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(int(round(value)), 60)
    return f"{minutes}m{seconds:02d}s"


def resolve_image_path(raw_path: str, snr_csv: Path) -> Path:
    diff_path = Path(raw_path)
    if not diff_path.is_absolute():
        diff_path = (snr_csv.parent / diff_path).resolve()
    else:
        diff_path = diff_path.resolve()
    return diff_path


def resolve_cli_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        (Path.cwd() / p),
        (repo_root / p),
        (repo_root / p.name),
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return (Path.cwd() / p).resolve()


def build_triplet_record(row, snr_csv: Path, diff_root: Path, out_root: Path, sci_index, ref_index, ref_meta, max_ref_sep_arcsec: float, download_missing: bool, coord_index):
    raw_diff = str(row["image_path"])
    diff_path = resolve_image_path(raw_diff, snr_csv)

    if not diff_path.exists():
        return {
            "image_path": raw_diff,
            "object_id": row.get("object_id", ""),
            "status": "missing_diff",
            "diff_path": "",
            "sci_status": "missing",
            "sci_path": "",
            "ref_status": "missing",
            "ref_path": "",
            "ref_match_sep_arcsec": np.nan,
        }

    parsed = parse_diff_name(diff_path.name)
    if parsed is None:
        return {
            "image_path": raw_diff,
            "object_id": row.get("object_id", ""),
            "status": "bad_diff_name",
            "diff_path": "",
            "sci_status": "missing",
            "sci_path": "",
            "ref_status": "missing",
            "ref_path": "",
            "ref_match_sep_arcsec": np.nan,
        }

    object_id = _extract_object_id(row.get("object_id", ""))
    coords_from_id = coord_index.get(object_id)
    if coords_from_id is not None:
        ra, dec = coords_from_id
        coord_source = "final_class_by_id"
    elif "ra" in row.index and "dec" in row.index and pd.notna(row["ra"]) and pd.notna(row["dec"]):
        ra = _safe_float(row["ra"])
        dec = _safe_float(row["dec"])
        coord_source = "snr_csv"
    else:
        ra = parsed["ra"]
        dec = parsed["dec"]
        coord_source = "diff_filename"
    filt = str(row["filter"]) if "filter" in row.index and pd.notna(row["filter"]) else parsed["filter"]
    filefracday = str(row["filefracday"]) if "filefracday" in row.index and pd.notna(row["filefracday"]) else parsed["filefracday"]
    field = _safe_int(row["field"]) if "field" in row.index and pd.notna(row["field"]) else parsed["field"]
    ccdid = _safe_int(row["ccdid"]) if "ccdid" in row.index and pd.notna(row["ccdid"]) else parsed["ccdid"]
    qid = _safe_int(row["qid"]) if "qid" in row.index and pd.notna(row["qid"]) else parsed["qid"]

    try:
        rel_from_diff_root = diff_path.relative_to(diff_root)
    except Exception:
        rel_from_diff_root = Path(diff_path.name)

    group_dir = out_root / rel_from_diff_root.parent / rel_from_diff_root.stem
    diff_dst = group_dir / "diff" / diff_path.name
    copy_if_exists(diff_path, diff_dst)

    sci_row = {
        "filefracday": filefracday,
        "field": field,
        "filtercode": filt,
        "ccdid": ccdid,
        "qid": qid,
        "imgtypecode": "o",
    }
    sci_url = build_sci_url(sci_row, suffix="sciimg.fits")
    sci_name = Path(sci_url).name
    sci_src = sci_index.get(sci_name)
    sci_dst = group_dir / "sci" / sci_name

    if sci_src is not None and sci_src.exists():
        copy_if_exists(sci_src, sci_dst)
        sci_status = "local"
    elif download_missing:
        cut_url = build_cutout_url(sci_url, ra, dec, size_arcsec=240)
        try:
            downloaded = download_file(cut_url, out_dir=sci_dst.parent, label=None, filtercode=None)
            sci_status = "downloaded"
            sci_dst = Path(downloaded)
        except Exception:
            sci_status = "missing"
    else:
        sci_status = "missing"

    row_for_ref = {
        "filter": filt,
        "field": field,
        "ccdid": ccdid,
        "qid": qid,
        "ra": ra,
        "dec": dec,
    }
    ref_choice = choose_ref_row(ref_meta, row_for_ref, max_ref_sep_arcsec)
    ref_dst = None
    ref_sep = np.nan
    if ref_choice is None:
        ref_status = "no_ref_match"
    else:
        ref_sep = float(ref_choice["match_sep_arcsec"])
        ref_row = {
            "filefracday": str(ref_choice["filefracday"]),
            "field": _safe_int(ref_choice["field"]),
            "filtercode": str(ref_choice["filtercode"]),
            "ccdid": _safe_int(ref_choice["ccdid"]),
            "qid": _safe_int(ref_choice["qid"]),
            "imgtypecode": str(ref_choice.get("imgtypecode", "o")),
        }
        ref_url = build_sci_url(ref_row, suffix="sciimg.fits")
        ref_name = Path(ref_url).name
        ref_src = ref_index.get(ref_name)
        ref_dst = group_dir / "ref" / ref_name

        if ref_src is not None and ref_src.exists():
            copy_if_exists(ref_src, ref_dst)
            ref_status = "local"
        elif download_missing:
            cut_url = build_cutout_url(ref_url, ra, dec, size_arcsec=240)
            try:
                downloaded = download_file(cut_url, out_dir=ref_dst.parent, label=None, filtercode=None)
                ref_status = "downloaded"
                ref_dst = Path(downloaded)
            except Exception:
                ref_status = "missing"
        else:
            ref_status = "missing"

    status = "ok" if (sci_status in {"local", "downloaded"} and ref_status in {"local", "downloaded"}) else "partial"
    return {
        "image_path": raw_diff,
        "object_id": row.get("object_id", ""),
        "coord_source": coord_source,
        "ra_used": ra,
        "dec_used": dec,
        "group_dir": str(group_dir),
        "diff_path": str(diff_dst),
        "sci_status": sci_status,
        "sci_path": str(sci_dst),
        "ref_status": ref_status,
        "ref_path": str(ref_dst) if ref_dst is not None else "",
        "ref_match_sep_arcsec": ref_sep,
        "status": status,
    }


def process_chunk(chunk_df: pd.DataFrame, *, chunk_idx: int, total_chunks: int, snr_csv: Path, diff_root: Path, out_root: Path, sci_index, ref_index, ref_meta, max_ref_sep_arcsec: float, download_missing: bool, max_retries: int, coord_index):
    pending = chunk_df.copy()
    collected = []
    attempt = 1

    while len(pending) > 0 and attempt <= max_retries:
        start = time.perf_counter()
        print(f"[chunk {chunk_idx}/{total_chunks}] attempt {attempt}/{max_retries} | rows={len(pending)}")

        next_pending = []
        for row_idx, row in pending.iterrows():
            try:
                rec = build_triplet_record(
                    row,
                    snr_csv=snr_csv,
                    diff_root=diff_root,
                    out_root=out_root,
                    sci_index=sci_index,
                    ref_index=ref_index,
                    ref_meta=ref_meta,
                    max_ref_sep_arcsec=max_ref_sep_arcsec,
                    download_missing=download_missing,
                    coord_index=coord_index,
                )
                collected.append(rec)
                print(
                    f"  [{chunk_idx}/{total_chunks} a{attempt}] {row_idx + 1}/{len(chunk_df)} "
                    f"{Path(str(row['image_path'])).name[:60]} | sci={rec['sci_status']} ref={rec['ref_status']} status={rec['status']}"
                )
            except Exception as e:
                next_pending.append((row_idx, row))
                print(
                    f"  [{chunk_idx}/{total_chunks} a{attempt}] FAILED {Path(str(row['image_path'])).name[:60]}: {e}"
                )

        elapsed = time.perf_counter() - start
        print(
            f"[chunk {chunk_idx}/{total_chunks}] attempt {attempt} done | succeeded={len(pending) - len(next_pending)} "
            f"failed={len(next_pending)} elapsed={fmt_seconds(elapsed)}"
        )

        if len(next_pending) == 0:
            pending = pending.iloc[0:0]
            break

        pending = pd.DataFrame([row for _, row in next_pending])
        attempt += 1

    if len(pending) > 0:
        print(f"[chunk {chunk_idx}/{total_chunks}] exhausted retries for {len(pending)} rows")
        for _, row in pending.iterrows():
            collected.append({
                "image_path": str(row.get("image_path", "")),
                "object_id": row.get("object_id", ""),
                "status": "failed_after_retries",
                "diff_path": "",
                "sci_status": "missing",
                "sci_path": "",
                "ref_status": "missing",
                "ref_path": "",
                "ref_match_sep_arcsec": np.nan,
            })

    return collected


def main():
    ap = argparse.ArgumentParser(description="Collect SCI/REF for SNR>=3 diff cutouts")
    ap.add_argument("--snr-csv", default="../images_unmasked_snr_gt3_all_images.csv")
    ap.add_argument("--diff-root", default="../snr_ge3_files")
    ap.add_argument("--sci-root", default="../ztf_sci_cutout_imgs")
    ap.add_argument("--ref-root", default="../ztf_ref_cutout_imgs")
    ap.add_argument("--ref-meta", default="../metadata_ref_best.csv")
    ap.add_argument("--out-root", default="../snr_ge3_triplets")
    ap.add_argument("--final-class-csv", default="../final_class_table_new_(only_in_dist_bounds).csv")
    ap.add_argument("--max-ref-sep-arcsec", type=float, default=3.0)
    ap.add_argument("--download-missing", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    snr_csv = resolve_cli_path(args.snr_csv)
    diff_root = resolve_cli_path(args.diff_root)
    sci_root = resolve_cli_path(args.sci_root)
    ref_root = resolve_cli_path(args.ref_root)
    ref_meta_path = resolve_cli_path(args.ref_meta)
    out_root = resolve_cli_path(args.out_root)
    final_class_csv = resolve_cli_path(args.final_class_csv)

    if not snr_csv.exists():
        raise FileNotFoundError(f"Missing SNR csv: {snr_csv}")
    if not ref_meta_path.exists():
        raise FileNotFoundError(f"Missing reference metadata: {ref_meta_path}")
    if not final_class_csv.exists():
        raise FileNotFoundError(f"Missing final class table: {final_class_csv}")

    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(snr_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    required = ["image_path", "object_id"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing required columns: {miss}")

    ref_meta = pd.read_csv(ref_meta_path, low_memory=False)
    final_class_df = pd.read_csv(final_class_csv, low_memory=False)
    coord_index = _build_exact_coord_index(final_class_df)

    print("Building local file index...")
    sci_index = build_file_index(sci_root)
    ref_index = build_file_index(ref_root)
    print(f"SCI indexed: {len(sci_index)}")
    print(f"REF indexed: {len(ref_index)}")
    print(f"Exact coord ids indexed: {len(coord_index)}")

    rows_out = []
    n_total = len(df)

    chunk_size = max(1, int(args.chunk_size))
    total_chunks = int(np.ceil(n_total / chunk_size)) if n_total > 0 else 0

    print(f"Total rows: {n_total}")
    print(f"Chunk size: {chunk_size}")
    print(f"Max retries per chunk: {args.max_retries}")
    print(f"Download missing: {args.download_missing}")

    started = time.perf_counter()
    for chunk_idx, start in enumerate(range(0, n_total, chunk_size), start=1):
        stop = min(start + chunk_size, n_total)
        chunk_df = df.iloc[start:stop].copy()
        chunk_started = time.perf_counter()
        print(f"\n[chunk {chunk_idx}/{total_chunks}] rows {start + 1}-{stop} / {n_total}")

        chunk_rows = process_chunk(
            chunk_df,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            snr_csv=snr_csv,
            diff_root=diff_root,
            out_root=out_root,
            sci_index=sci_index,
            ref_index=ref_index,
            ref_meta=ref_meta,
            max_ref_sep_arcsec=args.max_ref_sep_arcsec,
            download_missing=args.download_missing,
            max_retries=args.max_retries,
            coord_index=coord_index,
        )
        rows_out.extend(chunk_rows)

        completed_triplets = sum(
            1
            for rec in chunk_rows
            if rec.get("sci_status") in {"local", "downloaded"}
            and rec.get("ref_status") in {"local", "downloaded"}
        )
        elapsed = time.perf_counter() - chunk_started
        done = stop
        rate = done / max(1e-9, (time.perf_counter() - started))
        eta = (n_total - done) / max(rate, 1e-9)
        print(
            f"[chunk {chunk_idx}/{total_chunks}] finished | complete_triplets_in_chunk={completed_triplets}/{len(chunk_rows)} "
            f"chunk_time={fmt_seconds(elapsed)} eta={fmt_seconds(eta)}"
        )

    out_csv = out_root / "triplet_manifest.csv"
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)

    n_ok = sum(
        1
        for rec in rows_out
        if rec.get("sci_status") in {"local", "downloaded"}
        and rec.get("ref_status") in {"local", "downloaded"}
    )

    print("\nDone")
    print(f"Total rows: {n_total}")
    print(f"Complete triplets (diff+sci+ref): {n_ok}")
    print(f"Manifest: {out_csv}")


if __name__ == "__main__":
    main()
