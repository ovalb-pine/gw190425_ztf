import re
import urllib.parse
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import _load_json, _save_json
import math
import time

"""
Science Exposures File Path Pattern:
'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/'+year+'/'+month+day+'/
'+fracday+'/ztf_'+filefracday+'_'+paddedfield+'_'+filtercode+'_c'+paddedccdid+'_'+imgtypecode+'_q'+qid+'_'+suffix

Source: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_metadata.html
"""

# base URL for ZTF data download
DATA_ROOT = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products"

_SESSION = None


def get_session():
    """Return a shared requests session with connection pooling."""
    global _SESSION
    if _SESSION is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _SESSION = session
    return _SESSION

def build_sci_url(row, suffix="sciimg.fits"):
    """Return a direct URL for a 'sci' product row (pandas Series or dict)."""
    filefracday = str(row["filefracday"])
    year = filefracday[0:4]
    month = filefracday[4:6]
    day = filefracday[6:8]
    fracday = filefracday[8:14]
    paddedfield = str(row.get("paddedfield", str(int(row.get("field"))).zfill(6))).zfill(6)
    filtercode = str(row["filtercode"])
    paddedccdid = str(row.get("paddedccdid", row.get("ccdid"))).zfill(2)
    imgtypecode = str(row.get("imgtypecode", row.get("imgtype", "o")))
    qid = str(int(row["qid"]))
    filename = f"ztf_{filefracday}_{paddedfield}_{filtercode}_c{paddedccdid}_{imgtypecode}_q{qid}_{suffix}"
    return f"{DATA_ROOT}/sci/{year}/{month}{day}/{fracday}/{filename}"

def build_ref_url(row):
    """Return a direct URL for a 'ref' product row (pandas Series or dict)."""
    filefracday = str(row["filefracday"])
    year = filefracday[0:4]
    month = filefracday[4:6]
    day = filefracday[6:8]
    fracday = filefracday[8:14]
    paddedfield = str(row.get("paddedfield", str(int(row.get("field"))).zfill(6))).zfill(6)
    prefield = paddedfield[:3]
    filtercode = str(row["filtercode"])
    paddedccdid = str(row.get("paddedccdid", row.get("ccdid"))).zfill(2)
    imgtypecode = str(row.get("imgtypecode", row.get("imgtype", "o")))
    qid = str(int(row["qid"]))
    filename = f"ztf_{paddedfield}_{filtercode}_c{paddedccdid}_q{qid}_refimg.fits"
    return f"{DATA_ROOT}/ref/{prefield}/field{paddedfield}/{filtercode}/ccd{paddedccdid}/q{qid}/{filename}"

def build_cutout_url(file_url, ra, dec, size_arcsec=240, gzip=False):
    sep = "&" if "?" in file_url else "?"
    size_str = f"{int(size_arcsec)}arcsec"
    gz = "false" if not gzip else "true"
    return f"{file_url}{sep}center={ra},{dec}&size={size_str}&gzip={gz}"

def _safe_label(label):
    # make a short filesystem-safe label: letters, digits, dash, underscore, dot
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(label))

def download_file(url, out_dir="ztf_files", timeout=300, overwrite=False, label=None, filtercode=None):
    """Stream-download a URL to out_dir; returns pathlib.Path or None on 404.
    If label is provided, the file is saved as {label}{ext} (ext taken from URL).
    """
    out_dir = Path(out_dir)
    if filtercode:
        out_dir = out_dir / str(filtercode)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract original filename and extension
    original_fname = Path(urllib.parse.urlparse(url).path).name
    if original_fname.endswith('.fits.fz'):
        ext = '.fits.fz'
    else:
        ext = Path(original_fname).suffix

    if label:
        label_str = str(label)
        label_l = label_str.lower()
        if label_l.endswith('.fits.fz') or label_l.endswith('.fits') or label_l.endswith('.fz'):
            fname = _safe_label(label_str)
        else:
            fname = f"{_safe_label(label_str)}{ext}"
    else:
        fname = original_fname

    out_path = out_dir / fname

    if out_path.exists() and not overwrite:
        return out_path

    session = get_session()
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*64):
                    if chunk:
                        f.write(chunk)
        return out_path
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            # File not found – skip silently
            print(f"Skipping 404: {url}")
            return None
        raise

def batch_download(urls, out_dir="ztf_files", max_workers=6, max_retries=3, retry_delay=5, **dl_kwargs):
    """
    Parallel download with per‑file retries.
    Skips files that return 404 (None result) without raising an error.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    failed = []
    skipped = []
    i = 1

    def download_with_retry(url, out_dir, label, filtercode):
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                result = download_file(url, out_dir, label=label, filtercode=filtercode, **dl_kwargs)
                return result  # can be None for 404
            except Exception as e:
                last_exc = e
                if attempt == max_retries:
                    raise
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"Retry {attempt}/{max_retries} for {url} in {wait}s")
                time.sleep(wait)
        raise last_exc

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for item in urls:
            if isinstance(item, (tuple, list)):
                item_list = list(item) if isinstance(item, tuple) else item
                url, label, filtercode = (item_list + [None, None, None])[:3]
            else:
                url, label, filtercode = item, None, None
            fut = ex.submit(download_with_retry, url, out_dir, label, filtercode)
            futures[fut] = (url, label, filtercode)

        for fut in as_completed(futures):
            url, label, filt = futures[fut]
            try:
                p = fut.result()
                if p is not None:
                    downloaded.append(str(p))
                    print(i, "Downloaded:", p.name)
                else:
                    skipped.append(url)
                    print(f"Skipped (404): {url}")
                i += 1
            except Exception as e:
                print("Failed:", url, e)
                failed.append((url, str(e)))

    if failed:
        raise RuntimeError(f"{len(failed)} downloads failed in this chunk: {failed[:3]}{'...' if len(failed)>3 else ''}")

    print(f"Finished: {len(downloaded)} downloaded, {len(skipped)} skipped (404).")
    return downloaded

def batch_download_resumable(
    items,
    *,
    out_dir,
    progress_json,
    max_workers=9,
    chunk_size=250,
    max_retries=5,
    retry_delay=5,
):
    """
    Resumable batch download with checkpoint-based recovery and automatic retries.
    Skips 404 errors per file; chunk only fails on other HTTP errors.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_json = Path(progress_json)

    state = _load_json(progress_json, {"completed_chunks": []})
    completed = set(state.get("completed_chunks", []))
    total_chunks = math.ceil(len(items) / chunk_size)

    for chunk_idx in range(total_chunks):
        if chunk_idx in completed:
            continue

        i0 = chunk_idx * chunk_size
        i1 = min((chunk_idx + 1) * chunk_size, len(items))
        chunk = items[i0:i1]

        for attempt in range(1, max_retries + 1):
            try:
                print(f"Download chunk {chunk_idx + 1}/{total_chunks} ({i0}:{i1}) attempt {attempt}")
                batch_download(chunk, out_dir=str(out_dir), max_workers=max_workers)
                break  # success
            except Exception as e:
                if attempt == max_retries:
                    print(f"Chunk failed after {max_retries} attempts. Saving progress.")
                    state["last_error"] = str(e)
                    state["last_failed_chunk"] = chunk_idx
                    state["completed_chunks"] = sorted(completed)
                    state["total_chunks"] = total_chunks
                    _save_json(progress_json, state)
                    raise
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"Retrying chunk in {wait}s due to: {e}")
                time.sleep(wait)

        completed.add(chunk_idx)
        state["completed_chunks"] = sorted(completed)
        state["total_chunks"] = total_chunks
        state["out_dir"] = str(out_dir)
        _save_json(progress_json, state)

    print(f"All chunks complete: {len(items)} items")