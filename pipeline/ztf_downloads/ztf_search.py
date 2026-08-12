import requests
import pandas as pd
from io import StringIO
from typing import List, Sequence,Tuple, Optional
import math
from pathlib import Path
from .utils import _load_json, _save_json
import time

# base URL for ZTF data search
# source: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html
SEARCH_BASE = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products"


def metadata_search_single(ra: float, dec: float, size_deg: float=0.01, 
                    product_type: str= "sci", ct: str="csv", date_start: str=None, date_end: str=None, 
                    extra_params: dict=None, timeout: int=60) -> pd.DataFrame:
    """
    Search for a single position and return metadata as a pandas DataFrame.
    ra, dec: position in degrees
    size_deg: search radius in degrees (default 0.01)
    product_type: type of product to search for (default "sci")
    ct: output format (default "csv")
    date_start, date_end: date range for filtering results
    extra_params: additional parameters for the API request
    timeout: request timeout in seconds (default 60)
    """

    assert product_type in ("sci","ref","raw","cal","deep"), "unknown product_type"
    url = f"{SEARCH_BASE}/{product_type}"

    params = {
    "POS": f"{ra},{dec}",
    "SIZE": size_deg,
    "WHERE": f"obsdate>'{date_start}' AND obsdate<'{date_end}'" if date_start and date_end else None,
    "CT": "csv"
    }
    
    r = requests.get(url, params=params)
    r.raise_for_status()

    if ct.lower() == "csv":
        return pd.read_csv(StringIO(r.text))
    else:
        return r.text


def build_ipac_table(positions: List[Tuple[float, float]], float_fmt="{:0.8f}") -> bytes:
    """
    Build a minimal IPAC ASCII table bytes for two columns: ra, dec (both double).
    positions: list of (ra, dec) tuples in degrees
    float_fmt: format string for floating-point values (default 8 decimal places)
    """

    # sanitize coords
    clean = []
    for p in positions:
        try:
            ra = float(p[0]); dec = float(p[1])
            if math.isfinite(ra) and math.isfinite(dec):
                clean.append((ra, dec))
        except Exception:
            continue
    if not clean:
        raise ValueError("No valid positions supplied")

    # format strings
    ra_strs = [float_fmt.format(ra) for ra, _ in clean]
    dec_strs = [float_fmt.format(dec) for _, dec in clean]

    # column meta
    col_names = ["ra", "dec"]
    col_types = ["double", "double"]
    col_units = ["deg", "deg"]         # optional
    col_nulls = ["null", "null"]      # optional

    # compute widths so nothing falls under '|'
    widths = [
        max(len(col_names[0]), len(col_types[0]), len(col_units[0]), max(len(s) for s in ra_strs)),
        max(len(col_names[1]), len(col_types[1]), len(col_units[1]), max(len(s) for s in dec_strs))
    ]

    # build header lines (each must start with '|')
    def cell_left(text, w): return " " + text.ljust(w) + " "
    header_names = "|" + "|".join(cell_left(col_names[i], widths[i]) for i in range(2)) + "|"
    header_types = "|" + "|".join(cell_left(col_types[i], widths[i]) for i in range(2)) + "|"
    header_units = "|" + "|".join(cell_left(col_units[i], widths[i]) for i in range(2)) + "|"
    header_nulls = "|" + "|".join(cell_left(col_nulls[i], widths[i]) for i in range(2)) + "|"

    # build data rows: right-align numeric values so decimals line up
    def cell_right(text, w): return " " + text.rjust(w) + " "
    data_lines = []
    for i in range(len(clean)):
        rcell = cell_right(ra_strs[i], widths[0])
        dcell = cell_right(dec_strs[i], widths[1])
        data_lines.append(rcell + " " + dcell)

    # optional fixlen keyword
    keyword = "\\fixlen = T\n"

    txt = keyword + header_names + "\n" + header_types + "\n" + header_units + "\n" + header_nulls + "\n" + "\n".join(data_lines) + "\n"
    return txt.encode("utf-8")


def metadata_search_batch(positions: list, size_deg: float=0.01, intersect="CENTER",
                          product_type: str= "sci", ct: str="csv", date_start: str=None, 
                          date_end: str=None, filtercodes: Optional[Sequence[str]] = None,
                          extra_params: dict=None, timeout: int=120) -> pd.DataFrame:
    """
    Search for multiple positions and return combined metadata as a pandas DataFrame.
    positions: list of (ra, dec) tuples in degrees
    size_deg: search radius in degrees (default 0.01)
    intersect: "CENTER" (default), "COVER", or "ENCLOSE
    product_type: type of product to search for (default "sci")
    ct: output format (default "csv")
    date_start, date_end: date range for filtering results
    extra_params: additional parameters for the API request
    timeout: request timeout in seconds (default 120)
    """

    assert product_type in ("sci","ref","raw","cal","deep"), "unknown product_type"
    url = f"{SEARCH_BASE}/{product_type}"

    # build an IPAC ASCII table
    table_bytes = build_ipac_table(positions)

    # files param: name "POS" with the file contents
    files = {"POS": ("positions.tbl", table_bytes, "text/plain; charset=utf-8")}
    data = {"INTERSECT": intersect, "CT": ct}
    if date_start and date_end:
        data["WHERE"] = f"obsdate>'{date_start}' AND obsdate<'{date_end}'"
        
    r = requests.post(url, data=data, files=files, timeout=timeout)
    r.raise_for_status()

    # df = pd.read_csv(StringIO(r.text))
    # df.to_csv("metadata_search_batch.csv", index=False)
    if ct.lower() == "csv":
        return pd.read_csv(StringIO(r.text))
    else:
        return r.text


def metadata_search_resumable(
    positions,
    output_csv,
    progress_json,
    *,
    batch_size=100,
    dedup_subset=None,
    max_retries=5,
    retry_delay=5,
    **search_kwargs,
):
    """
    Resumable batch metadata search with checkpoint-based recovery and automatic retries.

    Args:
        positions: List of (ra, dec) tuples
        output_csv: Path to save accumulated metadata
        progress_json: Path to save progress state
        batch_size: Number of positions per batch (default 100)
        dedup_subset: Columns to deduplicate on (optional)
        max_retries: Number of retry attempts for a failed batch (default 5)
        retry_delay: Initial delay in seconds before retry (exponential backoff)
        **search_kwargs: Passed to metadata_search_batch()
    """
    output_csv = Path(output_csv)
    progress_json = Path(progress_json)

    state = _load_json(progress_json, {"completed_batches": []})
    completed = set(state.get("completed_batches", []))
    total_batches = math.ceil(len(positions) / batch_size)

    for batch_idx in range(total_batches):
        if batch_idx in completed:
            continue

        i0 = batch_idx * batch_size
        i1 = min((batch_idx + 1) * batch_size, len(positions))
        batch_positions = positions[i0:i1]

        # Retry logic
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Metadata batch {batch_idx + 1}/{total_batches} ({i0}:{i1}) attempt {attempt}")
                df_batch = metadata_search_batch(positions=batch_positions, **search_kwargs)
                if df_batch is None:
                    df_batch = pd.DataFrame()
                break  # success
            except requests.RequestException as e:
                if attempt == max_retries:
                    print(f"Failed after {max_retries} attempts: {e}")
                    raise
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"Retrying in {wait}s due to: {e}")
                time.sleep(wait)

        # Save batch result
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_csv.exists()
        df_batch.to_csv(output_csv, mode="a", header=write_header, index=False)

        completed.add(batch_idx)
        state["completed_batches"] = sorted(completed)
        state["total_batches"] = total_batches
        state["output_csv"] = str(output_csv)
        state["rows_last_batch"] = int(len(df_batch))
        _save_json(progress_json, state)

    if output_csv.exists():
        df_all = pd.read_csv(output_csv)
    else:
        df_all = pd.DataFrame()

    if dedup_subset and len(df_all) > 0:
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
        if len(df_all) != before:
            df_all.to_csv(output_csv, index=False)
            print(f"Deduplicated metadata: {before} -> {len(df_all)}")

    return df_all