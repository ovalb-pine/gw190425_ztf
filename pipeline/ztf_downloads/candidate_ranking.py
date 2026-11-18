"""Catalog tagging (star / galaxy / no match) and final A/B/C ranking of candidates.

Two independent steps for the blind full-image search:

1. ``tag_objects_with_catalogs``  (right after the stage-1 SNR measurement)
   Nearest object within ``radius_arcsec`` over the galaxy catalog
   (final_class_*.csv) and the star catalog (star_catalog.csv) decides the tag:
   ``star_from_catalog`` / ``galaxy_from_catalog`` / ``no_catalog_match``.
   Nothing is rejected here; every candidate continues to the next stage.

2. ``build_ranking_table``  (after stage 5, the minor-planet check)
   Points per candidate -> category A / B / C.  Asteroids are dropped; stars stay
   but get a penalty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

TAG_STAR = "star_from_catalog"
TAG_GALAXY = "galaxy_from_catalog"
TAG_NONE = "no_catalog_match"

_RA_COLS = ("ra_fin", "RA_fin", "ra_SDSS-DR16", "RA", "ra")
_DEC_COLS = ("dec_fin", "DEC_fin", "dec_SDSS-DR16", "DEC", "dec")


# --------------------------------------------------------------------------- #
# 1. Catalog tagging
# --------------------------------------------------------------------------- #
def _catalog_skycoord(cat: pd.DataFrame | None) -> SkyCoord | None:
    if cat is None or cat.empty:
        return None
    ra_col = next((c for c in _RA_COLS if c in cat.columns), None)
    dec_col = next((c for c in _DEC_COLS if c in cat.columns), None)
    if ra_col is None or dec_col is None:
        raise KeyError(f"No RA/Dec columns found in catalog (looked for {_RA_COLS} / {_DEC_COLS})")
    ra = pd.to_numeric(cat[ra_col], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(cat[dec_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if not ok.any():
        return None
    return SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg)


def _nearest_sep_arcsec(coords: SkyCoord, cat_coords: SkyCoord | None) -> np.ndarray:
    if cat_coords is None:
        return np.full(len(coords), np.inf)
    _, sep, _ = coords.match_to_catalog_sky(cat_coords)
    return sep.arcsec


def tag_objects_with_catalogs(
    objects: pd.DataFrame,
    galaxy_cat: pd.DataFrame | None,
    star_cat: pd.DataFrame | None,
    *,
    radius_arcsec: float = 10.0,
) -> pd.DataFrame:
    """Return ``objects`` with ``catalog_tag`` and ``catalog_sep_arcsec`` columns.

    The nearest catalog object within ``radius_arcsec`` decides the tag.  If a
    star and a galaxy are at exactly the same distance the galaxy wins.
    """
    out = objects.copy()
    if out.empty:
        out["catalog_tag"] = pd.Series(dtype="object")
        out["catalog_sep_arcsec"] = pd.Series(dtype="float")
        return out

    coords = SkyCoord(
        pd.to_numeric(out["ra"], errors="coerce").to_numpy(float) * u.deg,
        pd.to_numeric(out["dec"], errors="coerce").to_numpy(float) * u.deg,
    )
    sep_gal = _nearest_sep_arcsec(coords, _catalog_skycoord(galaxy_cat))
    sep_star = _nearest_sep_arcsec(coords, _catalog_skycoord(star_cat))

    in_gal = sep_gal <= radius_arcsec
    in_star = sep_star <= radius_arcsec
    star_wins = in_star & (~in_gal | (sep_star < sep_gal))
    gal_wins = in_gal & ~star_wins

    tag = np.full(len(out), TAG_NONE, dtype=object)
    sep = np.full(len(out), np.nan)
    tag[star_wins], sep[star_wins] = TAG_STAR, sep_star[star_wins]
    tag[gal_wins], sep[gal_wins] = TAG_GALAXY, sep_gal[gal_wins]
    out["catalog_tag"] = tag
    out["catalog_sep_arcsec"] = sep
    return out


# --------------------------------------------------------------------------- #
# 2. Scoring and hierarchy
# --------------------------------------------------------------------------- #
def score_candidate(braai: float, snr: float, dt_days: float, n_det: int, tag: str, *, star_penalty: float = -5.0) -> int:
    """Sum of points (see module docstring); NaN inputs earn no points."""
    pts = 0
    if np.isfinite(braai):
        if braai >= 0.8:
            pts += 2
        elif braai >= 0.5:
            pts += 1
    if np.isfinite(snr):
        if snr >= 4.0:
            pts += 3
        elif snr >= 3.5:
            pts += 2
        elif snr >= 3.0:
            pts += 1
    if np.isfinite(dt_days) and dt_days <= 3.0:
        pts += 3
    if n_det >= 2:
        pts += 3
    if tag == TAG_STAR:
        pts += star_penalty
    return int(pts)


def category_from_score(score: float) -> str:
    if score >= 8:
        return "A"
    if score >= 4:
        return "B"
    if score >= 0:
        return "C"
    return "C-"  # negative sum (in practice: penalised stars)


def flag_asteroids(mp_df: pd.DataFrame, *, match_radius_arcsec: float = 10.0) -> pd.Series:
    """Boolean Series (indexed like ``mp_df``): True where a known minor planet is
    predicted within max(match_radius_arcsec, its position uncertainty) of the
    candidate, using the stage-5 SkyBoT summary columns."""
    sep = pd.to_numeric(mp_df.get("minor_planet_nearest_sep_arcsec"), errors="coerce")
    unc = pd.to_numeric(mp_df.get("minor_planet_nearest_posunc_arcsec"), errors="coerce").fillna(0.0)
    limit = np.maximum(float(match_radius_arcsec), unc)
    return (sep <= limit).fillna(False)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def build_ranking_table(
    final_candidates: pd.DataFrame,
    *,
    stage1_objects: pd.DataFrame,
    braai_scores: pd.DataFrame,
    stage3_images: pd.DataFrame,
    catalog_tags: pd.DataFrame,
    minor_planets: pd.DataFrame | None,
    trigger_datetime,
    parse_filefracday,
    normalize_id=lambda x: str(x).strip(),
    star_penalty: float = -5.0,
    asteroid_match_radius_arcsec: float = 10.0,
) -> pd.DataFrame:
    """One row per stage-4 final candidate with score and category.

    Per-object inputs
      braai   : best BRAAI score over the object's detections
      snr     : best matched-filter ``snr_empirical`` (stage-1 object table)
      dt_days : first post-trigger detection time - trigger time (days)
      n_det   : number of distinct post-trigger *exposures* (filefracday) with a
                detection, so one source in two overlapping quadrants counts once
    Asteroids (stage 5) are removed from the output.
    """
    if final_candidates.empty:
        return final_candidates.assign(score=pd.Series(dtype=int), category=pd.Series(dtype="object"))

    key = lambda s: s.astype(str).map(normalize_id)  # noqa: E731
    table = final_candidates.copy()
    table["object_id"] = key(table["object_id"])

    # --- asteroids ------------------------------------------------------------
    table["is_asteroid"] = False
    if minor_planets is not None and not minor_planets.empty and "object_id" in minor_planets.columns:
        mp = minor_planets.copy()
        mp["object_id"] = key(mp["object_id"])
        mp["_ast"] = flag_asteroids(mp, match_radius_arcsec=asteroid_match_radius_arcsec)
        table["is_asteroid"] = table["object_id"].map(mp.drop_duplicates("object_id").set_index("object_id")["_ast"]).fillna(False).astype(bool)

    # --- best BRAAI per object ------------------------------------------------
    b = braai_scores[["object_id", "braai_score"]].copy()
    b["object_id"] = key(b["object_id"])
    b["braai_score"] = pd.to_numeric(b["braai_score"], errors="coerce")
    table["braai_best"] = table["object_id"].map(b.groupby("object_id")["braai_score"].max())

    # --- best SNR -------------------------------------------------------------
    s = stage1_objects[["object_id", "best_snr_empirical"]].copy()
    s["object_id"] = key(s["object_id"])
    table["snr_best"] = table["object_id"].map(s.set_index("object_id")["best_snr_empirical"].astype(float))

    # --- post-trigger detections: first time and number of exposures ----------
    img = stage3_images.copy()
    img["object_id"] = key(img["object_id"])
    post = img[(img.get("phase", "").astype(str) == "posttrigger") & _truthy(img["detection"])].copy()
    post["_t"] = post["filefracday"].map(parse_filefracday)
    post = post.dropna(subset=["_t"])
    first_t = post.groupby("object_id")["_t"].min()
    n_epochs = post.groupby("object_id")["filefracday"].nunique()
    table["first_detection_utc"] = table["object_id"].map(first_t)
    table["dt_days"] = (pd.to_datetime(table["first_detection_utc"]) - pd.Timestamp(trigger_datetime)).dt.total_seconds() / 86400.0
    table["n_detections"] = table["object_id"].map(n_epochs).fillna(0).astype(int)

    # --- catalog tag ------------------------------------------------------------
    tags = catalog_tags[["object_id", "catalog_tag", "catalog_sep_arcsec"]].copy()
    tags["object_id"] = key(tags["object_id"])
    tags = tags.drop_duplicates("object_id").set_index("object_id")
    table["catalog_tag"] = table["object_id"].map(tags["catalog_tag"]).fillna(TAG_NONE)
    table["catalog_sep_arcsec"] = table["object_id"].map(tags["catalog_sep_arcsec"])

    # --- score ------------------------------------------------------------------
    table["score"] = [
        score_candidate(br, sn, dt, nd, tg, star_penalty=star_penalty)
        for br, sn, dt, nd, tg in zip(
            table["braai_best"].astype(float),
            table["snr_best"].astype(float),
            table["dt_days"].astype(float),
            table["n_detections"],
            table["catalog_tag"],
        )
    ]
    table["category"] = table["score"].map(category_from_score)

    table = table[~table["is_asteroid"]].drop(columns="is_asteroid")
    return table.sort_values(["score", "snr_best"], ascending=[False, False]).reset_index(drop=True)
