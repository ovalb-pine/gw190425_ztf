from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np


def sdss_grouper(gr: Table) -> Table:
    if len(gr) == 0:
        raise ValueError("Empty SDSS group received.")
    if len(gr) == 1:
        return gr[:1].copy()

    gr = gr.copy()

    # Prefer the cleanest row, then highest Q
    clean = np.asarray(gr["clean"])
    q = np.asarray(gr["Q"])

    max_clean = np.nanmax(clean)
    cand = np.flatnonzero(clean == max_clean)

    if len(cand) == 1:
        best = cand[0]
    else:
        best = cand[np.nanargmax(q[cand])]

    return gr[best:best + 1].copy()


def remove_duplicates(
    cat: Table,
    ra_col: str = "ra",
    dec_col: str = "dec",
    match_radius: u.Quantity = 3.0 * u.arcsec,
    grouper: callable = lambda x: x[0],
) -> Table:
    ra = cat[ra_col]
    dec = cat[dec_col]
    coord = SkyCoord(ra, dec, unit="deg")

    idx = np.arange(len(coord))
    groups = []

    print("Grouping duplicated sources ...")
    while len(idx) > 0:
        i = idx[0]
        idx = idx[1:]

        sep = coord[i].separation(coord[idx])
        match = sep < match_radius

        if np.count_nonzero(match):
            group = np.concatenate((np.atleast_1d(i), idx[match]))
            idx = idx[~match]
        else:
            group = np.atleast_1d(i)

        groups.append(group)

    print("Groups of duplicated sources were created.\n"
          "Stacking groups in a single table ...")

    def add_group(i, gr_idx):
        if len(gr_idx) > 1:
            groups[i] = grouper(cat[gr_idx])
        else:
            groups[i] = cat[gr_idx]

    for i, gr_idx in enumerate(groups):
        add_group(i, gr_idx)

    out = vstack(groups)
    print("Done")
    return out


def remove_duplicates_sdss(sdss: Table, match_radius: u.Quantity = 1.0 * u.arcsec) -> Table:
    """
    Deduplicate SDSS rows by sky position, keeping the best row from each group.
    The original objID of the chosen best row is preserved.
    """
    if len(sdss) == 0:
        return sdss.copy()

    # Deduplicate – sdss_grouper already selects the best row and keeps its objID
    dedup = remove_duplicates(
        sdss,
        ra_col="RA_ICRS",
        dec_col="DE_ICRS",
        grouper=sdss_grouper,
    )

    # No extra reassignment – the objID in dedup is already correct
    return dedup