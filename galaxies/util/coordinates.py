from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
import re


DEFAULT_RA_REGEX = re.compile(r"RA((J2000)|(_ICRS))?")
DEFAULT_DEC_REGEX = re.compile(r"DE((J2000)|(_ICRS))?")


def rename_coord_cols(tab: Table, new_ra_name: str = "ra",
                      new_dec_name: str = "dec",
                      ra_regex: str = DEFAULT_RA_REGEX,
                      dec_regex: str = DEFAULT_DEC_REGEX) -> Table:
    """
    Rename coordinate columns in the original table, while modifying it
    in-place.
    
    Parameters
    ----------
    tab: Table
        a table with coordinate columns
    new_ra_name: str
        a new name for RA column
    new_dec_name: str
        the same for Dec column
    ra_regex: str
        a regular expression that matches RA column
    dec_regex: str
        the same for Dec column
        
    Returns
    -------
    tab: Table
        the original table with renamed in-place columns
    """
    for col in tab.colnames:
        if re.fullmatch(DEFAULT_RA_REGEX, col):
            tab.rename_columns([col], [new_ra_name])
            break
            
    for col in tab.colnames:
        if re.fullmatch(DEFAULT_DEC_REGEX, col):
            tab.rename_columns([col], [new_dec_name])
            break
    
    return tab


def validate_query_region(radius: Angle | None = None,
                          width: Angle | None = None,
                          height: Angle | None = None) -> dict[str, Angle]:
    if radius is None and (width is None or height is None):
        raise ValueError("Either radius or pair of width, height must be specified")
        
    if radius is not None:
        query = {"radius": radius}
    else:
        w = width
        h = height
        if w is None and h is not None:
            w = h
        if h is None and w is not None:
            h = w
        
        query = {"width": w, "height": h}
                          
    return query