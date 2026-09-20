from astroquery.vizier import Vizier
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
import os.path
from util.coordinates import validate_query_region


def load_vizier_catalog(vizier_name: str,
                        center: SkyCoord,
                        radius: Angle | None = None, 
                        width: Angle | None = None, 
                        height: Angle | None = None,
                        path: str = "catalog.fit",
                        query: dict[str, str] | None = None) -> tuple[Table | None, str]:
    """
    Load a catalog of sources from the conical or rectangular search in Vizier, 
    and save it into the FITS file.
    
    Parameters
    ----------
    vizier_name: str
        the Vizier id of the catalog
    center: SkyCoord
        the center around to search for catalog sources
    radius: Angle | None
        the radius of searching region, default None
    width, height: Angle | None
        the width and the height of the searching region
    path: str
        the path to where to save the catalog
    
    Returns
    -------
    cat, path: Table | None, str
        2-tuple of the catalog table or None if not found, and path to the catalog file
    """
    
    cat = None
    if os.path.exists(path) and os.path.isfile(path):
        cat = Table.read(path, format="fits")
    else:
        cats = Vizier(catalog=vizier_name, row_limit=-1).query_region(center, radius=radius, 
                                                                      width=width, height=height)
        if len(cats):
            cat = cats[0]
            cat.write(path, format="fits", overwrite=True)

    return cat, path
    
    
def load_sdss(**kwargs) -> tuple[Table | None, str]:
    path = "sdss_unique.fit" if os.path.exists("sdss_unique.fit") else "sdss_dr16.fit"
    kwargs.update(vizier_name="V/154/sdss16", path=path)
    sdss, sdss_path = load_vizier_catalog(**kwargs)
    return sdss, sdss_path


def load_gaia(**kwargs) -> tuple[Table | None, str]:
    path = "gaia_dr3.fit"
    kwargs.update(vizier_name="I/355/gaiadr3", path=path)
    gaia, gaia_path = load_vizier_catalog(**kwargs)
    return gaia, gaia_path


def load_desi_south(**kwargs) -> tuple[Table | None, str]:
    path = "desi_south.fit"
    kwargs.update(vizier_name="VII/292/south", path=path)
    desi, desi_path = load_vizier_catalog(**kwargs)
    return desi, desi_path


def load_gladep(**kwargs) -> tuple[Table | None, str]:
    path = "gladep.fit"
    kwargs.update(vizier_name="VII/291/gladep", path=path)
    gladep, gladep_path = load_vizier_catalog(**kwargs)
    return gladep, gladep_path


def load_catwise(**kwargs) -> tuple[Table | None, str]:
    path = "catwise.fit"
    kwargs.update(vizier_name="II/365/catwise", path=path)
    catwise, catwise_path = load_vizier_catalog(**kwargs)
    return catwise, catwise_path


def load_ned_lvs(path: str, center: SkyCoord,
                 radius: Angle | None = None, 
                 width: Angle | None = None, 
                 height: Angle | None = None) -> Table:
    query = validate_query_region(radius, width, height)
    tab = Table.read(path, format="fits")
    ra0 = center.ra.deg
    dec0 = center.dec.deg
    r = query.get("radius", None)
    w = query.get("width", None)
    h = query.get("height", None)
    ra = tab["ra"]
    dec = tab["dec"]
    
    if r is not None:
        coord = SkyCoord(ra, dec, unit=["deg"]*2)
        mask = center.separation(coord) < r
    else:
        w = w.to_value("deg")
        h = h.to_value("deg")
        mask = (ra < ra0 + w/2) & (ra > ra0 - w/2) & (dec < dec0 + h/2) & (dec > dec0 - h/2)
        
    return tab[mask]    