import argparse
import logging
import numpy as np
import sys

from lsst.afw.table import SourceCatalog
from lsst.daf.persistence import Butler
from lsst.geom import SpherePoint, degrees
from modelling_research.multiprofit_task import MultiProFitConfig, MultiProFitTask
from multiprofit.utils import str2bool


def get_patch_tract(ra, dec, butler):
    """Get the patch and tract that contains a given sky coordinate.

    Parameters
    ----------
    ra : `float`
        Right ascenscion in degrees.
    dec : `float`
        Declination in degrees.
    butler : `lsst.daf.persistence.Butler`
        A Generation 2 butler.

    Returns
    -------
    tract : `int`
        The tract ID.
    patch : `tuple` [`int`, `int`]
        The patch IDs containing the coordinates.
    """
    skymap = butler.get("deepCoadd_skyMap")
    spherePoint = SpherePoint(ra, dec, degrees)
    tract = skymap.findTract(spherePoint).getId()
    return tract, skymap[tract].findPatch(spherePoint)


def get_data(butler, tract, name_patch, cat_type=None, exposure_type=None, filters=None, sources=None):
    """Get the necessary data to run the MultiProFit task on a range of sources in a region.

    Parameters
    ----------
    butler : `lsst.daf.persistence.Butler`
        A Generation 2 butler.
    tract : `int`
        A tract number.
    name_patch : `str`
        The name of the patch in the tract to process.
    cat_type : `str`
        The type of catalog to retrieve from the butler. Default "deepCoadd_meas".
    exposure_type : `str`
        The type of exposure to retrieve from the butler. Default "deepCoadd_calexp".
    filters : iterable of `str`
        Names of bandpass filters for filter-dependent fields. Default ["HSC-I", "HSC-R", "HSC-G"].
    sources : `lsst.afw.table.SourceCatalog`
        A catalog containing deblended sources with footprints. Default butler.get("deepCoadd_meas", dataId).

    Returns
    -------
    data : `dict` [`str`, `dict` [`str`]]
        A dict of dicts keyed by filter name, each containing:
        ``"exposures"``
            The exposure of that filter (`lsst.afw.image.Exposure`)
        ``"sources"``
            The catalog of sources to fit (`lsst.afw.table.SourceCatalog`)

    """
    if filters is None:
        filters = ["HSC-I", "HSC-R", "HSC-G"]
    if exposure_type is None:
        exposure_type = "deepCoadd_calexp"
    has_sources = sources is not None
    if not has_sources and cat_type is None:
        cat_type = "deepCoadd_meas"

    dataId = {"tract": tract, "patch": name_patch}
    data = {}
    for i, band in enumerate(filters):
        data[band] = {
            'exposure': butler.get(exposure_type, dataId, filter=band),
            'sources': sources[i] if has_sources else butler.get(cat_type, dataId, filter=band),
        }
    return data


def get_flags():
    """ Return all valid MultiProFit Task flags with annotations.

    Returns
    -------
    flags : `dict` [`dict`]
        A dict of flag specifications (as dicts) by name.
    """
    # TODO: Figure out if there's a way to get help from docstrings (defaults can be read easily)
    flags = {
        'repo': dict(type=str, nargs='?', default="/datasets/hsc/repo/rerun/RC/w_2019_38/DM-21386/",
                     help="Path to Butler repository to read from"),
        'filenameIn': dict(type=str, nargs='?', default=None, help="Input catalog filename"),
        'filenameOut': dict(type=str, nargs='?', default=None, help="Output catalog filename", kwarg=True),
        'radec': dict(type=float, nargs=2, default=None, help="RA/dec coordinate of source"),
        'name_patch': dict(type=str, nargs='?', default="4,4", help="Butler patch string"),
        'tract': dict(type=int, nargs='?', default=9813, help="Butler tract ID"),
        'filters': dict(type=str, nargs='*', default=['HSC-I'], help="List of bandpass filters"),
        'idx_begin': dict(type=int, nargs='?', default=0, help="Initial row index to fit"),
        'idx_end': dict(type=int, nargs='?', default=np.Inf, help="Final row index to fit"),
        'img_multi_plot_max': dict(type=float, nargs='?', default=None,
                                   help="Max value for colour images in plots"),
        'weights_band': dict(type=float, nargs='*', default=None,
                             help="Weights per filter to rescale cdimages in multi-band plots"),
        'path_cosmos_galsim': dict(type=str, nargs='?',
                                   default="/project/dtaranu/cosmos/hst/COSMOS_25.2_training_sample",
                                   help="Path to GalSim COSMOS catalogs"),
        'plot': dict(action='store_true', default=False, help="Plot each source fit"),
        'printTrace': dict(action='store_true', default=False, help="Print traceback for errors"),
        'loglevel': {'type': int, 'nargs': '?', 'default': 21, 'help': 'logging.Logger default level'},
    }
    for param, field in MultiProFitConfig._fields.items():
        type_default = field.dtype
        flag = dict(type=type_default if type_default is not bool else str2bool, nargs='?',
                    default=field.default,
                    help=f'Value for MultiProFitConfig.{param}', kwarg=True)
        flags[param] = flag
    return flags


def main():
    parser = argparse.ArgumentParser(description='MultiProFit Butler Task running test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    flags = get_flags()
    kwargs = set()

    for key, value in flags.items():
        if 'kwarg' in value:
            kwargs.add(key)
            del value['kwarg']
        parser.add_argument(f'--{key}', **value)
    try:
        args = parser.parse_args()
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(status=1)
    logging.basicConfig(stream=sys.stdout, level=args.loglevel)
    butler = Butler(args.repo)
    sources = SourceCatalog.readFits(args.filenameIn) if args.filenameIn is not None else None

    if args.name_patch is not None and args.tract is not None:
        name_patch = args.name_patch
    else:
        ra, dec = args.radec
        tract, patch = get_patch_tract(ra, dec, butler)
        name_patch = ','.join([str(x) for x in patch.getIndex()])

    argsvars = vars(args)
    kwargs = {key: argsvars[key] for key in kwargs}
    config = MultiProFitTask.ConfigClass(**kwargs)
    task = MultiProFitTask(config=config)
    data = get_data(butler, args.tract, name_patch=name_patch, sources=sources, filters=args.filters)
    catalog, results = task.fit(data, idx_begin=args.idx_begin, idx_end=args.idx_end,
                                printTrace=args.printTrace, plot=args.plot,
                                img_multi_plot_max=args.img_multi_plot_max, weights_band=args.weights_band,
                                path_cosmos_galsim=args.path_cosmos_galsim)
    return catalog, results


if __name__ == '__main__':
    main()
