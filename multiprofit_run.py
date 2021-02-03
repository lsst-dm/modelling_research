import argparse
import logging
import sys

from lsst.daf.persistence import Butler
from lsst.geom import SpherePoint, degrees
from modelling_research.multiprofit_task import MultiProFitConfig, MultiProFitTask
import modelling_research.fit_multiband as mrFitmb
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


def get_data(butler, tract, name_patch, cat_type=None, exposure_type=None, bands=None):
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
    bands : iterable of `str`
        Names of bandpass filters for filter-dependent fields. Default ["HSC-I", "HSC-R", "HSC-G"].

    Returns
    -------
    data : `dict` [`str`, `dict` [`str`]]
        A dict of dicts keyed by filter name, each containing:
        ``"exposures"``
            The exposure of that filter (`lsst.afw.image.Exposure`)
        ``"sources"``
            The catalog of sources to fit (`lsst.afw.table.SourceCatalog`)

    """
    if bands is None:
        bands = ["HSC-I", "HSC-R", "HSC-G"]
    if exposure_type is None:
        exposure_type = "deepCoadd_calexp"
    if cat_type is None:
        cat_type = "deepCoadd_meas"

    dataId = {"tract": tract, "patch": name_patch}
    data = []
    for i, band in enumerate(bands):
        data.append(mrFitmb.CatalogExposure(
            band=band,
            catalog=butler.get(cat_type, dataId, filter=band),
            exposure=butler.get(exposure_type, dataId, filter=band),
        ))
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
        'repo': dict(type=str, nargs='?', default="/datasets/hsc/repo/rerun/RC/w_2021_02/DM-28282/",
                     help="Path to Butler repository to read from"),
        'filenameOut': dict(type=str, nargs='?', default=None, help="Output catalog filename", kwarg=True),
        'radec': dict(type=float, nargs=2, default=None, help="RA/dec coordinate of source"),
        'name_patch': dict(type=str, nargs='?', default="4,4", help="Butler patch string"),
        'tract': dict(type=int, nargs='?', default=9813, help="Butler tract ID"),
        'img_multi_plot_max': dict(type=float, nargs='?', default=None,
                                   help="Max value for colour images in plots"),
        'mag_prior_field': dict(type=str, nargs='?', default='base_PsfFlux_mag'),
        'filter_prior': dict(type=str, nargs='?', default='HSC-I'),
        'weights_band': dict(type=float, nargs='*', default=None,
                             help="Weights per filter to rescale images in multi-band plots"),
        'loglevel': {'type': int, 'nargs': '?', 'default': 21, 'help': 'logging.Logger default level'},
    }
    for param, field in MultiProFitConfig._fields.items():
        is_list = hasattr(field, 'itemtype')
        type_default = field.itemtype if is_list else field.dtype
        flag = dict(type=type_default if type_default is not bool else str2bool, nargs='*' if is_list else '?',
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
        print(f'Error: {e}')
        parser.print_help()
        sys.exit(status=1)
    logging.basicConfig(stream=sys.stdout, level=args.loglevel)
    butler = Butler(args.repo)

    if args.name_patch is not None and args.tract is not None:
        tract = args.tract
        name_patch = args.name_patch
    else:
        ra, dec = args.radec
        tract, patch = get_patch_tract(ra, dec, butler)
        name_patch = ','.join([str(x) for x in patch.getIndex()])
    sources = butler.get('deepCoadd_ref', tract=tract, patch=name_patch)

    argsvars = vars(args)
    kwargs = {key: argsvars[key] for key in kwargs}
    config = MultiProFitTask.ConfigClass(**kwargs)

    if True or config.usePriorShapeDefault:
        field_prior = args.mag_prior_field
        dataId_prior = {'tract': tract, 'patch': name_patch, 'filter': args.filter_prior}
        cat_prior = butler.get("deepCoadd_meas", dataId_prior)
        calib_prior = butler.get("deepCoadd_photoCalib", dataId_prior)
        mags_prior = calib_prior.calibrateCatalog(cat_prior)[field_prior]
    else:
        mags_prior = None

    task = MultiProFitTask(config=config)
    data = get_data(butler, tract, name_patch=name_patch, bands=args.bands)
    catalog, results = task.fit(
        data, sources, img_multi_plot_max=args.img_multi_plot_max, weights_band=args.weights_band,
        mags_prior=mags_prior,
    )
    return catalog, results


if __name__ == '__main__':
    cat, res = main()
