# This file is part of modelling_research.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import logging
import sys

from lsst.daf.butler import Butler
from lsst.geom import SpherePoint, degrees
import lsst.pipe.tasks.fit_multiband as fitMb
from lsst.meas.extensions.multiprofit.fit_multiband import MultiProFitConfig, MultiProFitTask
from multiprofit.utils import str2bool


def get_patch_tract(ra, dec, butler, collections, skymap):
    """Get the patch and tract that contains a given sky coordinate.

    Parameters
    ----------
    ra : `float`
        Right ascenscion in degrees.
    dec : `float`
        Declination in degrees.
    butler : `lsst.daf.butler.Butler`
        A Generation 3 butler.

    Returns
    -------
    tract : `int`
        The tract ID.
    patch : `tuple` [`int`, `int`]
        The patch IDs containing the coordinates.
    """
    skymap = butler.get("skyMap", collections=collections, skymap=skymap)
    spherePoint = SpherePoint(ra, dec, degrees)
    tract = skymap.findTract(spherePoint).getId()
    return tract, skymap[tract].findPatch(spherePoint)


def get_data(
        butler, tract, patch, skymap, collections, cat_type=None, exposure_type=None, bands=None,
):
    """Get the necessary data to run the MultiProFit task on a range of sources in a region.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Generation 3 butler.
    tract : `int`
        A tract number.
    patch : `str`
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

    dataId = {"tract": tract, "patch": patch}
    data = []
    for i, band in enumerate(bands):
        data.append(fitMb.CatalogExposure(
            catalog=butler.get(cat_type, dataId, band=band, skymap=skymap, collections=collections),
            exposure=butler.get(exposure_type, dataId, band=band, skymap=skymap, collections=collections),
            dataId={'tract': tract, 'patch': patch, 'band': band},
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
        'repo': dict(type=str, nargs='?', default="/repo/main",
                     help="Path to Butler repository to read from"),
        'collections': dict(type=str, nargs='?', default="HSC/runs/DM-30xxx", help="Collections to read from"),
        'skymap': dict(type=str, nargs='?', default="hsc_rings_v1", help="Skymap name"),
        'filenameOut': dict(type=str, nargs='?', default=None, help="Output catalog filename", kwarg=True),
        'radec': dict(type=float, nargs=2, default=None, help="RA/dec coordinate of source"),
        'patch': dict(type=int, nargs='?', default=40, help="Butler patch number"),
        'tract': dict(type=int, nargs='?', default=9813, help="Butler tract ID"),
        'img_multi_plot_max': dict(type=float, nargs='?', default=None,
                                   help="Max value for colour images in plots"),
        'filter_prior': dict(type=str, nargs='?', default='HSC-I'),
        'weights_band': dict(type=float, nargs='*', default=None,
                             help="Weights per filter to rescale images in multi-band plots"),
        'loglevel': {'type': int, 'nargs': '?', 'default': None, 'help': 'logging.Logger default level'},
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

    if args.loglevel is None:
        logger = None
    else:
        logger = logging.getLogger('mpf_run')
        logging.basicConfig(stream=sys.stdout, level=args.loglevel)
    butler = Butler(args.repo)

    if args.patch is not None and args.tract is not None:
        tract = args.tract
        patch = args.patch
    else:
        ra, dec = args.radec
        tract, patch = get_patch_tract(ra, dec, butler, args.collections, args.skymap)
    sources = butler.get('deepCoadd_ref', tract=tract, patch=patch, skymap=args.skymap, collections=args.collections)

    argsvars = vars(args)
    kwargs = {key: argsvars[key] for key in kwargs}
    config = MultiProFitTask.ConfigClass(**kwargs)

    task = MultiProFitTask(config=config, schema=sources.schema)
    bands = set(args.bands_fit)
    if hasattr(args, 'bands_read'):
        bands_read = args.bands_read
        if bands_read is not None:
            bands = bands.union(set(bands_read))
    data = get_data(butler, tract, patch=patch, bands=bands, skymap=args.skymap, collections=args.collections)
    print(bands, [datum.dataId for datum in data])
    catalog, results = task.fit(
        data, sources, img_multi_plot_max=args.img_multi_plot_max, weights_band=args.weights_band, logger=logger,
    )
    return catalog, results


if __name__ == '__main__':
    cat, res = main()
