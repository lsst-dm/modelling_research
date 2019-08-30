import argparse
import logging
import numpy as np
import sys

import lsst.afw.table as afwTable
from lsst.daf.persistence import Butler
from lsst.geom import SpherePoint, degrees
from multiprofit_task import MultiProFitTask


def getPatch(ra, dec, butler, tract=9813):
    skymap = butler.get("deepCoadd_skyMap", dataId={"tract": tract})
    spherePoint = SpherePoint(ra, dec, degrees)
    tract = skymap.findTract(spherePoint).getId()
    if tract != 9813:
        raise RuntimeError('ra, dec is in tract {} != 9813'.format(tract))
    return skymap[tract].findPatch(spherePoint)


def multiProFit(butler, tract, patchname, filters=None, catalog=None, idx_begin=0, idx_end=np.Inf,
                printTrace=False, **kwargs):
    if filters is None:
        filters = ["HSC-I", "HSC-R", "HSC-G"]
    dataId = {"tract": tract, "patch": patchname, "filter": filters[0]}
    measCat = butler.get("deepCoadd_meas", dataId) if catalog is None else catalog
    #frame = 60
    #wcs = skymap[tract].getWcs()
    #point = wcs.skyToPixel(spherePoint)
    #sampleBBox = Box2I(Point2I(point[0] - frame, point[1] - frame), Extent2I(2 * frame, 2 * frame))
    coadds = {band: butler.get("deepCoadd_calexp", dataId, filter=band) for band in filters}
    config = MultiProFitTask.ConfigClass(**kwargs)
    task = MultiProFitTask(config=config)
    catalog, results = task.fit(coadds, measCat, idx_begin=idx_begin, idx_end=idx_end, printTrace=printTrace)
    return catalog, results, dataId


def main():
    parser = argparse.ArgumentParser(description='MultiProFit Butler Task running test')
    flags = {
        'repo': dict(type=str, nargs='?', default="/datasets/hsc/repo/rerun/RC/w_2019_26/DM-19560/",
                     help="Path to Butler repository to read from"),
        'inputFilename': dict(type=str, nargs='?', default=None, help="Output catalog filename"),
        'outputFilename': dict(type=str, nargs='?', default=None, help="Input catalog filename"),
        'radec': dict(type=float, nargs=2, default=None, help="RA/dec coordinate of source"),
        'patchName': dict(type=str, nargs='?', default="4,4", help="Butler patch string"),
        'tract': dict(type=int, nargs='?', default=9813, help="Butler tract ID"),
        'filters': dict(type=str, nargs='*', default=['HSC-I'], help="List of bandpass filters"),
        'idx_begin': dict(type=int, nargs='?', default=0, help="Initial row index to fit"),
        'idx_end': dict(type=int, nargs='?', default=np.Inf, help="Final row index to fit"),
        'printTrace': dict(type=bool, nargs='?', default=False, help="Print traceback for errors"),
        'loglevel': {'type': int, 'nargs': '?', 'default': 30, 'help': 'logging.Logger default level'},
        'computeMeasModelfitLikelihood': dict(type=bool, nargs='?', default=False, kwarg=True,
                                              help="Set config computeMeasModelfitLikelihood flag", ),
        'fitCModelExp': dict(type=bool, nargs='?', default=False, kwarg=True,
                             help="Set config fitCModelExp flag"),
        'fitSersicFromCModel': dict(type=bool, nargs='?', default=False, kwarg=True,
                                    help="Set config fitSersicFromCModel flag"),
    }
    kwargs = {}
    for key, value in flags.items():
        if 'help' in value:
            value['help'] = f"{value['help']} (default: {str(value['default'])})"
        if 'kwarg' in value:
            kwargs[key] = None
            del value['kwarg']
        parser.add_argument('--' + key, **value)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=args.loglevel)
    butler = Butler(args.repo)
    catalog = afwTable.SimpleCatalog.readFits(args.inputFilename) if args.inputFilename is not None else None
    if args.patchName is not None:
        patchname = args.patchName
    else:
        ra, dec = args.radec
        patch = getPatch(ra, dec, butler)
        patchname = ','.join([str(x) for x in patch.getIndex()])
    argsvars = vars(args)
    kwargs = {key: argsvars[key] for key in kwargs}
    catalog, results, dataId = multiProFit(
        butler, args.tract, patchname=patchname, catalog=catalog, filters=args.filters,
        idx_begin=args.idx_begin, idx_end=args.idx_end, **kwargs)
    if args.outputFilename is not None:
        catalog.writeFits(args.outputFilename)


if __name__ == '__main__':
    main()
