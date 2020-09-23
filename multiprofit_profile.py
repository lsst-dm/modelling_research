from lsst.daf.persistence import Butler
from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
import numpy as np
import multiprofit.fitutils as mpfFit
import multiprofit.objects as mpfObj
import cProfile
import pstats

butler = Butler("/datasets/hsc/repo/rerun/RC/w_2020_36/DM-26637/")
filters = ['HSC-I']
dataId = {"tract": 9813, "patch": "4,4", "filter": filters[0]}
sources = butler.get("deepCoadd_meas", dataId)
exposures = {band: butler.get("deepCoadd_calexp", dataId, filter=band) for band in filters}

# 1338 takes a while if you want to benchmark a source dominated by model evaluation time
idx_source = 1337
name_source = f"profile_{idx_source}"
source = sources[idx_source]
foot = source.getFootprint()
bbox = foot.getBBox()
center = bbox.getCenter()

gaussianOrderSersic, gaussianOrderPsf = 8, 2
nameMG = f"mg{gaussianOrderSersic}"
namePsfModel = f"gaussian:{gaussianOrderPsf}"
nameSersicPrefix = f"mgsersic{gaussianOrderSersic}"
nameSersicModel = f"{nameSersicPrefix}:1"
nameSersicAmpModel = f"gaussian:{gaussianOrderSersic}+rscale:1"

modelspecs = [
    dict(name="gausspx", model=nameSersicModel, fixedparams='nser', initparams="nser=0.5",
         inittype="moments", psfmodel=namePsfModel, psfpixel="T"),
    dict(name=f"{nameMG}expgpx", model=nameSersicModel, fixedparams='nser', initparams="nser=1",
         inittype="guessgauss2exp:gausspx", psfmodel=namePsfModel, psfpixel="T"),
    dict(name=f"{nameMG}devepx", model=nameSersicModel, fixedparams='nser', initparams="nser=4",
         inittype=f"guessexp2dev:{nameMG}expgpx", psfmodel=namePsfModel, psfpixel="T"),
    dict(name=f"{nameMG}cmodelpx", model=f"{nameSersicPrefix}:2",
         fixedparams="cenx;ceny;nser;sigma_x;sigma_y;rho", initparams="nser=4,1",
         inittype=f"{nameMG}devepx;{nameMG}expgpx", psfmodel=namePsfModel, psfpixel="T"),
]

noiseReplacers = {band: rebuildNoiseReplacer(exposure, sources) for band, exposure in exposures.items()}

for noiseReplacer in noiseReplacers.values():
    noiseReplacer.insertSource(source.getId())

exposurePsfs = []
for band, exposure in exposures.items():
    mpfExposure = mpfObj.Exposure(
        band=band, image=np.float64(exposure.image.subset(bbox).array),
        error_inverse=1 / np.float64(exposure.variance.subset(bbox).array),
        is_error_sigma=False)
    mpfPsf = mpfObj.PSF(band, image=exposure.getPsf().computeKernelImage(center),
                        engine="galsim")
    exposurePsfs.append((mpfExposure, mpfPsf))

cProfile.run("mpfFit.fit_galaxy_exposures(exposurePsfs, exposures.keys(), modelspecs)", name_source)
for noiseReplacer in noiseReplacers.values():
    noiseReplacer.removeSource(source.getId())

p = pstats.Stats(name_source)
p.sort_stats(pstats.SortKey.TIME).print_stats(50)