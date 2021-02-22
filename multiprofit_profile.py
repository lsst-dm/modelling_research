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

n_eval = 10
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

defaults = dict(psfmodel=namePsfModel, psfpixel=True)

modelspecs = [
    mpfFit.ModelSpec(name="gausspx", model=nameSersicModel, fixedparams='nser', initparams="nser=0.5",
                     inittype="moments", **defaults),
    mpfFit.ModelSpec(name=f"{nameMG}expgpx", model=nameSersicModel, fixedparams='nser', initparams="nser=1",
                     inittype="guessgauss2exp:gausspx", **defaults),
    mpfFit.ModelSpec(name=f"{nameMG}devepx", model=nameSersicModel, fixedparams='nser', initparams="nser=4",
                     inittype=f"guessexp2dev:{nameMG}expgpx", **defaults),
    mpfFit.ModelSpec(name=f"{nameMG}cmodelpx", model=f"{nameSersicPrefix}:2",
                     fixedparams="cenx;ceny;nser;sigma_x;sigma_y;rho", initparams="nser=4,1",
                     inittype=f"{nameMG}devepx;{nameMG}expgpx", **defaults),
]

noiseReplacers = {band: rebuildNoiseReplacer(exposure, sources) for band, exposure in exposures.items()}

for noiseReplacer in noiseReplacers.values():
    noiseReplacer.insertSource(source.getId())

exposures_data = {
    band: (
        mpfObj.Exposure(
            band=band, image=np.float64(exposure.image.subset(bbox).array),
            error_inverse=1 / np.float64(exposure.variance.subset(bbox).array),
            is_error_sigma=False,
        ),
        exposure.getPsf().computeKernelImage(center),
    )
    for band, exposure in exposures.items()
}


def get_exposurePsfs(exposures_data):
    return [
        (exposure_data[0], mpfObj.PSF(band, image=exposure_data[1], engine="galsim"))
        for band, exposure_data in exposures_data.items()
    ]


cProfile.run(
    "for _ in range(n_eval): "
    "mpfFit.fit_galaxy_exposures(get_exposurePsfs(exposures_data), exposures.keys(), modelspecs)",
    name_source,
)
for noiseReplacer in noiseReplacers.values():
    noiseReplacer.removeSource(source.getId())

p = pstats.Stats(name_source)
p.sort_stats(pstats.SortKey.TIME).print_stats(50)