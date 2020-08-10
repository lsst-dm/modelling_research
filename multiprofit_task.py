from collections import defaultdict, namedtuple
import copy
import logging
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
from lsst.meas.modelfit.display import buildCModelImages
from lsst.meas.modelfit.cmodel.cmodelContinued import CModelConfig
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import modelling_research.make_cutout as cutout
import matplotlib.pyplot as plt
import multiprofit.fitutils as mpfFit
import multiprofit.objects as mpfObj
from multiprofit.priors import get_hst_size_prior
import numpy as np
import os
import time
import traceback

pixel_scale_hst = 0.03
pixel_scale_hsc = 0.168


class MultiProFitConfig(pexConfig.Config):
    """Configuration for the MultiProFit profile fitter.

    Notes
    -----
    gaussianOrderSersic only has a limited number of valid values (those supported by multiprofit's
    MultiGaussianApproximationComponent).
    """
    backgroundPriorMultiplier = pexConfig.Field(dtype=float, default=None,
                                                doc="Multiplier for background level prior sigma")
    bandMeasCatToCopyFrom = pexConfig.Field(
        dtype=str, default=None, doc="The band of the measurement catalog to copy data from, even if a "
                                     "source cat is provided")
    computeMeasModelfitLikelihood = pexConfig.Field(dtype=bool, default=False,
                                                    doc="Whether to compute the log-likelihood of best-fit "
                                                        "meas_modelfit parameters per model")
    deblend = pexConfig.Field(dtype=bool, default=False, doc="Whether to fit parents simultaneously with "
                                                             "children")
    deblendFromDeblendedFits = pexConfig.Field(
        dtype=bool, default=False, doc="Whether to fit parents simultaneously and linearly using prior "
                                       "fits to deblended children")
    deblendFromDeblendedFitsNonlinear = pexConfig.Field(
        dtype=bool, default=False, doc="Whether to fit parents simultaneously and non-linearly using prior "
                                       "fits to deblended children")
    estimateContiguousDenoisedMoments = pexConfig.Field(
        dtype=bool, default=True, doc="Whether models initiated from moments should estimate within "
                                      "contiguous positive pixels in a naively de-noised image")
    filenameOut = pexConfig.Field(dtype=str, default="", doc="Filename for output of FITS table")
    filenameOutDeblend = pexConfig.Field(dtype=str, default="", doc="Filename for output of FITS table with"
                                                                    " deblended fits")
    fitBackground = pexConfig.Field(dtype=bool, default=False,
                                    doc="Whether to fit a flat background level for each band")
    fitCModel = pexConfig.Field(dtype=bool, default=True,
                                doc="Whether to perform a CModel (linear combo of exponential and "
                                    "deVaucouleurs) fit per source; necessitates doing exp. + deV. fits")
    fitCModelExp = pexConfig.Field(dtype=bool, default=False,
                                   doc="Whether to perform an exponential fit with a fixed center (as "
                                       "CModel does in meas_modelfit) per source")
    fitGaussian = pexConfig.Field(dtype=bool, default=False,
                                  doc="Whether to perform a single Gaussian fit without PSF convolution")
    fitGaussianPsfConv = pexConfig.Field(dtype=bool, default=True,
                                         doc="Whether to perform a PSF-convolved single Gaussian fit")
    fitHstCosmos = pexConfig.Field(dtype=bool, default=False,
                                   doc="Whether to fit COSMOS HST F814W images instead of repo images")
    fitDevExpFromCModel = pexConfig.Field(dtype=bool, default=False,
                                          doc="Whether to perform a MG Sersic approximation Dev+Exp profile "
                                              "fit (initialized from previous exp./Dev. fits) per source")
    fitPrereqs = pexConfig.Field(dtype=bool, default=False, doc="Set fit(Model) flags for necessary "
                                                                "prerequisites even if not specified")
    fitSersic = pexConfig.Field(dtype=bool, default=True, doc="Whether to perform a MG Sersic approximation "
                                                              "profile fit per source")
    fitSersicFromCModel = pexConfig.Field(dtype=bool, default=False,
                                          doc="Whether to perform a MG Sersic approximation profile fit "
                                              "(initalized from previous exp./dev. fits) per source;"
                                              " ignored if fitCModel is False")
    fitSersicFromGauss = pexConfig.Field(dtype=bool, default=True,
                                         doc="Whether to perform a MG Sersic approximation profile fit "
                                             "(initalized from previous gauss fit) per source")
    fitSersicAmplitude = pexConfig.Field(dtype=bool, default=True,
                                         doc="Whether to perform a linear fit of the Gaussian"
                                             " amplitudes for the MG Sersic approximation profile fit per"
                                             " source; has no impact if fitSersic is False")
    fitSersicFromCModelAmplitude = pexConfig.Field(dtype=bool, default=True,
                                                   doc="Whether to perform a linear fit of the Gaussian"
                                                       " amplitudes for the MG Sersic approximation profile"
                                                       " fit (initialized from previous exp.Dev. fits) per"
                                                       " source; has no impact if fitSersicFromCModel is"
                                                       " False")
    fitSersicX2FromDevExp = pexConfig.Field(dtype=bool, default=False,
                                            doc="Whether to perform a MG Sersic approximation SersicX2 "
                                                "profile fit (initialized from previous devExp fit) per "
                                                "source; ignored if fitDevExpFromCModel is False")
    fitSersicX2DEAmplitude = pexConfig.Field(dtype=bool, default=False,
                                             doc="Whether to perform a linear fit of the Gaussian "
                                                 "amplitudes for the MG SersicX2 approximation profile fit "
                                                 "for each source; ignored if fitSersicX2FromDevExp is False")
    fitSersicX2FromSerExp = pexConfig.Field(dtype=bool, default=False,
                                            doc="Whether to perform a MG Sersic approximation SersicX2 "
                                                "profile fit (initialized from previous serExp fit) per "
                                                "source; ignored if fitSersicFromCModel is False")
    fitSersicX2SEAmplitude = pexConfig.Field(dtype=bool, default=False,
                                             doc="Whether to perform a linear fit of the Gaussian "
                                                 "amplitudes for the MG SersicX2 approximation profile fit "
                                                 "for each source; ignored if fitSersicX2FromSeRExp is False")
    gaussianOrderPsf = pexConfig.Field(dtype=int, default=2, doc="Number of Gaussians components for the PSF")
    gaussianOrderSersic = pexConfig.Field(dtype=int, default=8, doc="Number of Gaussians components for the "
                                                                    "MG Sersic approximation galaxy profile")
    gaussianSizePriorSigma = pexConfig.Field(
        dtype=float, default=0.2, doc="Std. dev. of the size (sigma) prior for the Gaussian model (pixels)")
    intervalOutput = pexConfig.Field(dtype=int, default=100, doc="Number of sources to fit before writing "
                                                                 "output")
    isolatedOnly = pexConfig.Field(dtype=bool, default=False, doc="Whether to fit only isolated sources")
    maxParentFootprintPixels = pexConfig.Field(dtype=int, default=1000000,
                                               doc="Maximum number of pixels in a parent footprint allowed "
                                                   "before failing or reverting to child footprint")
    maxNChildParentFit = pexConfig.Field(dtype=int, default=25, doc="Maximum number of children allowed to "
                                                                    "fit a parent footprint")
    outputChisqred = pexConfig.Field(dtype=bool, default=True, doc="Whether to save the reduced chi^2 of "
                                                                   "each model's best fit")
    outputLogLikelihood = pexConfig.Field(dtype=bool, default=True, doc="Whether to save the log likelihood "
                                                                        "of each model's best fit")
    outputRuntime = pexConfig.Field(dtype=bool, default=True, doc="Whether to save the runtime of each "
                                                                  "model")
    plotOnly = pexConfig.Field(dtype=bool, default=False, doc="Whether to only attempt to plot existing "
                                                              "fits; requires resume=True")
    priorCentroidSigma = pexConfig.Field(dtype=float, default=np.Inf, doc="Centroid prior sigma")
    psfHwhmShrink = pexConfig.Field(
        dtype=float, default=0.,
        doc="Length (pix) to subtract from PSF HWHM (x,y) in quadrature before fitting PSF-convolved models")
    resume = pexConfig.Field(dtype=bool, default=False, doc="Whether to resume from the previous output file")
    skipDeblendTooManyPeaks = pexConfig.Field(dtype=bool, default=False,
                                              doc="Whether to skip fitting sources with "
                                                  "deblend_tooManyPeaks flag set")
    useSpans = pexConfig.Field(dtype=bool, default=False,
                               doc="Whether to use spans for source cutouts (i.e. the detected pixels) rather"
                                   " than the whole box")
    useSdssShape = pexConfig.Field(dtype=bool, default=False,
                                   doc="Whether to use the baseSdssShape* moments to initialize Gaussian "
                                       "fits")
    useParentFootprint = pexConfig.Field(dtype=bool, default=False,
                                         doc="Whether to use the parent's footprint when fitting deblended "
                                             "children")
    usePriorShapeDefault = pexConfig.Field(dtype=bool, default=False,
                                           doc="Whether to use the default shape prior")
    usePriorBackgroundLocalEstimate = pexConfig.Field(
        dtype=bool, default=False, doc="Whether to use a local estimate of the background level to set the"
                                       " background prior mean/sigma; generally a bad idea")

    def getModelSpecs(self):
        """Get a list of dicts of model specifications for MultiProFit/

        Returns
        -------
        modelSpecs : `list` [`dict`]
            MultiProFit model specifications, as used by multiprofit.fitutils.fit_galaxy_exposures().
        """
        modelSpecs = []
        nameMG = f"mg{self.gaussianOrderSersic}"
        namePsfModel = f"gaussian:{self.gaussianOrderPsf}"
        nameSersicPrefix = f"mgsersic{self.gaussianOrderSersic}"
        nameSersicModel = f"{nameSersicPrefix}:1"
        nameSersicAmpModel = f"gaussian:{self.gaussianOrderSersic}+rscale:1"
        nameSersicX2Model = f"{nameSersicPrefix}:2"
        nameSersicX2AmpModel = f"gaussian:{2*self.gaussianOrderSersic}+rscale:2"
        allParams = "cenx;ceny;nser;sigma_x;sigma_y;rscale;rho"
        if self.fitPrereqs:
            prereqs = {
                'fitGaussianPsfConv': ['fitSersicFromGauss', 'fitCModel'],
                'fitSersic': ['fitSersicAmplitude'],
                'fitSersicFromCModel': ['fitSersicFromCModelAmplitude', 'fitSersicX2FromSerExp'],
                'fitCModel': ['fitSersicFromCModel', 'fitDevExpFromCModel'],
                'fitSersicX2FromDevExp': ['fitSersicX2DEAmplitude'],
                'fitDevExpFromCModel': ['fitSersicX2FromDevExp'],
                'fitSersicX2FromSerExp': ['fitSersicX2SEAmplitude'],
            }
            for req, depends in prereqs.items():
                dict_self = self.toDict()
                if (not dict_self[req]) and any([dict_self[dep] for dep in depends]):
                    self.update(**{req: True})
        defaults = {
            'psfmodel': namePsfModel,
            'psfpixel': "T",
        }
        init_values = self.plotOnly or self.deblendFromDeblendedFits
        inittype_moments = "moments" if not init_values else "values"
        if self.fitGaussianPsfConv:
            modelSpecs.append(
                dict(name="gausspx", model=nameSersicModel, fixedparams='nser', initparams="nser=0.5",
                     inittype=inittype_moments, **defaults))
        if self.fitCModel:
            modelSpecs.extend([
                dict(name=f"{nameMG}expgpx", model=nameSersicModel, fixedparams='nser', initparams="nser=1",
                     inittype="guessgauss2exp:gausspx" if not init_values else "values", **defaults),
                dict(name=f"{nameMG}devepx", model=nameSersicModel, fixedparams='nser', initparams="nser=4",
                     inittype=f"guessexp2dev:{nameMG}expgpx" if not init_values else "values", **defaults),
                dict(name=f"{nameMG}cmodelpx", model=f"{nameSersicPrefix}:2",
                     fixedparams="cenx;ceny;nser;sigma_x;sigma_y;rho", initparams="nser=4,1",
                     inittype=f"{nameMG}devepx;{nameMG}expgpx", **defaults),
            ])
            if self.fitSersicFromCModel:
                modelSpecs.extend([
                    dict(name=f"{nameMG}serbpx", model=nameSersicModel, fixedparams='', initparams='',
                         inittype="best" if not init_values else "values", **defaults),
                ])
                if self.fitSersicFromCModelAmplitude:
                    modelSpecs.append(
                        dict(name=f"{nameMG}serbapx", model=nameSersicAmpModel, fixedparams=allParams,
                             initparams="rho=inherit;rscale=modify", inittype=f"{nameMG}serbpx",
                             unlimitedparams="sigma_x;sigma_y", **defaults)
                    )
                if self.fitSersicX2FromSerExp:
                    modelSpecs.append(
                        dict(name=f"{nameMG}serx2sepx", model=nameSersicX2Model, fixedparams='',
                             initparams='',
                             inittype=f"{nameMG}serbpx;{nameMG}expgpx" if not init_values else "values",
                             **defaults)
                    )
                    if self.fitSersicX2SEAmplitude:
                        modelSpecs.append(
                            dict(name=f"{nameMG}serx2seapx", model=nameSersicX2AmpModel,
                                 fixedparams=allParams, initparams="rho=inherit;rscale=modify",
                                 inittype=f"{nameMG}serx2sepx", unlimitedparams="sigma_x;sigma_y", **defaults)
                        )
            if self.fitDevExpFromCModel:
                modelSpecs.append(
                    dict(name=f"{nameMG}devexppx", model=nameSersicX2Model, fixedparams='nser',
                         initparams='nser=4,1',
                         inittype=f"{nameMG}devepx;{nameMG}expgpx" if not init_values else "values",
                         **defaults)
                )
                if self.fitSersicX2FromDevExp:
                    modelSpecs.append(
                        dict(name=f"{nameMG}serx2px", model=nameSersicX2Model, fixedparams='', initparams='',
                             inittype=f"{nameMG}devexppx" if not init_values else "values", **defaults)
                    )
                    if self.fitSersicX2DEAmplitude:
                        modelSpecs.append(
                            dict(name=f"{nameMG}serx2apx", model=nameSersicX2AmpModel, fixedparams=allParams,
                                 initparams="rho=inherit;rscale=modify", inittype=f"{nameMG}serx2px",
                                 **defaults)
                        )
        if self.fitCModelExp:
            modelSpecs.append(
                dict(name=f"{nameMG}expcmpx", model=nameSersicModel, fixedparams='cenx;ceny;nser',
                     initparams="nser=1", inittype="moments", **defaults)
            )
        if self.fitSersicFromGauss:
            modelSpecs.append(
                dict(name=f"{nameMG}sergpx", model=nameSersicModel, fixedparams='', initparams="nser=1",
                     inittype="guessgauss2exp:gausspx" if not init_values else "values", **defaults)
            )
        if self.fitSersic:
            modelSpecs.append(
                dict(name=f"{nameMG}sermpx", model=nameSersicModel, fixedparams='', initparams="nser=1",
                     inittype=inittype_moments, **defaults)
            )
            if self.fitSersicAmplitude:
                modelSpecs.append(
                    dict(name=f"{nameMG}serapx", model=nameSersicAmpModel, fixedparams=allParams,
                         initparams="rho=inherit;rscale=modify", inittype=f"{nameMG}sermpx",
                         unlimitedparams="sigma_x;sigma_y", **defaults)
                )
        return modelSpecs


def defaultdictNested():
    """Get a nested defaultdict with defaultdict default value.

    Returns
    -------
    defaultdict : `defaultdict`
        A `defaultdict` with `defaultdict` default values.
    """
    return defaultdict(defaultdictNested)


def joinFilter(separator, items, exclusion=None):
    """Join an iterable of items by a separator, filtering out an exclusion.

    Parameters
    ----------
    separator : `string`
        The separator to join items by.
    items : iterable of `str`
        Items to join.
    exclusion : `str`, optional
        The pattern to exclude; default None.

    Returns
    -------
    joined : `str`
        The joined string.
    """
    return separator.join(filter(exclusion, items))


class MultiProFitTask(pipeBase.Task):
    """A task to run the MultiProFit source modelling code on a catalog with detections and heavy footprints,
    returning additional measurements in a new SourceCatalog.

    This task uses MultiProFit to fit the PSF and various analytic profiles to every source in an Exposure.
    It is currently set up to fit a Gaussian mixture model for the PSF, and various forms of Gaussian
    mixture models for the source itself. These are primarily multi-Gaussian Sersic approximations,
    representing a Sersic profile as a sum of 4 or 8 Gaussians. The advantage to this approach is that the
    model and its derivatives are analytic, even including PSF convolution.

    MultiProFit can model multiple filters simultaneously by fitting a separate magnitude for each
    component in each filter. It can also fit more general non-parametric Gaussian mixtures, where a number of
    Gaussians share the same ellipse shape but have different amplitudes and/or sizes. This allows for
    non-parametric, monotonically-decreasing radial profiles of (nearly) arbitrary shape and with color
    gradients (in multi-band mode).

    The task can be configured to fit a variety of different useful model specifications via the fit* Config
    parameters. The two most useful options are:

    - `fitSersic` : A Gaussian mixture Sersic profile, initialized from the image moments;
    - `fitCModel` : The SDSS/HSC CModel - a linear combination of exponential and de Vaucouleurs profile fits.
                    It initially fits a Gaussian, and then an exponential and de Vaucouleurs profile.

    Other options are largely intended for testing purposes:
    - `fitCModelExp` : Fits CModel again, but with a fixed centroid, as is done in meas_modelfit;
    - `fitSersicFromCModel`: Fits a Sersic profile initialized from the best of the CModel prerequisite fits.

    More experimental options include
    - `fitSersicAmplitude` : If `fitSersic` is enabled, it will begin with that fit and then fit the
                             amplitudes (magnitudes) of each Gaussian, forming a non-parametric radial
                             profile with color gradients (in multi-band mode).

    Expert users can supply their own model specifications to the run method, in which case the fit* Config
    parameters are ignored.

    The remaining Config parameters either control output options or set the order of the PSF/Sersic profile.

    This task has numerous private methods. These are only intended to be called by run (fit) and should
    not be called by any other method. Most are used to add the necessary new fields to the catalog,
    broadly categorized as base fields (directly fit parameters), extra fields (likelihoods, runtimes,
    etc.) for both the source model fits and the PSF fits.

    Notes
    -----

    See https://github.com/lsst-dm/multiprofit for more information about MultiProFit, as well as
    https://github.com/lsst-dm/modelling_research for various investigation about its suitability for LSST.

    """
    ConfigClass = MultiProFitConfig
    _DefaultName = "multiProFit"
    meas_modelfit_models = ("dev", "exp", "cmodel")
    keynames = {'runtimeKey': 'time_total', 'failFlagKey': 'fail_flag'}
    ParamDesc = namedtuple('ParamInfo', ['doc', 'unit'])
    params_multiprofit = {
        'cenx': ParamDesc('Centroid x coordinate', 'pixel'),
        'ceny': ParamDesc('Centroid y coordinate', 'pixel'),
        'flux': ParamDesc('Total flux', 'Jy'),
        'nser': ParamDesc('Sersic index', ''),
        'rho': ParamDesc('Ellipse correlation coefficient', ''),
        'sigma_x': ParamDesc('Ellipse x half-light distance', 'pixel'),
        'sigma_y': ParamDesc('Ellipse y half-light distance', 'pixel'),
        'fluxFrac': ParamDesc('Flux fraction', ''),
    }
    params_extra = {'chisqred', 'loglike', 'time', 'nEvalFunc', 'nEvalGrad'}
    prefix = 'multiprofit_'
    prefix_psf = f'psf_'
    postfix_nopsf = '-nopsf'

    def __init__(self, modelSpecs=None, **kwargs):
        """Initialize the task with model specifications.

        Parameters
        ----------
        modelSpecs : iterable of `dict`, optional
            MultiProFit model specifications, as used by multiprofit.fitutils.fit_galaxy_exposures. Defaults
            to `self.config.getModelSpecs`().
        **kwargs
            Additional keyword arguments passed to `lsst.pipe.base.Task.__init__`
        """
        pipeBase.Task.__init__(self, **kwargs)
        if modelSpecs is None:
            modelSpecs = self.config.getModelSpecs()
        self.modelSpecs = modelSpecs
        self.modelSpecsDict = {modelspec['name']: modelspec for modelspec in modelSpecs}
        self.schema = None
        self.modeller = mpfObj.Modeller(None, 'scipy')
        self.models = {}
        self.mask_names_zero = ['BAD', 'EDGE', 'SAT', 'NO_DATA']

    @staticmethod
    def _getMapper(schema):
        """Return a suitably configured schema mapper.

        Parameters
        ----------
        schema: `lsst.afw.table.Schema`
            A table schema to setup a mapper for and add basic keys to.

        Returns
        -------
        mapper: `lsst.afw.table.SchemaMapper`
            A mapper for `schema`.
        """
        mapper = afwTable.SchemaMapper(schema, True)
        mapper.addMinimalSchema(schema, True)
        mapper.editOutputSchema().disconnectAliases()
        return mapper

    def _getExposurePsfs(
            self, exposures, source, extras, footprint=None, failOnLargeFootprint=False
    ):
        """ Get exposure-PSF pairs as required by various MultiProFit fitutils functions.

        Parameters
        ----------
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        source : `lsst.afw.table.SourceRecord`
            A deblended source to fit.
        extras : iterable of `lsst.meas.base.NoiseReplacer` or `multiprofit.object.Exposure`
            An iterable of NoiseReplacers that will insert the source into every exposure, or a tuple of
            HST exposures if fitting HST data.
        footprint : `lsst.afw.detection.Footprint`, optional
            The footprint to fit within. Default source.getFootprint().
        failOnLargeFootprint : `bool`, optional
            Whether to return a failure if the fallback (source) footprint dimensions also exceed
            `self.config.maxParentFootprintPixels`.

        Returns
        -------
        exposurePsfs : `list` [`tuple`]
            A list of `multiprofit.objects.Exposure`, `multiprofit.objects.PSF` pairs.
        footprint :
        bbox :
        cen_hst :
        """
        if footprint is not None:
            if footprint.getBBox().getArea() > self.config.maxParentFootprintPixels:
                footprint = None
        if footprint is None:
            footprint = source.getFootprint()
        if self.config.useSpans:
            spans = footprint.getSpans()
        fit_hst = self.config.fitHstCosmos
        bbox = footprint.getBBox()
        area = bbox.getArea()
        if failOnLargeFootprint and (area > self.config.maxParentFootprintPixels):
            raise RuntimeError(f'Source footprint (fallback) area={area} pix exceeds '
                               f'max={self.config.maxParentFootprintPixels}')
        elif not (area > 0):
            raise RuntimeError(f'Source bbox={bbox} has area={area} !>0')
        center = bbox.getCenter()
        # TODO: Implement multi-object fitting/deblending
        # peaks = footprint.getPeaks()
        # nPeaks = len(peaks)
        # isSingle = nPeaks == 1
        exposurePsfs = []
        # TODO: Check total flux first
        if fit_hst:
            wcs_src = next(iter(exposures.values())).getWcs()
            corners, cens = cutout.get_corners_src(source, wcs_src)
            exposure, cen_hst, psf = cutout.get_exposure_cutout_HST(
                corners, cens, extras, get_inv_var=True, get_psf=True)
            if np.sum(exposure.image > 0) == 0:
                raise RuntimeError('HST cutout has zero positive pixels')
            exposurePsfs.append((exposure, psf))
        else:
            for noiseReplacer, (band, exposure) in zip(extras, exposures.items()):
                if not self.config.deblendFromDeblendedFits:
                    noiseReplacer.insertSource(source.getId())
                bitmask = 0
                mask = exposure.mask.subset(bbox)
                for bitname in self.mask_names_zero:
                    bitval = mask.getPlaneBitMask(bitname)
                    bitmask |= bitval

                err = np.sqrt(1. / np.float64(exposure.variance.subset(bbox).array))
                mask = (mask.array & bitmask) != 0
                if self.config.useSpans:
                    mask_span = afwImage.Mask(bbox)
                    spans.setMask(mask_span, 2)
                    mask |= (mask_span.array == 0)

                err[mask] = 0

                img = np.float64(exposure.image.subset(bbox).array)
                exposure_mpf = mpfObj.Exposure(
                    band=band, image=img,
                    error_inverse=err, is_error_sigma=True, mask_inverse=~mask, use_mask_inverse=True
                )
                psf_mpf = mpfObj.PSF(
                    band, image=exposure.getPsf().computeKernelImage(center), engine="galsim")
                exposurePsfs.append((exposure_mpf, psf_mpf))

        return exposurePsfs, footprint, bbox, cen_hst if fit_hst else None

    @staticmethod
    def __addExtraField(extra, schema, prefix, name, doc, dtype=np.float64, unit=None, exists=False):
        """Add an extra field to a schema and store a reference to it by its short name.

        Parameters
        ----------
        extra : `dict` of `str`
            An input dictionary to store a reference to the new `Key` by its field name.
        schema : `lsst.afw.table.Schema`
            An existing table schema to add the field to.
        prefix : `str`
            A prefix for field full name.
        name : `str`
            A short name for the field, which serves as the key for `extra`.
        doc : `str`
            A brief documentation string for the field.
        unit : `str`
            A string convertible to an astropy unit.
        exists : `bool`
            Check if the field already exists and validate it instead of adding a new one.

        Returns
        -------
        No return. The new field is added to `schema` and a reference to it is stored in `extra`.
        """
        if doc is None:
            doc = ''
        if unit is None:
            unit = ''
        fullname = joinFilter('_', [prefix, name])
        if exists:
            item = schema.find(fullname)
            field = item.field
            if field.dtype != dtype or field.getUnits() != unit:
                raise RuntimeError(f'Existing field {field} has dtype {field.dtype}!={dtype} and/or units'
                                   f'{field.getUnits()}!={unit}')
            key = item.key
        else:
            key = schema.addField(fullname, type=dtype, doc=doc, units=unit)
        extra[name] = key

    def __addExtraFields(self, extra, schema, prefix=None, exists=False):
        """Add all extra fields for a given model based on `self.config` settings.

        Parameters
        ----------
        extra : `dict` of `str`
            An input dictionary to store reference to the new `Key`s by their field names.
        schema : `lsst.afw.table.Schema`
            An existing table schema to add the field to.
        prefix : `str`, optional
            A string such as a model name to prepend to each field name; default None.
        exists : `bool`
            Check if the fields already exist and validate them instead of adding new ones.

        Returns
        -------
        No return. The new fields are added to `schema` and reference to them are stored in `extra`.
        """
        if self.config.outputChisqred:
            self.__addExtraField(extra, schema, prefix, 'chisqred', 'reduced chi-squared of the best fit',
                                 exists=exists)
        if self.config.outputLogLikelihood:
            self.__addExtraField(extra, schema, prefix, 'loglike', 'log-likelihood of the best fit',
                                 exists=exists)
        if self.config.outputRuntime:
            self.__addExtraField(extra, schema, prefix, 'time', 'model runtime excluding setup', unit='s',
                                 exists=exists)
        self.__addExtraField(extra, schema, prefix, 'nEvalFunc', 'number of objective function evaluations',
                             exists=exists)
        self.__addExtraField(extra, schema, prefix, 'nEvalGrad', 'number of Jacobian evaluations',
                             exists=exists)

    @staticmethod
    def __fitModel(model, exposurePsfs, modeller=None, sources=None, resetPsfs=False, **kwargs):
        """Fit a model to sources in a series of exposures, initializing from the estimated moments or
        provided values.

        Parameters
        ----------
        model : `multiprofit.objects.Model`
            A MultiProFit model to fit.
        exposurePsfs : `iterable` [(`multiprofit.objects.Exposure`, `multiprofit.objects.PSF`)]
            An iterable of exposure-PSF pairs to fit.
        modeller : `multiprofit.objects.Modeller`, optional
            A MultiProFit modeller to use to fit the model; default creates a new modeller.
        sources : `list` [`dict`]
            A list of sources specified as a dict of values by parameter name, with flux a dict by filter.
        resetPsfs : `bool`, optional
            Whether to set the PSFs to None and thus fit a model with no PSF convolution.
        kwargs
            Additional keyword arguments to pass to multiprofit.fitutils.fit_model.

        Returns
        -------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_model, if no error occurs.
        """
        if sources is None:
            sources = [{}]
        # Set the PSFs to None in each exposure to skip convolution
        exposures_no_psf = {}
        for exposure, _ in exposurePsfs:
            if resetPsfs:
                exposure.psf = None
            exposures_no_psf[exposure.band] = [exposure]
        model.data.exposures = exposures_no_psf
        params_free = [src.get_parameters(free=True) for src in model.sources]
        n_sources = len(sources)
        if n_sources == 1 and ('sigma_x' not in sources[0]):
            fluxes, sources[0], _, _, _ = mpfFit.get_init_from_moments(
                exposurePsfs, cenx=sources[0].get('cenx', 0), ceny=sources[0].get('ceny', 0))
            sources[0]['flux'] = fluxes
        if n_sources != len(params_free):
            raise ValueError(f'len(sources)={n_sources} != len(model.sources)={len(model.sources)}')
        for values, params in zip(sources, params_free):
            fluxes = values.get('flux', {})
            for param in params:
                value = fluxes.get(param.band, 1) if isinstance(param, mpfObj.FluxParameter) else \
                    values.get(param.name)
                if param.name.startswith('cen'):
                    param.limits.upper = exposurePsfs[0][0].image.shape[param.name[-1] == 'x']
                if value is not None:
                    param.set_value(value, transformed=False)
        result, _ = mpfFit.fit_model(model=model, modeller=modeller, **kwargs)
        return result

    @pipeBase.timeMethod
    def __fitSource(self, source, exposures, extras, children=None, printTrace=False, plot=False,
                    footprint=None, failOnLargeFootprint=False,
                    usePriorShapeDefault=False, priorCentroidSigma=np.Inf, mag_prior=None,
                    backgroundPriors=None, fields=None, kwargs_fit=None, **kwargs):
        """Fit a single deblended source with MultiProFit.

        Parameters
        ----------
        source : `lsst.afw.table.SourceRecord`
            A deblended source to fit.
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        extras : iterable of `lsst.meas.base.NoiseReplacer` or `multiprofit.object.Exposure`
            An iterable of NoiseReplacers that will insert the source into every exposure, or a tuple of
            HST exposures if fitting HST data.
        printTrace : `bool`, optional
            Whether to print the traceback in case of an error; default False.
        plot : `bool`, optional
            Whether to generate a plot window with the final output; default False.
        footprint : `lsst.afw.detection.Footprint`, optional
            The footprint to fit within. Default source.getFootprint().
        failOnLargeFootprint : `bool`, optional
            Whether to return a failure if the fallback (source) footprint dimensions also exceed
            `self.config.maxParentFootprintPixels`.
        usePriorShapeDefault : `bool`, optional
            Whether to use the default MultiProFit shape prior.
        priorCentroidSigma : `float`, optional
            The sigma on the Gaussian centroid prior. Non-positive-finite values disable the prior.
        mag_prior: `float`, optional
            The magnitude for setting magnitude-dependent priors. A None value disables such priors.
        backgroundPriors: `dict` [`str`, `tuple`], optional
            Dict by band of 2-element tuple containing background level prior mean and sigma.
        kwargs_fit: `dict`, optional
            Dictionary of additonal kwargs to pass to
        **kwargs
            Additional keyword arguments to pass to `multiprofit.fitutils.fit_galaxy_exposures`.

        Returns
        -------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_galaxy_exposures, if no error occurs.
        error : `Exception`
            The first exception encountered while fitting, if any.
        noiseReplaced : `bool`
            Whether the method inserted the source using the provided noiseReplacers.
        """
        has_children = children is not None
        deblend_no_init = self.config.deblend and has_children
        if deblend_no_init and len(self.modelSpecs) > 0:
            raise RuntimeError("Can only deblend with gausspx-nopsf model")
        fit_hst = self.config.fitHstCosmos
        pixel_scale = pixel_scale_hst if fit_hst else pixel_scale_hsc
        noiseReplaced = False

        try:
            exposurePsfs, footprint, bbox, cen_hst = self._getExposurePsfs(
                exposures, source, extras, footprint=footprint, failOnLargeFootprint=failOnLargeFootprint
            )
            if not self.config.deblendFromDeblendedFits:
                noiseReplaced = True
            cen_src = source.getCentroid()
            begin = bbox.getBegin()
            cens = cen_hst if fit_hst else cen_src - begin
            if self.config.fitGaussian:
                rho_min, rho_max = -0.9, 0.9
                if fit_hst and deblend_no_init:
                    # Use wcs_hst/src instead
                    # wcs_hst = exposure.meta['wcs']
                    scale_x = pixel_scale_hsc / pixel_scale_hst
                    scales = np.array([scale_x, scale_x])
                sources = [{}]
                if deblend_no_init or self.config.useSdssShape:
                    for child in (children if deblend_no_init else [source]):
                        cen_child = ((cen_hst + scales*(child.getCentroid() - cen_src)) if fit_hst else (
                                child.getCentroid() - begin)) if deblend_no_init else cens
                        ellipse = {'cenx': cen_child[0], 'ceny': cen_child[1]}
                        cov = child['base_SdssShape_xy']
                        if np.isfinite(cov):
                            sigma_x, sigma_y = (np.sqrt(child[f'base_SdssShape_{ax}']) for ax in ['xx', 'yy'])
                            if sigma_x > 0 and sigma_y > 0:
                                ellipse['rho'] = np.clip(cov / (sigma_x * sigma_y), rho_min, rho_max)
                                ellipse['sigma_x'], ellipse['sigma_y'] = sigma_x, sigma_y
                        sources.append(ellipse)

            bands = [item[0].band for item in exposurePsfs]
            params_prior = {}
            if usePriorShapeDefault:
                size_mean, size_stddev = get_hst_size_prior(mag_prior if np.isfinite(mag_prior) else np.Inf)
                size_mean_stddev = (size_mean - np.log10(pixel_scale), size_stddev)
                params_prior['shape'] = {
                    True: {
                        'size_mean_std': (self.config.psfHwhmShrink, self.config.gaussianSizePriorSigma),
                        'size_log10': False,
                        'axrat_params': (-0.1, 0.5, 1.1),
                    },
                    False: {
                        'size_mean_std': size_mean_stddev,
                        'size_log10': True,
                        'axrat_params': (-0.3, 0.2, 1.2),
                    }
                }
            if np.isfinite(priorCentroidSigma):
                if not priorCentroidSigma > 0:
                    raise ValueError(f'Invalid priorCentroidSigma={priorCentroidSigma} !>0')
                for coord in ('cenx', 'ceny'):
                    params_prior[coord] = {
                        True: {'stddev': priorCentroidSigma},
                        False: {'stddev': priorCentroidSigma},
                    }
            if backgroundPriors:
                priors_background = {}
                for idx_band, (band, (bg_mean, bg_sigma)) in enumerate(backgroundPriors.items()):
                    if bg_sigma is None:
                        exposure = exposurePsfs[idx_band][0]
                        if exposure.band != band:
                            raise RuntimeError(f'exposure.band={exposure.band}!=band={band} setting bg prior')
                        err = exposure.error_inverse
                        pix_good = err > 0
                        bg_sigma = np.nanmedian(np.sqrt(1/err[pix_good]))/np.sqrt(np.sum(pix_good))
                    if not bg_sigma > 0:
                        raise ValueError(f'Non-positive bg_sigma={bg_sigma}')
                    priors_background[band] = {'mean': bg_mean, 'stddev': bg_sigma}
                params_prior['background'] = priors_background

            deblend_from_fits = self.config.deblendFromDeblendedFits and children is not None
            if not deblend_from_fits:
                skip_fit = self.config.plotOnly is True
                results = mpfFit.fit_galaxy_exposures(
                    exposurePsfs, bands, self.modelSpecs, plot=plot, print_exception=True,
                    cenx=cens[0], ceny=cens[1], fit_background=self.config.fitBackground,
                    psf_shrink=self.config.psfHwhmShrink, prior_specs=params_prior,
                    skip_fit=skip_fit, skip_fit_psf=skip_fit, **kwargs
                )
                if (not self.config.plotOnly) and self.config.fitGaussian:
                    n_sources = len(sources)
                    name_modeltype = f'gausspx{self.postfix_nopsf}'
                    name_model_full = f'{name_modeltype}_{n_sources}'
                    if name_model_full in self.models:
                        model = self.models[name_model_full]
                    else:
                        filters = [x[0].band for x in exposurePsfs]
                        model = mpfFit.get_model(
                            {band: 1 for band in filters}, "gaussian:1", (1, 1), slopes=[0.5],
                            engine='galsim',
                            engineopts={
                                'use_fast_gauss': True,
                                'drawmethod': mpfObj.draw_method_pixel['galsim']
                            },
                            name_model=name_model_full
                        )
                        for _ in range(n_sources-1):
                            model.sources.append(copy.deepcopy(model.sources[0]))
                        # Probably don't need to cache anything more than this
                        if n_sources < 10:
                            self.models[name_model_full] = model
                    self.modeller.model = model
                    result = self.__fitModel(model, exposurePsfs, modeller=self.modeller, sources=sources,
                                             resetPsfs=True, plot=plot and len(self.modelSpecs) == 0)
                    results['fits']['galsim'][name_modeltype] = {'fits': [result], 'modeltype': 'gaussian:1'}
                    results['models']['gaussian:1'] = model
            else:
                results = {}
                skip_fit_psf = True
                # Check if any PSF fit failed (e.g. because a large blend was skipped entirely) and redo if so
                values_init_psf = self.modelSpecs[0]['values_init_psf']
                for band, params_psf in values_init_psf.items():
                    for name_param, value_param in params_psf:
                        if not np.isfinite(value_param):
                            skip_fit_psf = False
                            break
                # Check if a model fit failed (e.g. as above) and zero init params
                # This will need to be updated if background models become non-linear
                for modelSpec in self.modelSpecs:
                    values_init_model = modelSpec['values_init']
                    if not skip_fit_psf:
                        del modelSpec['values_init_psf']
                    for name_param, value_param in values_init_model:
                        if not np.isfinite(value_param):
                            values_init_model = [
                                (name_p, self.__getParamValueDefault(name_p))
                                for name_p, _ in values_init_model
                            ]
                            modelSpec['values_init'] = values_init_model
                            break
                mpfFit.fit_galaxy_exposures(
                    exposurePsfs, bands, [self.modelSpecs[0]], plot=False, print_exception=True,
                    cenx=cens[0], ceny=cens[1], fit_background=self.config.fitBackground,
                    psf_shrink=self.config.psfHwhmShrink, skip_fit=True, skip_fit_psf=skip_fit_psf, **kwargs
                )
                fields_base = fields["base"]
                if kwargs_fit is None:
                    kwargs_fit = {}
                values_init = {}
                # Iterate through each model and do a simultaneous re-fit
                for modelSpec in self.modelSpecsDict.values():
                    name_modeltype = modelSpec['model']
                    name_model = modelSpec['name']
                    fields_base_model = fields_base[name_model]

                    inittype_split = modelSpec['inittype'].split(';')
                    # TODO: Come up with a better way to determine if initializing from models
                    if inittype_split[0] in self.modelSpecsDict:
                        modelSpecs = [self.modelSpecsDict[name_init] for name_init in inittype_split]
                        modelSpecs.append(modelSpec)
                    else:
                        modelSpecs = [modelSpec]

                    _, models = mpfFit.fit_galaxy(
                        exposurePsfs, modelSpecs, fit_background=self.config.fitBackground, skip_fit=True
                    )
                    model = models[name_modeltype]
                    sources_bg = []
                    for source in model.sources:
                        if isinstance(source, mpfObj.Background):
                            if not self.config.fitBackground:
                                raise TypeError("Model should not have Background objects with "
                                                "fitBackground=False")
                            else:
                                sources_bg.append(source)
                    model.sources = []
                    params_free = {}
                    values_init[name_model] = {}

                    for idx_c, child in enumerate(children):
                        is_good = True
                        values_init_model = []
                        for name_field, key in fields_base_model:
                            value_field = child[key]
                            if not np.isfinite(value_field):
                                is_good = False
                                values_init_model = None
                                break
                            values_init_model.append((name_field, value_field))
                        values_init[name_model][idx_c] = values_init_model
                        # Re-initialize the child model and transfer its sources to the full model
                        if is_good:
                            exposurePsfs_c, _, bbox_c, *_ = self._getExposurePsfs(
                                exposures, child, extras,
                                footprint=source.getFootprint() if self.config.useParentFootprint else None,
                                failOnLargeFootprint=failOnLargeFootprint
                            )
                            exposurePsfs_input = []
                            for (exposure_p, psf_p), (exposure_c, psf_c) in zip(exposurePsfs, exposurePsfs_c):
                                exposure_c.psf = exposure_p.psf
                                exposurePsfs_input.append((exposure_c, psf_p))

                            for specs in modelSpecs:
                                specs['values_init'] = values_init[specs['name']][idx_c]

                            _, models = mpfFit.fit_galaxy(
                                exposurePsfs_input, modelSpecs, fit_background=self.config.fitBackground,
                                skip_fit=True
                            )
                            # Another double entendre; it never ends.
                            model_child = models[name_modeltype]
                            bbox_min = bbox.getMin()
                            bbox_min_c = bbox_c.getMin()
                            # All centroid parameters need adjusting, free or not
                            if bbox_min != bbox_min_c:
                                offset_x, offset_y = bbox_min_c - bbox_min
                                params_all = model_child.get_parameters(free=True, fixed=True)
                                params_free_i = []
                                for param in params_all:
                                    if param.name.startswith('cen'):
                                        is_x = param.name[3] == 'x'
                                        param.limits.upper = bbox.width if is_x else bbox.height
                                        value = param.get_value(transformed=False)
                                        param.set_value(value + (offset_x if is_x else offset_y),
                                                        transformed=False)
                                    if not param.fixed:
                                        params_free_i.append(param)
                                params_free[child] = params_free_i
                            else:
                                params_free[child] = model_child.get_parameters(free=True, fixed=False)

                            # Ensure no duplicate background sources added
                            for source in models[name_modeltype].sources:
                                if isinstance(source, mpfObj.Background):
                                    if not self.config.fitBackground:
                                        raise TypeError("Model should not have Background objects with "
                                                        "fitBackground=False")
                                else:
                                    model.sources.append(source)
                        else:
                            params_free[child] = None

                    results[name_model], _ = mpfFit.fit_model(model, plot=plot, kwargs_fit=kwargs_fit)

                    for child, params_free_c in params_free.items():
                        if params_free_c is not None:
                            if len(params_free_c) != len(fields_base_model):
                                raise RuntimeError(
                                    f'len(params_free_c={params_free_c})={len(params_free_c)} != '
                                    f'len(fields_base_model={fields_base_model})={len(fields_base_model)}'
                                )
                            for param, (_, key) in zip(params_free_c, fields_base_model):
                                child[key] = param.get_value(transformed=False)
                results = {'fits': {'galsim': results}}
            if plot:
                plt.show()
            return results, None, noiseReplaced
        except Exception as e:
            if printTrace:
                traceback.print_exc()
            if plot:
                n_exposures = len(exposures)
                if n_exposures > 1:
                    fig, axes = plt.subplots(1, n_exposures)
                    for idx, (band, exposure) in enumerate(exposures.items()):
                        axes[idx].imshow(exposure.image)
                        axes[idx].set_title(f'{band} [{idx}/{n_exposures}]')
                else:
                    plt.figure()
                    band, exposure = list(exposures.items())[0]
                    plt.imshow(exposure.image)
                    plt.title(band)
            return None, e, noiseReplaced

    def __getCatalog(self, filters, results, sources):
        """Get a catalog and a dict containing the keys of extra fields to enter for each row.

        Parameters
        ----------
        filters : iterable of `str`
            Names of bandpass filters for filter-dependent fields.
        results : `dict`
            Results structure as returned by `__fitSource`.
        sources : `iterable` of `lsst.afw.table.BaseRecord`

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            A new SourceCatalog with extra fields.
        fields : `dict` [`str`, `dict`]
            A dict of dicts, keyed by the field type. The values may contain further nested dicts e.g. those
            keyed by filter for PSF fit-related fields.
        """
        resume = self.config.resume
        schema = sources.schema
        if not resume:
            mapper = MultiProFitTask._getMapper(schema)
        keys_extra = {
            'runtimeKey': {'name': f'{self.prefix}{self.keynames["runtimeKey"]}', 'dtype': np.float64,
                           'doc': 'Source fit CPU runtime', 'unit': 's'},
            'failFlagKey': {'name': f'{self.prefix}{self.keynames["failFlagKey"]}', 'dtype': 'Flag',
                            'doc': 'MultiProFit general failure flag'},
        }
        fields_attr = {}
        for name_attr, specs in keys_extra.items():
            name_field = specs.get('name', f'{self.prefix}{name_attr}')
            self.__addExtraField(
                fields_attr, schema if resume else mapper.editOutputSchema(),
                prefix=None, name=name_field, doc=specs.get('doc', ''),
                dtype=specs.get('dtype'), unit=specs.get('unit'), exists=resume)
            setattr(self, name_attr, fields_attr[name_field])
        if not resume:
            schema = mapper.getOutputSchema()

        fields = {key: {} for key in ["base", "base_extra", "psf", "psf_extra", "measmodel"]}
        # Set up the fields for PSF fits, which are independent per filter
        for idxBand, band in enumerate(filters):
            prefix = f'{self.prefix}{self.prefix_psf}{band}'
            resultsPsf = results['psfs'][idxBand]['galsim']
            fields["psf"][band] = {}
            fields["psf_extra"][band] = defaultdictNested()
            for name, fit in resultsPsf.items():
                fit = fit['fit']
                namesAdded = defaultdict(int)
                keyList = []
                for nameParam in fit['name_params']:
                    namesAdded[nameParam] += 1
                    fullname, doc, unit = self.__getParamFieldInfo(
                        f'{nameParam}{"Frac" if nameParam is "flux" else ""}',
                        f'{prefix}_c{namesAdded[nameParam]}_')
                    if resume:
                        key = schema.find(fullname).key
                    else:
                        key = schema.addField(fullname, doc=doc, units=unit, type=np.float64)
                    keyList.append(key)
                fields["psf"][band][name] = keyList
                self.__addExtraFields(fields["psf_extra"][band][name], schema, prefix, exists=resume)

        # Setup field names for source fits, which may have fluxes in multiple filters if run in multi-band.
        # Either way, flux parameters should contain a filter name.
        for name, result in results['fits']['galsim'].items():
            prefix = f'{self.prefix}{name}'
            fit = result['fits'][0]
            namesAdded = defaultdict(int)
            keyList = []
            bands = [f'{x.band}_' if hasattr(x, 'band') else '' for x, fixed in zip(
                fit['params'], fit['params_allfixed']) if not fixed]
            for nameParam, postfix in zip(fit['name_params'], bands):
                nameParamFull = f'{nameParam}{postfix}'
                namesAdded[nameParamFull] += 1
                fullname, doc, unit = self.__getParamFieldInfo(
                    nameParam, f'{prefix}_c{namesAdded[nameParamFull]}_{postfix}')
                if resume:
                    key = schema.find(fullname).key
                else:
                    key = schema.addField(fullname, doc=doc, units=unit, type=np.float64)
                keyList.append(key)
            fields["base"][name] = keyList
            fields["base_extra"][name] = defaultdictNested()
            self.__addExtraFields(fields["base_extra"][name], schema, prefix, exists=resume)

        if self.config.computeMeasModelfitLikelihood:
            for name in self.meas_modelfit_models:
                self.__addExtraField(fields["measmodel"], schema, f"{self.prefix}measmodel_like", name,
                                     f'MultiProFit log-likelihood for meas_modelfit {name} model',
                                     exists=resume)
        if not resume:
            catalog = afwTable.SourceCatalog(schema)
            catalog.extend(sources, mapper=mapper)
        return catalog, fields

    @staticmethod
    def __getFieldParamName(field):
        """ Get the standard MultiProFit parameter name for a given catalog field name.

        Parameters
        ----------
        field : `str`
            The name of the catalog field.

        Returns
        -------
        name : `str`
            The standard MultiProFit parameter name, as in
        """
        field_split = field.split('_')
        final = field_split[-1]
        if final.endswith('Flux') or final.endswith('Flux'):
            return 'flux'
        elif final == 'x' or final == 'y':
            if (len(field_split) > 1) and (field_split[-2] == 'sigma'):
                return f'sigma_{final}'
        return final

    @staticmethod
    def __getParamFieldInfo(nameParam, prefix=None):
        """Return standard information about a MultiProFit parameter by name.

        Parameters
        ----------
        nameParam : `str`
            The name of the parameter.
        prefix : `str`
            A prefix for the full name of the parameter; default None ('').

        Returns
        -------
        name_full : `str`
            The full name of the parameter including prefix, remapping 'flux' to 'instFlux'.
        doc : `str`
            The default docstring for the parameter, if any; '' otherwise.
        unit : `str`
            The default unit of the parameter, if any; '' otherwise.
        """
        name_full = f'{prefix if prefix else ""}{"instFlux" if nameParam is "flux" else nameParam}'
        doc, unit = MultiProFitTask.params_multiprofit.get(nameParam, ('', ''))
        return name_full, doc, unit

    @staticmethod
    def __getParamValueDefault(name_param):
        """Get a reasonable default value for a parameter by its name.

        Parameters
        ----------
        name_param : `str`:
            The name of the parameter.

        Returns
        -------
        value : `default`
            The MultiProFit-specified default value.
        """
        return mpfFit.get_param_default(
            MultiProFitTask.__getFieldParamName(name_param),
            return_value=True,
            is_value_transformed=True
        )

    @staticmethod
    def __setExtraField(extra, row, fit, name, nameFit=None, index=None):
        """Set the value of an extra field for a given row.

        Parameters
        ----------
        extra : container [`str`]
            An input container permitting retrieval of values by string keys.
        row : container [`str`]
            An output container permitting assignment by string keys.
        fit : `dict` [`str`]
            A fit result containing a value for `name` or `nameFit`.
        name : `str`
            The name of the field in `row`.
        nameFit : `str`, optional
            The name of the field in `fit`; default `name`.
        index : `int`, optional
            The index of the value in `fit`, if it is not scalar but is indexable. Ignored if None (default).
        Returns
        -------
        None
        """
        if nameFit is None:
            nameFit = name
        value = fit[nameFit]
        if index is not None:
            value = value[index]
        row[extra[name]] = value

    def _parseCatalogFields(self, catalog, validate_psf=True, add_keys=False):
        """ Get the keys for MultiProFit fields in an existing catalog in a dict by type.

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            The catalog to parse.
        validate_psf : `bool`, optional
            Whether to validate the retrieved PSF fields against those expected given config params.
        add_keys: `bool`, optional
            Whether to assign keys stored individually as attributes.

        Returns
        -------
        fields_mpf: `dict` [`str`, `list` [`str`]]
            A dictionary containing a list of parameter tuples, keyed by type. Each tuple contains:
            - name [`str`] : the short field name
            - key [] : the catalog (schema) field key

        Notes
        -----
        The return value is similar to but not identical to that from __getCatalog.
        """
        fields_mpf = {f'{pre}{post}': defaultdict(list) for pre in ('base', 'psf') for post in ('', '_extra')}
        prefixes = (self.prefix, f'{self.prefix}{self.prefix_psf}')
        prefix_psf = prefixes[True]
        schema = catalog.schema

        for name_field in schema.getOrderedNames():
            if name_field.startswith(self.prefix):
                is_psf = name_field.startswith(prefix_psf)
                prefix = prefixes[is_psf]
                name_field = name_field.split(prefix, 1)[1]
                key, name_field = name_field.split('_', 1)
                suffix = name_field.split('_')[-1]
                type_field = "psf" if is_psf else "base"
                is_extra = suffix in self.params_extra
                if is_extra:
                    type_field = f"{type_field}_extra"
                key_cat = schema.find(f'{prefix}{key}_{name_field}').key
                fields_mpf[type_field][key].append((name_field, key_cat))

        if validate_psf:
            fields_psf = ['c1_cenx', 'c1_ceny']
            params_shape = ('sigma_x', 'sigma_y', 'rho')
            for idx_comp in range(1, 1 + self.config.gaussianOrderPsf):
                if idx_comp != self.config.gaussianOrderPsf:
                    fields_psf.append(f'c{idx_comp}_fluxFrac')
                fields_psf.extend([f'c{idx_comp}_{name_param}' for name_param in params_shape])
            for band, fields_found in fields_mpf["psf"].items():
                fields_found_names = [field[0] for field in fields_found]
                if fields_found_names != fields_psf:
                    raise RuntimeError(f'PSF fields_found={fields_found_names} != expected fields_psf='
                                       f'{fields_psf}')
        if add_keys:
            for name_attr, name_field in self.keynames.items():
                setattr(self, name_attr, schema.find(f'{self.prefix}{name_field}').key)

        return fields_mpf

    def __setExtraFields(self, extra, row, fit):
        """Set the values of the extra fields specified by self.config parameters.

        Parameters
        ----------
        extra : container [`str`]
            An input container permitting retrieval of values by string keys.
        row : container [`str`]
            An output container permitting assignment by string keys.
        fit : `dict` [`str`]
            A fit result containing the required values.

        Returns
        -------
        None
        """
        if self.config.outputChisqred:
            self.__setExtraField(extra, row, fit, 'chisqred')
        if self.config.outputLogLikelihood:
            self.__setExtraField(extra, row, fit, 'loglike', nameFit='likelihood', index=0)
        if self.config.outputRuntime:
            self.__setExtraField(extra, row, fit, 'time')
        self.__setExtraField(extra, row, fit, 'nEvalFunc', nameFit='n_eval_func')
        self.__setExtraField(extra, row, fit, 'nEvalGrad', nameFit='n_eval_grad')

    def __setFieldsSource(self, results, fieldsBase, fieldsExtra, row):
        """Set fields for a source's fit parameters.

        Parameters
        ----------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_galaxy_exposures, if no error occurs.
        fieldsBase : `dict` [`str`, `dict` [`str`]]
            A dict of dicts of field keys by name for base fields (i.e. fit parameters), keyed by model name.
        fieldsExtra : `dict` [`str`, `dict` [`str`]]
            A dict of dicts of field keys by name for extra fields, keyed by model name.
        row : container [`str`]
            An output container permitting assignment by string keys.

        Returns
        -------
        None
        """
        for idxfit, (name, result) in enumerate(results['fits']['galsim'].items()):
            result_fit = result.get('fits', None) if hasattr(result, 'get') else None
            if result_fit is not None:
                fit = result_fit[0]
                values = [x for x, fixed in zip(fit['params_bestall'], fit['params_allfixed'])
                          if not fixed]
                for value, key in zip(values, fieldsBase[name]):
                    row[key] = value
                self.__setExtraFields(fieldsExtra[name], row, fit)

    def __setFieldsPsf(self, results, fieldsBase, fieldsExtra, row, filters):
        """Set fields for a source's PSF fit parameters.

        Parameters
        ----------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_galaxy_exposures, if no error occurs.
        fieldsBase : `dict` [`str`, `dict` [`str`]]
            A dict of dicts of field keys by name for base fields (i.e. fit parameters), keyed by filter name.
        fieldsExtra : `dict` [`str`, `dict` [`str`]]
            A dict of dicts of field keys by name for extra fields, keyed by filter name.
        row : container [`str`]
            An output container permitting assignment by string keys.
        filters : iterable of `str`
            Names of bandpass filters that a PSF has been fit for.

        Returns
        -------
        None
        """
        results_psfs = results.get('psfs', None)
        if results_psfs:
            for idxBand, band in enumerate(filters):
                resultsPsf = results_psfs[idxBand]['galsim']
                for name, fit in resultsPsf.items():
                    fit = fit['fit']
                    values = [x for x, fixed in zip(fit['params_bestall'], fit['params_allfixed'])
                              if not fixed]
                    for value, key in zip(values, fieldsBase[band][name]):
                        row[key] = value
                    self.__setExtraFields(fieldsExtra[band][name], row, fit)

    def __setFieldsMeasmodel(self, exposures, model, source, fieldsMeasmodel, row):
        """Set fields for a source's meas_modelfit-derived fields, including likelihoods for the
        meas_modelfit models as computed by MultiProFit.

        Parameters
        ----------
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        model : `multiprofit.objects.Model`
            A MultiProFit model to compute likelihoods with.
        source : `lsst.afw.table.SourceRecord`
            A deblended source to build meas_modelfit models for.
        fieldsMeasmodel : `dict` [`str`, container [`str`]]
            A dict of output fields, keyed by model name.
        row : container [`str`]
            An output container permitting assignment by string keys.

        Returns
        -------
        None
        """
        configMeasModelfit = CModelConfig()
        measmodels = {key: {} for key in self.meas_modelfit_models}
        for band, exposure in exposures.items():
            _, measmodels['dev'][band], measmodels['exp'][band], measmodels['cmodel'][band] = \
                buildCModelImages(exposure, source, configMeasModelfit)
        # Set the values of meas_modelfit model likelihood fields
        for measmodel_type, measmodel_images in measmodels.items():
            likelihood = 0
            for band, exposure in exposures.items():
                likelihood += model.get_exposure_likelihood(
                    model.data.exposures[band][0], measmodel_images[band].array)[0]
            likelihood = {measmodel_type: likelihood}
            self.__setExtraField(fieldsMeasmodel, row, likelihood, measmodel_type)

    def __setRow(self, filters, results, fields, row, exposures, source, runtime=0):
        """Set all necessary field values for a given source's row.

        Parameters
        ----------
        filters : iterable of `str`
            Names of bandpass filters for filter-dependent fields.
        results : `dict`
            The results returned by multiprofit.fitutils.fit_galaxy_exposures, if no error occurs.
        fields : `dict` [`str`, `dict`]
            A dict of dicts of field keys by name for base fields (i.e. fit parameters), keyed by filter name.
        row : container [`str`]
            An output container permitting assignment by string keys.
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        source : `lsst.afw.table.SourceRecord`
            A deblended source to build meas_modelfit models for.
        runtime : `float`; optional
            The CPU time spent fitting the source, in seconds; default zero.
        """
        self.__setFieldsPsf(results, fields["psf"], fields["psf_extra"], row, filters)
        self.__setFieldsSource(results, fields["base"], fields["base_extra"], row)
        if self.config.computeMeasModelfitLikelihood:
            model = results['models'][self.modelSpecs[0]["model"]]
            self.__setFieldsMeasmodel(exposures, model, source, fields["measmodel"], row)
        row[self.runtimeKey] = runtime

    def fit(self, data, idx_begin=0, idx_end=np.Inf, logger=None, printTrace=False,
            plot=False, path_cosmos_galsim=None, sources=None, mags_prior=None,
            field_localbg='base_LocalBackground_instFlux', **kwargs):
        """Fit a catalog of sources with MultiProFit.

        Each source has its PSF fit with a configureable Gaussian mixture PSF model and then fits a
        sequence of different models, some of which can be configured to be initialized from previous fits.
        See `MultiProFitConfig` for more information on how to configure models.

        Plots can be generated on success (with MultiProFit defaults), or on failure, in which case only the
        images themselves are shown. Tracebacks are suppressed on failure by default.

        Parameters
        ----------
        data : `dict` [`str`, `dict` [`str`]]
            A dict of data for each filter, each containing:
            ``"exposures"``
                The exposure of that filter (`lsst.afw.image.Exposure`)
            ``"sources"``
                The catalog of sources to fit (`lsst.afw.table.SourceCatalog`)
        idx_begin : `int`
            The first index (row number) of the catalog to process.
        idx_end : `int`
            The last index (row number) of the catalog to process.
        logger : `logging.Logger`, optional
            A Logger to log output; default logging.getLogger(__name__).
        printTrace : `bool`, optional
            Whether to print the traceback in case of an error; default False.
        plot : `bool`, optional
            Whether to generate a plot window with the final output for each source; default False.
        path_cosmos_galsim : `str`, optional
            A file path to a directory containing real_galaxy_catalog_25.2.fits and
            real_galaxy_PSF_images_25.2_n[1-88].fits; required if config.fitHstCosmos is True.
            See https://zenodo.org/record/3242143.
        sources : `lsst.afw.table.SourceCatalog`, optional
            A source catalog to override filter-specific catalogs provided in `data`, e.g. deepCoadd_ref.
            Default None.
        mags_prior : array-like [`float`]
            Magnitudes to pass to any magnitude-dependent priors.
        field_localbg : `str`
            The name (prefix to _instFlux) of the field to read the local background level from.
        **kwargs
            Additional keyword arguments to pass to `__fitSource`.

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            A new catalog containing all of the fields from `sources` and those generated by MultiProFit.
        results : `dict`
            A results structure as returned by mpfFit.fit_galaxy_exposures() for the first successfully fit
            source.
        """
        # Set up a logger to suppress output for now
        if logger is None:
            logger = logging.getLogger(__name__)
        if (self.config.resume or self.config.deblendFromDeblendedFits) and \
                not os.path.isfile(self.config.filenameOut):
            raise FileNotFoundError(f"Can't resume or deblendFromDeblendedFits from non-existent file"
                                    f"={self.config.filenameOut}")
        if self.config.resume and self.config.deblendFromDeblendedFits and not os.path.isfile(
                self.config.filenameOutDeblend):
            raise FileNotFoundError(f"Can't resume deblendFromDeblendedFits from non-existent file"
                                    f"={self.config.filenameOutDeblend}")
        if self.config.plotOnly and not self.config.resume:
            raise ValueError("Can't set plotOnly=True without resume=True")
        filters = data.keys()
        exposures = {band: data[band]['exposure'] for band in filters}
        filenameOut = (self.config.filenameOut if not self.config.deblendFromDeblendedFits else
                       self.config.filenameOutDeblend)

        if sources is None:
            sources = data[list(filters)[0]]['sources']
        if self.config.fitHstCosmos:
            if path_cosmos_galsim is None:
                raise ValueError("Must specify path to COSMOS GalSim catalog if fitting HST images")
            tiles = cutout.get_tiles_HST_COSMOS()
            ra_corner, dec_corner = cutout.get_corners_exposure(next(iter(data.values()))['exposure'])
            extras = cutout.get_exposures_HST_COSMOS(ra_corner, dec_corner, tiles, path_cosmos_galsim)
            # TODO: Generalize this for e.g. CANDELS
            filters = [extras[0].band]
        elif not self.config.deblendFromDeblendedFits:
            extras = [rebuildNoiseReplacer(datum['exposure'], datum['sources']) for datum in data.values()]
        else:
            extras = [None] * len(data)
        timeInit = time.time()
        processTimeInit = time.process_time()
        addedFields = self.config.deblendFromDeblendedFits is True
        resultsReturn = None
        indicesFailed = {}
        toWrite = bool(filenameOut)
        nFit = 0
        numSources = len(sources)
        if idx_end > numSources:
            idx_end = numSources
        numSources = idx_end - idx_begin

        if self.config.fitGaussian:
            if len(filters) > 1:
                raise ValueError(f'Cannot fit Gaussian (no PSF) model with multiple filters ({filters})')
            self.models['gausspx_no_psf_1'] = mpfFit.get_model(
                {band: 1 for band in filters}, "gaussian:1", (1, 1), slopes=[0.5], engine='galsim',
                engineopts={'use_fast_gauss': True, 'drawmethod': mpfObj.draw_method_pixel['galsim']},
                name_model='gausspx_no_psf_1'
            )

        flags_failure = ['base_PixelFlags_flag_saturatedCenter']

        if self.config.skipDeblendTooManyPeaks:
            flags_failure.append('deblend_tooManyPeaks')
        backgroundPriorMultiplier = self.config.backgroundPriorMultiplier
        backgroundPriors = {}
        if backgroundPriorMultiplier is not None and np.isfinite(backgroundPriorMultiplier):
            if not backgroundPriorMultiplier > 0:
                raise ValueError(f'Invalid backgroundPriorMultiplier={backgroundPriorMultiplier} !>0')
            for band in data:
                backgroundPriors[band] = None
        catalog_in = afwTable.SourceCatalog.readFits(self.config.filenameOut) if (
            self.config.resume or self.config.deblendFromDeblendedFits) else (
            sources if self.config.bandMeasCatToCopyFrom is None else
            data[self.config.bandMeasCatToCopyFrom]['sources'])

        init_from_cat = self.config.plotOnly or self.config.deblendFromDeblendedFits
        fields = self._parseCatalogFields(catalog_in, add_keys=True) if init_from_cat else None

        catalog = catalog_in if (toWrite and self.config.deblendFromDeblendedFits) else None
        deblend = self.config.deblend or self.config.deblendFromDeblendedFits
        kwargs_fit = {'do_linear_only': self.config.deblendFromDeblendedFitsNonlinear == False}

        for idx in range(np.max([idx_begin, 0]), idx_end):
            src = sources[idx]
            results = None
            id_parent = src['parent']
            n_child = src['deblend_nChild']
            is_parent = n_child > 0
            runtime = 0
            noiseReplaced = False
            skipped, succeeded = False, False
            if self.config.deblendFromDeblendedFits and not is_parent:
                skipped = True
                error = f'deblendFromDeblendedFits and not is_parent'
            else:
                flags_failed = {flag: src[flag] for flag in flags_failure}
                skipped = any(flags_failed.values())
                if skipped:
                    error = f'{[key for key, fail in flags_failed.items() if fail]} flag(s) set'
            if not skipped:
                errors = []
                is_child = src['parent'] != 0
                isolated = not is_child and not is_parent

                if self.config.isolatedOnly and not isolated:
                    errors.append('not isolated')
                if not self.config.deblendFromDeblendedFits and (
                        is_parent and n_child > self.config.maxNChildParentFit):
                    errors.append(f'is_parent and n_child={n_child} > max={self.config.maxNChildParentFit}')
                if errors:
                    error = ' & '.join(errors)
                else:
                    children = None if (not is_parent or not deblend) else [
                        catalog_in[int(x)] for x in np.where(sources['parent'] == src['id'])[0]]
                    footprint = sources.find(id_parent).getFootprint() if (
                        self.config.useParentFootprint and is_child) else None

                    for band in backgroundPriors:
                        if self.config.usePriorBackgroundLocalEstimate:
                            cat_band = data[band]['sources']
                            bg_mean = 0. if cat_band[f'{field_localbg}_flag'] else \
                                cat_band[f'{field_localbg}_instFlux']
                            bg_sigma = backgroundPriorMultiplier * cat_band[f'{field_localbg}_instFluxErr']
                        else:
                            bg_mean, bg_sigma = 0, None
                        backgroundPriors[band] = (bg_mean, bg_sigma)

                    if init_from_cat:
                        values_init_psf = {}
                        row_in = catalog_in[idx]
                        for band, fields_band in fields['psf'].items():
                            values_init_psf[band] = [
                                (name_field, row_in[key]) for name_field, key in fields_band
                            ]
                        for modelspec in self.modelSpecs:
                            modelspec['values_init'] = [
                                (name_field, row_in[key])
                                for name_field, key in fields['base'][modelspec['name']]
                            ]
                            modelspec['values_init_psf'] = values_init_psf

                    results, error, noiseReplaced = self.__fitSource(
                        src, exposures, extras, children=children, printTrace=printTrace, plot=plot,
                        footprint=footprint, failOnLargeFootprint=is_parent,
                        usePriorShapeDefault=self.config.usePriorShapeDefault,
                        priorCentroidSigma=self.config.priorCentroidSigma,
                        sigma_min=np.max((1e-2, self.config.psfHwhmShrink)),
                        mag_prior=mags_prior[idx] if mags_prior is not None else None,
                        backgroundPriors=backgroundPriors,
                        denoise=self.config.estimateContiguousDenoisedMoments,
                        contiguous=self.config.estimateContiguousDenoisedMoments,
                        results=results, fields=fields, kwargs_fit=kwargs_fit,
                        **kwargs)
                    succeeded = error is None
                    runtime = (self.metadata["__fitSourceEndCpuTime"] -
                               self.metadata["__fitSourceStartCpuTime"])
            # Preserve the first successful result to return at the end
            if resultsReturn is None and succeeded:
                resultsReturn = results
            # Setup field names if necessary
            if not init_from_cat and not addedFields and succeeded:
                # If one of the models failed, we can't set up the catalog from it
                if all(['fits' in x for x in results['fits']['galsim'].values()]):
                    catalog, fields = self.__getCatalog(filters, results, catalog_in)
                    for idx_failed, runtime_failed in indicesFailed.items():
                        catalog[idx_failed][self.failFlagKey] = True
                        catalog[idx_failed][self.runtimeKey] = runtime_failed
                    addedFields = True
                else:
                    # Sadly this means that the successful models won't get written. Oh well.
                    error = "at least one model failed before catalog fields added from success"
            # Fill in field values if successful, or save just the runtime to enter later otherwise
            if not self.config.plotOnly:
                row = catalog[idx]
                row[self.failFlagKey] = succeeded
                if succeeded and addedFields and not (
                        (self.config.deblend and is_parent) or self.config.deblendFromDeblendedFits):
                    self.__setRow(filters, results, fields, row, exposures, src, runtime=runtime)
                else:
                    indicesFailed[idx] = runtime
            id_src = src.getId()
            # Returns the image to pure noise
            if noiseReplaced:
                for noiseReplacer in extras:
                    noiseReplacer.removeSource(id_src)
            errorMsg = '' if succeeded else f" {'skipped' if skipped else 'failed'}: {error}"
            # Log with a priority just above info, since MultiProFit itself will generate a lot of info
            #  logs per source.
            nFit += 1
            if not self.config.plotOnly:
                logger.log(
                    21, f"Fit src {idx} ({nFit}/{numSources}) id={src['id']} in {runtime:.3f}s "
                        f"(total time {time.time() - timeInit:.2f}s "
                        f"process_time {time.process_time() - processTimeInit:.2f}s)"
                        f"{errorMsg}")
                if toWrite and addedFields and (nFit % self.config.intervalOutput) == 0:
                    catalog.writeFits(filenameOut)
        if not self.config.plotOnly:
            if toWrite and addedFields and (nFit % self.config.intervalOutput) != 0:
                catalog.writeFits(filenameOut)
            # Return the exposures to their original state
            if not self.config.fitHstCosmos and not self.config.deblendFromDeblendedFits:
                for noiseReplacer in extras:
                    noiseReplacer.end()
        return catalog, resultsReturn

    @pipeBase.timeMethod
    def run(self, exposures, sources, **kwargs):
        """Run the MultiProFit task on a catalog of sources, fitting a dict of exposures keyed by filter.

        This function is currently a simple wrapper that calls self.fit().

        Parameters
        ----------
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        sources: `lsst.afw.table.SourceCatalog`
            A catalog containing deblended sources with footprints
        **kwargs
            Additional keyword arguments to pass to self.fit.

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            A new catalog containing all of the fields from `sources` and those generated by MultiProFit.
        results : `dict`
            A results structure as returned by mpfFit.fit_galaxy_exposures() for the first successfully fit
            source.
        """
        catalog, results = self.fit(exposures, sources, **kwargs)
        return catalog, results
