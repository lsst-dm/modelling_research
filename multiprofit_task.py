from collections import defaultdict
import logging
#from lsst.afw.image.exposure import MultibandExposure
import lsst.afw.table as afwTable
from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
from lsst.meas.modelfit.display import buildCModelImages
from lsst.meas.modelfit.cmodel.cmodelContinued import CModelConfig
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import multiprofit.fitutils as mpfFit
import multiprofit.objects as mpfObj
import numpy as np
import time
import traceback


class MultiProFitConfig(pexConfig.Config):
    """
    Configuration for the MultiProFit profile fitter.
    """
    computeMeasModelfitLikelihood = pexConfig.Field(dtype=bool, default=False,
                                                    doc="Whether to compute the log-likelihood of best-fit "
                                                        "meas_modelfit parameters for each model")
    fitCModel = pexConfig.Field(dtype=bool, default=True,
                                doc="Whether to perform a CModel (linear combo of exponential and "
                                    "deVaucouleurs) fit for each source; necessitates doing exp. + deV. fits")
    fitCModelExp = pexConfig.Field(dtype=bool, default=False,
                                   doc="Whether to perform an exponential fit with a fixed center (as "
                                       "CModel does in meas_modelfit) for each source")
    fitSersic = pexConfig.Field(dtype=bool, default=True, doc="Whether to perform a MG Sersic approximation "
                                                              "profile fit for each source")
    fitSersicFromCModel = pexConfig.Field(dtype=bool, default=False,
                                          doc="Whether to perform a MG Sersic approximation profile fit "
                                              "(initalized from previous exp./dev. fits) for each source")
    fitSersicAmplitude = pexConfig.Field(dtype=bool, default=True,
                                         doc="Whether to perform a linear fit of the Gaussian "
                                             "amplitudes for the MG Sersic approximation profile fit for "
                                             "each source; has no impact if fitSersic is False")
    gaussianOrderPsf = pexConfig.Field(dtype=int, default=2, doc="Number of Gaussians components for the PSF")
    gaussianOrderSersic = pexConfig.Field(dtype=int, default=8, doc="Number of Gaussians components for the "
                                                                    "MG Sersic approximation galaxy profile")
    outputChisqred = pexConfig.Field(dtype=bool, default=True, doc="Whether to save the reduced chi^2 of "
                                                                   "each model's best fit")
    outputLogLikelihood = pexConfig.Field(dtype=bool, default=True, doc="Whether to save the log likelihood "
                                                                        "of each model's best fit")
    outputRuntime = pexConfig.Field(dtype=bool, default=True, doc="Whether to save the runtime of each "
                                                                  "model")

    def getModelSpecs(self):
        """
        Get a list of dicts of model specifications for MultiProFit.
        :return: modelSpecs: list[dict] of model specifications
        """
        modelSpecs = []
        nameMG = f"mg{self.gaussianOrderSersic}"
        namePsfModel = f"gaussian:{self.gaussianOrderPsf}"
        nameSersicPrefix = f"mgsersic{self.gaussianOrderSersic}"
        nameSersicModel = f"{nameSersicPrefix}:1"
        nameSersicAmpModel = f"gaussian:{self.gaussianOrderSersic}+rscale:1"
        allParams = "cenx;ceny;nser;sigma_x;sigma_y;rscale;rho"
        if self.fitSersic:
            modelSpecs.append(
                dict(name=f"{nameMG}sermpx", model=nameSersicModel, fixedparams='', initparams="nser=1",
                     inittype="moments", psfmodel=namePsfModel, psfpixel="T")
            )
            if self.fitSersicAmplitude:
                modelSpecs.append(
                    dict(name=f"{nameMG}serapx", model=nameSersicAmpModel, fixedparams=allParams,
                         initparams="rho=inherit;rscale=modify", inittype=f"{nameMG}sermpx",
                         psfmodel=namePsfModel, psfpixel="T")
                )
        if self.fitCModel:
            modelSpecs.extend([
                dict(name="gausspx", model=nameSersicModel, fixedparams='nser', initparams="nser=0.5",
                     inittype="moments", psfmodel=namePsfModel, psfpixel="T"),
                dict(name=f"{nameMG}expgpx", model=nameSersicModel, fixedparams='nser', initparams="nser=1",
                     inittype="guessgauss2exp:gausspx", psfmodel=namePsfModel, psfpixel="T"),
                dict(name=f"{nameMG}devepx", model=nameSersicModel, fixedparams='nser', initparams="nser=4",
                     inittype=f"guessexp2dev:{nameMG}expgpx", psfmodel=namePsfModel, psfpixel="T"),
                dict(name=f"{nameMG}cmodelpx", model=f"{nameSersicPrefix}:2",
                     fixedparams="cenx;ceny;nser;sigma_x;sigma_y;rho", initparams="nser=4,1",
                     inittype=f"{nameMG}devepx;{nameMG}expgpx", psfmodel=namePsfModel, psfpixel="T"),
            ])
            if self.fitSersicFromCModel:
                modelSpecs.extend([
                    dict(name=f"{nameMG}sergpx", model=nameSersicModel, fixedparams='', initparams='',
                         inittype="gausspx", psfmodel=namePsfModel, psfpixel="T"),
                    dict(name=f"{nameMG}serbpx", model=nameSersicModel, fixedparams='', initparams='',
                         inittype="best", psfmodel=namePsfModel, psfpixel="T"),
                ])
                if self.fitSersicAmplitude:
                    modelSpecs.append(
                        dict(name=f"{nameMG}serbapx", model=nameSersicAmpModel, fixedparams=allParams,
                             initparams="rho=inherit;rscale=modify", inittype=f"{nameMG}sermpx",
                             psfmodel=namePsfModel, psfpixel="T")
                    )
        if self.fitCModelExp:
            modelSpecs.append(
                dict(name=f"{nameMG}expcmpx", model=nameSersicModel, fixedparams='cenx;ceny;nser',
                     initparams="nser=1", inittype="moments", psfmodel=namePsfModel, psfpixel="T")
            )
        return modelSpecs


def _defaultdictNested():
    return defaultdict(_defaultdictNested)


def _joinFilter(joint, items, exclude=None):
    """
    Join a list of strings, optionally excluding some.
    :param joint: string; pattern to join items
    :param items: list[string]; items to join
    :param exclude: string; pattern to exclude
    :return: string; Joined string
    """
    return joint.join(filter(exclude, items))


class MultiProFitTask(pipeBase.Task):
    """
    A task to run MultiProFit on a catalog with detections and heavy footprints, returning a new SimpleCatalog
    with additional measurements.
    """
    ConfigClass = MultiProFitConfig
    _DefaultName = "multiProFit"
    meas_modelfit_models = ("dev", "exp", "cmodel")

    def __init__(self, modelSpecs=None, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        if modelSpecs is None:
            modelSpecs = self.config.getModelSpecs()
        self.modelSpecs = modelSpecs
        self.schema = None

    def _getMapper(self, schema):
        mapper = afwTable.SchemaMapper(schema, True)
        mapper.addMinimalSchema(schema, True)
        mapper.editOutputSchema().disconnectAliases()
        self.runtimeKey = mapper.editOutputSchema().addField('multiprofit_time_total', type=np.float32,
                                                             doc='runtime in ms')
        self.failFlagKey = mapper.editOutputSchema().addField('multiprofit_fail_flag', type="Flag",
                                                              doc='generic MultiProFit failure flag')
        return mapper

    @staticmethod
    def __addExtraField(extra, schema, prefix, name, doc):
        """
        Add an extra field to a schema and store a reference to it by its short name
        :param extra: dict[str]; Dict to store new key
        :param schema: lsst.afw.table.Schema; Schema to add field to
        :param prefix: string; Prefix for field full name
        :param name: string; Short field name
        :param doc: string; Description of field
        :return: No return
        """
        if doc is None:
            doc = ''
        extra[name] = schema.addField(_joinFilter('_', [prefix, name]), type=np.float32, doc=doc)

    def __addExtraFields(self, extra, schema, prefix=None):
        """
        Add extra fields based on self.config settings.
        :param extra: dict[str]; Dict to store new key
        :param schema: lsst.afw.table.Schema; Schema to add field to
        :param prefix: string; Prefix for field full name
        :return: No return
        """
        if self.config.outputChisqred:
            self.__addExtraField(extra, schema, prefix, 'chisqred', 'reduced chi-squared of the best fit')
        if self.config.outputLogLikelihood:
            self.__addExtraField(extra, schema, prefix, 'loglike', 'log-likelihood of the best fit')
        if self.config.outputRuntime:
            self.__addExtraField(extra, schema, prefix, 'time', 'model runtime excluding setup')
        self.__addExtraField(extra, schema, prefix, 'nEvalFunc', 'number of objective function evaluations')
        self.__addExtraField(extra, schema, prefix, 'nEvalGrad', 'number of Jacobian evaluations')

    @pipeBase.timeMethod
    def __fitSource(self, src, noiseReplacers, exposures, filters, logger, printTrace=False, plot=False):
        results = None
        try:
            foot = src.getFootprint()
            bbox = foot.getBBox()
            center = bbox.getCenter()
            # TODO: Implement multi-object fitting/deblending
            # peaks = foot.getPeaks()
            # nPeaks = len(peaks)
            # isSingle = nPeaks == 1
            for noiseReplacer in noiseReplacers.values():
                noiseReplacer.insertSource(src.getId())
            exposurePsfs = []
            for band, exposure in exposures.items():
                # Check total flux first
                mpfExposure = mpfObj.Exposure(
                    band=band, image=np.float64(exposure.image.subset(bbox).array),
                    error_inverse=1 / np.float64(exposure.variance.subset(bbox).array),
                    is_error_sigma=False)
                mpfPsf = mpfObj.PSF(band, image=exposure.getPsf().computeKernelImage(center),
                                    engine="galsim")
                exposurePsfs.append((mpfExposure, mpfPsf))
            results = mpfFit.fit_galaxy_exposures(
                exposurePsfs, filters, self.modelSpecs, results=results, loggerPsf=logger,
                logger=logger)
            if plot:
                for model in results['models'].values():
                    model.evaluate(plot=True)
            return results, None
        except Exception as e:
            if plot:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, len(exposures))
                for idx, exposure in enumerate(exposures):
                    axes[idx].imshow(exposure.image)
            if printTrace:
                traceback.print_exc()
            return results, e

    def __getCatalog(self, filters, results, sources):
        """
        Get a new SimpleCatalog with added extra fields. Extra field keys are added to field, f
        :param filters: iterable[str]; Names of bandpass filters
        :param results: dict; Results structure as returned by mpfFit.fit_galaxy_exposures()
        :param sources: iterable[record]; Iterable of recods to pass to catalog.extend()
        :return: tuple of:
            catalog: afwTable.SimpleCatalog; Catalog with added fields
            fields: dict[str]; Dict of field keys by type
        """
        mapper = self._getMapper(sources.getSchema())
        schema = mapper.getOutputSchema()
        fields = {key: {} for key in ["base", "extra", "psf", "psf_extra", "measmodel"]}
        for idxBand, band in enumerate(filters):
            prefix = f'multiprofit_psf_{band}'
            resultsPsf = results['psfs'][idxBand]['galsim']
            fields["psf"][band] = {}
            fields["psf_extra"][band] = _defaultdictNested()
            fields["extra"][band] = _defaultdictNested()
            for name, fit in resultsPsf.items():
                fit = fit['fit']
                namesAdded = defaultdict(int)
                keyList = []
                for nameParam in fit['name_params']:
                    namesAdded[nameParam] += 1
                    fullname = f'{prefix}_{nameParam}_{namesAdded[nameParam]}'
                    key = schema.addField(fullname, type=np.float32)
                    keyList.append(key)
                fields["psf"][band][name] = _defaultdictNested()
                self.__addExtraFields(fields["psf_extra"][band][name], schema, prefix)
        for name, result in results['fits']['galsim'].items():
            prefix = f'multiprofit_{name}'
            fit = result['fits'][0]
            namesAdded = defaultdict(int)
            keyList = []
            bands = [x.band if hasattr(x, 'band') else '' for x, fixed in zip(
                fit['params'], fit['params_allfixed']) if not fixed]
            for nameParam, postfix in zip(fit['name_params'], bands):
                nameParam += postfix
                namesAdded[nameParam] += 1
                fullname = f'{prefix}_{nameParam}_{namesAdded[nameParam]}'
                key = schema.addField(fullname, type=np.float32)
                keyList.append(key)
            fields["base"][name] = keyList
            for band in filters:
                fields["extra"][band][name] = _defaultdictNested()
                self.__addExtraFields(fields["extra"][band][name], schema, prefix)
        if self.config.computeMeasModelfitLikelihood:
            for name in self.meas_modelfit_models:
                self.__addExtraField(fields["measmodel"], schema, "multiprofit_measmodel_like", name,
                                     'MultiProFit log-likelihood for meas_modelfit {name} model')
        catalog = afwTable.SimpleCatalog(schema)
        catalog.extend(sources, mapper=mapper)
        return catalog, fields

    @staticmethod
    def __setExtraField(extra, row, fit, name, nameFit=None, index=None):
        """
        Sets the value of an extra field for a given row
        :param extra: dict[str]; Dict containing field references; must contain name
        :param row: record; A row in a catalog containing the key extra[name]
        :param fit: dict[str]; A fit result containing a value for name/nameFit
        :param name: str; Record field name
        :param nameFit: str; Fit field name, if different from name
        :param index: int; Index of fit value, if it is not scaler but is indexable
        :return: No return
        """
        if nameFit is None:
            nameFit = name
        value = fit[nameFit]
        # TODO: Fix the bug in MPF that necessitates this - linear fits not returning prior
        if index is not None:
            shape = np.shape(value)
            if (not shape) and index != 0:
                raise RuntimeError(f"Tried to set extra field with index={index} from value={value} w/o len")
            if shape:
                value = value[index]
        row[extra[name]] = value

    def __setExtraFields(self, extra, row, fit):
        """
        Sets multiple extra fields based on self.config settings
        :param extra: dict[str]; Dict containing field references
        :param row: record; A row in a catalog containing the fields added by self.config settings
        :param fit: dict[str]; A fit result containing required values
        :return: No return
        """
        if self.config.outputChisqred:
            self.__setExtraField(extra, row, fit, 'chisqred')
        if self.config.outputLogLikelihood:
            self.__setExtraField(extra, row, fit, 'loglike', nameFit='likelihood', index=0)
        if self.config.outputRuntime:
            self.__setExtraField(extra, row, fit, 'time')
        self.__setExtraField(extra, row, fit, 'nEvalFunc', nameFit='n_eval_func')
        self.__setExtraField(extra, row, fit, 'nEvalGrad', nameFit='n_eval_grad')

    def __setFieldsSource(self, filters, results, fieldsBase, fieldsExtra, row):
        for idxfit, (name, result) in enumerate(results['fits']['galsim'].items()):
            fit = result['fits'][0]
            values = [x for x, fixed in zip(fit['params_bestall'], fit['params_allfixed'])
                      if not fixed]
            for value, key in zip(values, fieldsBase[name]):
                row[key] = value
            for band in filters:
                self.__setExtraFields(fieldsExtra[band][name], row, fit)

    def __setFieldsPsf(self, filters, results, fieldsBase, fieldsExtra, row):
        for idxBand, band in enumerate(filters):
            resultsPsf = results['psfs'][idxBand]['galsim']
            for name, fit in resultsPsf.items():
                fit = fit['fit']
                values = [x for x, fixed in zip(fit['params_bestall'], fit['params_allfixed'])
                          if not fixed]
                for value, key in zip(values, fieldsBase[band][name]):
                    row[key] = value
                self.__setExtraFields(fieldsExtra[band][name], row, fit)

    def __setFieldsMeasmodel(self, exposures, model, source, measmodelFields, row):
        """
        :measmodel_type: str; The type of corresponding meas_modelfit model (dev/exp/cmodel), if any
        :return: No return
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
            self.__setExtraField(measmodelFields, row, likelihood, measmodel_type)

    def __setRow(self, filters, results, fields, row, exposures, src, runtime=0):
        # Set up all of the appropriate fields in the schema
        # TODO: Do this prior to run time by e.g. setting up some extremely simple data that
        # should never fail, thereby also validating the modelSpecs.
        # Set PSF fit fields
        self.__setFieldsPsf(filters, results, fields["psf"], fields["psf_extra"], row)
        # Set the values of all extra fields - MultiProFit first
        self.__setFieldsSource(filters, results, fields["base"], fields["extra"], row)
        if self.config.computeMeasModelfitLikelihood:
            model = results['models'][self.modelSpecs[0]["model"]]
            self.__setFieldsMeasmodel(exposures, model, src, fields["measmodel"], row)
        row[self.runtimeKey] = runtime

    def fit(self, exposures, sources, logger=None, plot=False, idx_begin=0, idx_end=np.Inf, printTrace=False):
        """
        Fit every source with MultiProFit using the provided exposures/coadds
        :param exposures: Dict[band]; exposures/coadds to fit (one per filter)
        :param sources: Catalog; A catalog containing deblended sources with footprints
        :param logger: logging.Logger; A Logger object to (re-)direct MultiProFit output
        :param plot: bool; Whether to plot fit results for each source
        :param idx_begin: int; Row index to start fitting for
        :param idx_end: int; Row index
        :param printTrace: bool; Whether to print tracebacks when catching errors
        :return: catalog, results tuple containing:
            catalog: lsst.afw.table.SimpleCatalog; Catalog with fit results for each source
            results: dict; Results structure as returned by mpfFit.fit_galaxy_exposures() for the first
                successfully fit source
        """
        # Set up a logger to suppress output for now
        if logger is None:
            logger = logging.getLogger(__name__)
        numSources = len(sources)
        filters = exposures.keys()
        noiseReplacers = {
            band: rebuildNoiseReplacer(exposure, sources) for band, exposure in exposures.items()
        }
        timeInit = time.time()
        processTimeInit = time.process_time()
        addedFields = False
        resultsReturn = None
        indicesFailed = {}
        for idx, src in enumerate(sources):
            if idx_begin <= idx <= idx_end:
                results, error = self.__fitSource(src, noiseReplacers, exposures, filters, logger,
                                                  printTrace=printTrace, plot=plot)
                runtime = self.metadata["__fitSourceEndCpuTime"] - self.metadata["__fitSourceStartCpuTime"]
                failed = error is not None
                if resultsReturn is None and not failed:
                    resultsReturn = results
                if not addedFields:
                    if failed:
                        indicesFailed.add(idx)
                    else:
                        catalog, fields = self.__getCatalog(filters, results, sources)
                        for idxFailed, runtime in indicesFailed.items():
                            catalog[idxFailed][self.failFlagKey] = True
                            catalog[idxFailed][self.runtimeKey] = runtime
                        addedFields = True
                if addedFields:
                    row = catalog[idx]
                    if not failed:
                        self.__setRow(filters, results, fields, row, exposures, src, runtime=runtime)
                    row[self.failFlagKey] = failed
                elif failed:
                    indicesFailed[idx] = runtime
                id_src = src.getId()
                for noiseReplacer in noiseReplacers.values():
                    noiseReplacer.removeSource(id_src)
                errorMsg = '' if not failed else f" but got exception {error}"
                logger.log(
                    21, f"Fit src {idx}/{numSources} id={src['id']} in {runtime:.3f}s "
                        f"(total time {time.time() - timeInit:.2f}s "
                        f"process_time {time.process_time() - processTimeInit:.2f}s)"
                        f"{errorMsg}")
        # Return the exposures to their original state
        for noiseReplacer in noiseReplacers.values():
            noiseReplacer.end()
        return catalog, resultsReturn

    @pipeBase.timeMethod
    def run(self, coaddsByBand, sources, **kwargs):
        """
        Run MultiProFit on a catalog containing sources on a set of (multiband) exposures
        :param coaddsByBand: Dict[band]; A dict with one exposure/coadd per band
        :param sources: Catalog; A catalog containing deblended sources with footprints
        :param kwargs: dict; kwargs to pass to self.fit
        :return: catalog, results tuple containing:
            catalog: lsst.afw.table.SimpleCatalog; Catalog with fit results for each source
            results: dict; Results structure as returned by mpfFit.fit_galaxy_exposures() for the final source
        """
        catalog, results = self.fit(coaddsByBand, sources, **kwargs)
        return catalog, results
