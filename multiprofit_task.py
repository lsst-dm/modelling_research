from collections import defaultdict
import logging
import lsst.afw.table as afwTable
from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
from lsst.meas.modelfit.display import buildCModelImages
from lsst.meas.modelfit.cmodel.cmodelContinued import CModelConfig
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import matplotlib.pyplot as plt
import multiprofit.fitutils as mpfFit
import multiprofit.objects as mpfObj
import numpy as np
import time
import traceback


class MultiProFitConfig(pexConfig.Config):
    """Configuration for the MultiProFit profile fitter.

    Notes
    -----
    gaussianOrderSersic only has a limited number of valid values (those supported by multiprofit's
    MultiGaussianApproximationComponent).
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
    returning additional measurements in a new SimpleCatalog.

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
        self.schema = None

    def _getMapper(self, schema):
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
        self.runtimeKey = mapper.editOutputSchema().addField('multiprofit_time_total', type=np.float32,
                                                             doc='runtime in ms')
        self.failFlagKey = mapper.editOutputSchema().addField('multiprofit_fail_flag', type="Flag",
                                                              doc='generic MultiProFit failure flag')
        return mapper

    @staticmethod
    def __addExtraField(extra, schema, prefix, name, doc):
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
        Returns
        -------
        No return. The new field is added to `schema` and a reference to it is stored in `extra`.
        """
        if doc is None:
            doc = ''
        extra[name] = schema.addField(joinFilter('_', [prefix, name]), type=np.float32, doc=doc)

    def __addExtraFields(self, extra, schema, prefix=None):
        """Add all extra fields for a given model based on `self.config` settings.

        Parameters
        ----------
        extra : `dict` of `str`
            An input dictionary to store reference to the new `Key`s by their field names.
        schema : `lsst.afw.table.Schema`
            An existing table schema to add the field to.
        prefix : `str`, optional
            A string such as a model name to prepend to each field name; default None.

        Returns
        -------
        No return. The new fields are added to `schema` and reference to them are stored in `extra`.
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
    def __fitSource(self, source, noiseReplacers, exposures, logger, printTrace=False, plot=False):
        """
        Fit a single deblended source with MultiProFit.

        Parameters
        ----------
        source : `lsst.afw.table.SourceRecord`
            A deblended source to fit.
        noiseReplacers : iterable of `lsst.meas.base.NoiseReplacer`
            An iterable NoiseReplacers that will insert the source into every exposure
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        logger : `logging.Logger`
            A Logger to log output.
        printTrace : `bool`, optional
            Whether to print the traceback in case of an error; default False.
        plot : `bool`, optional
            Whether to generate a plot window with the final output; default False.
        Returns
        -------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_galaxy_exposures, if no error occurs.
        error : `Exception`
            The first exception encountered while fitting, if any.
        """
        results = None
        try:
            foot = source.getFootprint()
            bbox = foot.getBBox()
            center = bbox.getCenter()
            # TODO: Implement multi-object fitting/deblending
            # peaks = foot.getPeaks()
            # nPeaks = len(peaks)
            # isSingle = nPeaks == 1
            for noiseReplacer in noiseReplacers:
                noiseReplacer.insertSource(source.getId())
            exposurePsfs = []
            for band, exposure in exposures.items():
                # TODO: Check total flux first
                mpfExposure = mpfObj.Exposure(
                    band=band, image=np.float64(exposure.image.subset(bbox).array),
                    error_inverse=1 / np.float64(exposure.variance.subset(bbox).array),
                    is_error_sigma=False)
                mpfPsf = mpfObj.PSF(band, image=exposure.getPsf().computeKernelImage(center),
                                    engine="galsim")
                exposurePsfs.append((mpfExposure, mpfPsf))
            results = mpfFit.fit_galaxy_exposures(
                exposurePsfs, exposures.keys(), self.modelSpecs, results=results, loggerPsf=logger,
                logger=logger)
            if plot:
                for model in results['models'].values():
                    model.evaluate(plot=True)
            return results, None
        except Exception as e:
            if plot:
                fig, axes = plt.subplots(1, len(exposures))
                for idx, exposure in enumerate(exposures):
                    axes[idx].imshow(exposure.image)
            if printTrace:
                traceback.print_exc()
            return results, e

    def __getCatalog(self, filters, results, sources):
        """Get a new catalog and a dict containing the keys of extra fields to enter for each row.

        Parameters
        ----------
        filters : iterable of `str`
            Names of bandpass filters for filter-dependent fields.
        results : `dict`
            Results structure as returned by `__fitSource`.
        sources : `iterable` of `lsst.afw.table.BaseRecord`

        Returns
        -------
        catalog : `lsst.afw.table.SimpleCatalog`
            A new `SimpleCatalog` with extra fields.
        fields : `dict` [`str`, `dict`]
            A dict of dicts, keyed by the field type. The values may contain further nested dicts e.g. those
            keyed by filter for PSF fit-related fields.
        """
        mapper = self._getMapper(sources.getSchema())
        schema = mapper.getOutputSchema()
        fields = {key: {} for key in ["base", "extra", "psf", "psf_extra", "measmodel"]}
        # Set up the fields for PSF fits, which are independent per filter
        for idxBand, band in enumerate(filters):
            prefix = f'multiprofit_psf_{band}'
            resultsPsf = results['psfs'][idxBand]['galsim']
            fields["psf"][band] = {}
            fields["psf_extra"][band] = defaultdictNested()
            for name, fit in resultsPsf.items():
                fit = fit['fit']
                namesAdded = defaultdict(int)
                keyList = []
                for nameParam in fit['name_params']:
                    namesAdded[nameParam] += 1
                    fullname = f'{prefix}_{nameParam}_{namesAdded[nameParam]}'
                    key = schema.addField(fullname, type=np.float32)
                    keyList.append(key)
                fields["psf"][band][name] = defaultdictNested()
                self.__addExtraFields(fields["psf_extra"][band][name], schema, prefix)
        # Setup field names for source fits, which may have fluxes in multiple filters if run in multi-band.
        # Either way, flux parameters should contain a filter name.
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
            fields["extra"][name] = defaultdictNested()
            self.__addExtraFields(fields["extra"][name], schema, prefix)

        if self.config.computeMeasModelfitLikelihood:
            for name in self.meas_modelfit_models:
                self.__addExtraField(fields["measmodel"], schema, "multiprofit_measmodel_like", name,
                                     'MultiProFit log-likelihood for meas_modelfit {name} model')
        catalog = afwTable.SimpleCatalog(schema)
        catalog.extend(sources, mapper=mapper)
        return catalog, fields

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
        # TODO: Fix the bug in MPF that necessitates this - linear fits not returning prior (DM-21197)
        if index is not None:
            shape = np.shape(value)
            if (not shape) and index != 0:
                raise RuntimeError(f"Tried to set extra field with index={index} from value={value} w/o len")
            if shape:
                value = value[index]
        row[extra[name]] = value

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
            fit = result['fits'][0]
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
        for idxBand, band in enumerate(filters):
            resultsPsf = results['psfs'][idxBand]['galsim']
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

        Returns
        -------
        None
        """
        self.__setFieldsPsf(results, fields["psf"], fields["psf_extra"], row, filters)
        self.__setFieldsSource(results, fields["base"], fields["extra"], row)
        if self.config.computeMeasModelfitLikelihood:
            model = results['models'][self.modelSpecs[0]["model"]]
            self.__setFieldsMeasmodel(exposures, model, source, fields["measmodel"], row)
        row[self.runtimeKey] = runtime

    def fit(self, exposures, sources, idx_begin=0, idx_end=np.Inf, logger=None, printTrace=False, plot=False):
        """Fit a catalog of sources with MultiProFit.

        Each source has its PSF fit with a configureable Gaussian mixture PSF model and then fits a
        sequence of different models, some of which can be configured to be initialized from previous fits.
        See `MultiProFitConfig` for more information on how to configure models.

        Plots can be generated on success (with MultiProFit defaults), or on failure, in which case only the
        images themselves are shown. Tracebacks are suppressed on failure by default.

        Parameters
        ----------
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        sources: `lsst.afw.table`
            A catalog containing deblended sources with footprints
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

        Returns
        -------
        catalog : `lsst.afw.table.SimpleCatalog`
            A new catalog containing all of the fields from `sources` and those generated by MultiProFit.
        results : `dict`
            A results structure as returned by mpfFit.fit_galaxy_exposures() for the first successfully fit
            source.
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
                results, error = self.__fitSource(src, noiseReplacers.values(), exposures, logger,
                                                  printTrace=printTrace, plot=plot)
                runtime = self.metadata["__fitSourceEndCpuTime"] - self.metadata["__fitSourceStartCpuTime"]
                failed = error is not None
                # Preserve the first successful result to return at the end
                if resultsReturn is None and not failed:
                    resultsReturn = results
                # Setup field names if necessary
                if not addedFields and not failed:
                    catalog, fields = self.__getCatalog(filters, results, sources)
                    for idxFailed, runtime in indicesFailed.items():
                        catalog[idxFailed][self.failFlagKey] = True
                        catalog[idxFailed][self.runtimeKey] = runtime
                    addedFields = True
                # Fill in field values if successful, or save just the runtime to enter later otherwise
                if addedFields:
                    row = catalog[idx]
                    if not failed:
                        self.__setRow(filters, results, fields, row, exposures, src, runtime=runtime)
                    row[self.failFlagKey] = failed
                elif failed:
                    indicesFailed[idx] = runtime
                id_src = src.getId()
                # Returns the image to pure noise
                for noiseReplacer in noiseReplacers.values():
                    noiseReplacer.removeSource(id_src)
                errorMsg = '' if not failed else f" but got exception {error}"
                # Log with a priority just above info, since MultiProFit itself will generate a lot of info
                #  logs per source.
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
        catalog : `lsst.afw.table.SimpleCatalog`
            A new catalog containing all of the fields from `sources` and those generated by MultiProFit.
        results : `dict`
            A results structure as returned by mpfFit.fit_galaxy_exposures() for the first successfully fit
            source.
        """
        catalog, results = self.fit(exposures, sources, **kwargs)
        return catalog, results
