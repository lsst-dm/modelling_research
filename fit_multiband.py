import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.daf.butler as dafButler
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from typing import Dict, Iterable, List, NamedTuple, Union

__all__ = ["CatalogExposure", "MultibandFitSubTask", "MultibandFitConfig", "MultibandFitTask"]


class CatalogExposure(NamedTuple):
    band: str
    catalog: afwTable.SourceCatalog
    exposure: afwImage.Exposure
    calib: afwImage.PhotoCalib = None
    dataId: Union[dafButler.DataCoordinate, None] = None
    metadata: Dict = None


multibandFitBaseTemplates = {"name_input_coadd": "deep", "name_output_coadd": "deep", "name_output_cat": "fit"}


class MultibandFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap"),
    defaultTemplates=multibandFitBaseTemplates,
):
    cat_ref = cT.Input(
        doc="Reference multiband source catalog",
        name="{name_input_coadd}Coadd_ref",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )
    cats_meas = cT.Input(
        doc="Deblended single-band source catalogs",
        name="{name_input_coadd}Coadd_meas",
        storageClass="SourceCatalog",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )
    coadds = cT.Input(
        doc="Exposures on which to run fits",
        name="{name_input_coadd}Coadd_calexp",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )
    cat_output = cT.Output(
        doc="Measurement multi-band catalog",
        name="{name_output_coadd}Coadd_{name_output_cat}",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )
    cat_ref_schema = cT.InitInput(
        doc="Schema associated with a ref source catalog",
        storageClass="SourceCatalog",
        name="{name_input_coadd}Coadd_ref_schema",
    )
    cat_output_schema = cT.InitOutput(
        doc="Output of the schema used in deblending task",
        name="{name_output_coadd}Coadd_{name_output_cat}_schema",
        storageClass="SourceCatalog"
    )

    def adjustQuantum(self, datasetRefMap):
        """ Validates the `lsst.daf.butler.DatasetRef` bands against the
        subtask's list of bands to fit and drops unnecessary bands.

        Parameters
        ----------
        datasetRefMap : `NamedKeyDict`
            Mapping from dataset type to a `set` of
            `lsst.daf.butler.DatasetRef` objects

        Returns
        -------
        datasetRefMap : `NamedKeyDict`
            Modified mapping of input with possibly adjusted
            `lsst.daf.butler.DatasetRef` objects.

        Raises
        ------
        ValueError
            Raised if any of the per-band datasets have an inconsistent band
            set, or if the band set to fit is not a subset of the data bands.

        """
        datasetRefMap = super().adjustQuantum(datasetRefMap)
        # Check which bands are going to be fit
        bands_fit, bands_read = self.config.get_band_sets()
        bands_needed = bands_fit.union(bands_read)

        bands_data = None
        has_extra_bands = False

        for type_d, ref_d in datasetRefMap.items():
            # Datasets without bands in their dimensions should be fine
            if 'band' in type_d.dimensions:
                bands = [dref.dataId['band'] for dref in ref_d]
                bands_set = set(bands)
                if bands_data is None:
                    bands_data = bands_set
                    if bands_data != bands_fit:
                        if not bands_needed.issubset(bands_data):
                            raise ValueError(f'bands_needed={bands_needed}) !subsetof bands_set={bands_set}'
                                             f' from refs={ref_d})')
                        has_extra_bands = True
                elif bands_set != bands_data:
                    raise ValueError(f'Datatype={type_d} bands_set={bands_set}) != previous={bands_data})')
                if has_extra_bands:
                    for dref in ref_d:
                        if dref.dataId['band'] not in bands_needed:
                            ref_d.remove(dref)
        return datasetRefMap


class MultibandFitSubTask(pipeBase.Task):
    ConfigClass = pexConfig.Config

    def __init__(self, schema: afwTable.Schema, **kwargs):
        """ Initialize the task, defining and setting the output catalog schema.

        Parameters
        ----------
        schema : `lsst.afw.table.Schema`
            The input schema for the reference source catalog, used to initialize the output schema.
        kwargs
            Additional arguments to be passed to the subclass constructor(s).

        Notes
        -----
        This method must initialize an attribute named schema with the schema of the output catalog.
        """
        raise RuntimeError("Not implemented")

    def run(
        self, catexps: Iterable[CatalogExposure], cat_ref: afwTable.SourceCatalog, **kwargs
    ) -> pipeBase.Struct:
        """ Fit sources from a reference catalog using data from multiple exposures in the same region (patch).

        Parameters
        ----------
        catexps : `typing.List [CatalogExposure]`
            A list of catalog-exposure pairs in a given band. Subclasses may require there to only be one catexp per
            band, and/or for the catalogs to contain HeavyFootprints with deblended images.
        cat_ref : `lsst.afw.table.SourceCatalog`
            A reference source catalog to fit. Subclasses may be configured fit only a subset of these sources.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            A struct with a cat_output attribute containing the output measurement catalog.
        """
        raise RuntimeError("Not implemented")


class MultibandFitConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=MultibandFitConnections,
):
    fit_multiband = pexConfig.ConfigurableField(
        target=MultibandFitSubTask,
        doc="Task to fit sources using multiple bands",
    )

    def get_band_sets(self):
        dict_fit = self.toDict()['fit_multiband']
        bands_to_fit = dict_fit.get('bands_fit')
        if bands_to_fit is None:
            raise RuntimeError(f'{__class__}.fit_multiband must have bands_fit attribute '
                               f'(self.toDict()["fit_multiband"]={dict_fit})')
        return set(bands_to_fit), set(dict_fit.get('bands_read', ()))


class MultibandFitTask(pipeBase.PipelineTask):
    ConfigClass = MultibandFitConfig
    _DefaultName = "multibandFit"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("fit_multiband", schema=initInputs["cat_ref_schema"].schema)
        self.cat_output_schema = afwTable.SourceCatalog(self.fit_multiband.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        input_objs = [inputRefs.cats_meas, inputRefs.coadds]
        dataIds_cats, dataIds = [[dRef.dataId for dRef in obj] for obj in input_objs]
        if not all((x == y for x, y in zip(dataIds, dataIds_cats))):
            raise RuntimeError(f'Mismatched cats_meas, coadds dataIds {dataIds_cats} {dataIds}')
        bands_data = [dataId['band'] for dataId in dataIds]
        bandset_fit, bandset_read = self.config.get_band_sets()
        bandset_needed = bandset_fit.union(bandset_read)
        bandset_data = set(bands_data)
        if bandset_data != bandset_needed:
            raise RuntimeError(f'set(band={bands_data})={bandset_data} != bandset_needed={bandset_needed}'
                               f'=(bands_fit={bandset_fit}).union(bands_read={bandset_read}))')
        cat_ref, cats_meas, coadds = (inputs[x] for x in ('cat_ref', 'cats_meas', 'coadds'))
        n_data = [len(x) for x in (bands_data, cats_meas, coadds)]
        if not all(x == n_data[0] for x in n_data[1:]):
            raise RuntimeError(f'bands, cats_meas, coadds, dataIds lens={n_data} not matched')
        catexps = [
            CatalogExposure(band=band, catalog=cat, exposure=exp, dataId=dataId, metadata={},
                            calib=exp.getPhotoCalib() if hasattr(exp, 'getPhotoCalib') else None)
            for band, cat, exp, dataId in zip(bands_data, cats_meas, coadds, dataIds)
        ]
        outputs = self.run(catexps=catexps, cat_ref=cat_ref)
        butlerQC.put(outputs, outputRefs)

    def run(self, catexps: List[CatalogExposure], cat_ref: afwTable.SourceCatalog) -> pipeBase.Struct:
        """ Fit sources from a reference catalog using data from multiple exposures in the same region (patch).

        Parameters
        ----------
        catexps : `typing.List [CatalogExposure]`
            A list of catalog-exposure pairs in a given band. Subtasks may require there to only be one catexp per band,
            and/or for the catalogs to contain HeavyFootprints with deblended images.
        cat_ref : `lsst.afw.table.SourceCatalog`
            A reference source catalog to fit. Subtasks may be configured fit only a subset of these sources.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            A struct with a cat_output attribute containing the output measurement catalog.
        """
        cat_output = self.fit_multiband.run(catexps, cat_ref).output
        # It would be best to validate this earlier but I'm not sure how
        if cat_output.schema != self.cat_output_schema.schema:
            raise RuntimeError(f'fit_multiband.run schema != initOutput schema:'
                               f' {cat_output.schema} vs {self.cat_output_schema.schema}')
        retStruct = pipeBase.Struct(cat_output=cat_output)
        return retStruct
