import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from typing import List, NamedTuple, Optional, Set


class CatalogExposure(NamedTuple):
    band: str
    catalog: afwTable.SourceCatalog
    exposure: afwImage.Exposure
    calib: afwImage.PhotoCalib = None


multibandFitBaseTemplates = {
    "name_input_coadd": "deep", "name_output_coadd": "deep",
    "name_output_cat": "fit", "name_output_bands": "all",
}


class MultibandFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates=multibandFitBaseTemplates,
):
    # Should this really be included here, or in subclasses if that's possible?
    calibs = cT.Input(
        doc="Photometric calibration for exposures",
        name="{name_input_coadd}Coadd_calib",
        storageClass="PhotoCalib",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )
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
        name="{name_output_coadd}Coadd_{name_output_cat}_{name_output_bands}",
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
        name="{name_output_coadd}Coadd_{name_output_cat}_{name_output_bands}_schema",
        storageClass="SourceCatalog"
    )

    # Is this needed?
    def setDefaults(self):
        super().setDefaults()

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.require_calibs:
            self.inputs.remove("calibs")


# Should this class do anything?
class MultibandFitSubTask(pipeBase.Task):
    _DefaultName = "multibandFitSub"
    ConfigClass = pexConfig.Config

    def get_bands_calib_required(self) -> Set[str]:
        raise RuntimeError("Not implemented")

    def get_schema(self, **kwargs) -> afwTable.Schema:
        raise RuntimeError("Not implemented")

    def run(self, catexps, cat_ref, **kwargs) -> afwTable.SourceCatalog:
        raise RuntimeError("Not implemented")


class MultibandFitConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=MultibandFitConnections,
):
    fit_multiband = pexConfig.ConfigurableField(
        target=MultibandFitSubTask,
        doc="Task to fit sources using multiple bands",
    )
    # But this should be determined by fit_multiband.config...
    require_calibs = False


class MultibandFitTask(pipeBase.PipelineTask):
    ConfigClass = MultibandFitConfig
    _DefaultName = "multibandFit"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("fit_multiband")
        self.cat_output_schema = afwTable.SourceCatalog(
            self.fit_multiband.get_schema(initInputs["cat_ref_schema"].schema)
        )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        # I don't know if these are really necessary - I think only deblending really needs an IdFactory?
        #packedId, maxBits = butlerQC.quantum.dataId.pack("tract_patch", returnMaxBits=True)
        #inputs["idFactory"] = afwTable.IdFactory.makeSource(packedId, 64 - maxBits)
        has_calibs = hasattr(inputRefs, 'calibs')
        input_objs = [inputRefs.cats_meas, inputRefs.coadds]
        if has_calibs:
            input_objs.append(inputRefs.calibs)
        bands_inputs = [[dRef.dataId["band"] for dRef in obj] for obj in input_objs]
        bands = bands_inputs[0]
        if not all(x == bands for x in bands_inputs[1:]):
            raise RuntimeError(f'bands_outputs={bands_inputs} not all identical')
        bands_set = set(bands)
        bands_calib = self.fit_multiband.get_bands_calib_required()
        if len(bands_calib) > 0:
            if not self.config.require_calibs:
                raise RuntimeError(f"self.fit_multiband={self.fit_multiband} has bands_calib={bands_calib} but "
                                   f"require_calib={require_calib} is False")
            if not bands_calib.issubset(bands_set):
                raise RuntimeError(f"self.fit_multiband={self.fit_multiband} bands_calib={bands_calib} not subset of "
                                   f"bands_set={bands_set}")
        else:
            # Warn instead?
            if self.config.require_calibs:
                raise RuntimeError(f"self.fit_multiband={self.fit_multiband} has bands_calib={bands_calib} but "
                                   f"require_calib={require_calib} is True")
        inputs['bands'] = bands
        outputs = self.run(**inputs)
        # How is consistency with the connections enforced? e.g. if the next line is commented out
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        bands: List[str],
        cat_ref: afwTable.SourceCatalog,
        cats_meas: List[afwTable.SourceCatalog],
        coadds: List[afwImage.Exposure],
        calibs: Optional[List[afwImage.PhotoCalib]] = None
    ):
        if calibs is None:
            calibs = [None]*len(bands)
        n_data = [len(x) for x in (bands, cats_meas, coadds, calibs)]
        if not all(x == n_data[0] for x in n_data[1:]):
            raise RuntimeError(f'bands, cats_meas, coadds lens={n_data} not matched')
        catexps = [
            CatalogExposure(band=band, catalog=cat, exposure=exp, calib=calib)
            for band, cat, exp, calib in zip(bands, cats_meas, coadds, calibs)
        ]
        cat_output = self.fit_multiband.run(catexps, cat_ref)
        # It would make much more sense to validate this earlier, but I'm not sure how - it's easy only for the subtask
        if cat_output.schema != self.cat_output_schema.schema:
            print(type(cat_output.schema), type(self.cat_output_schema.schema))
            raise RuntimeError(f'fit_multiband.run schema != initOutput schema:'
                               f' {cat_output.schema} vs {self.cat_output_schema.schema}')
        retStruct = pipeBase.Struct(cat_output=cat_output)
        return retStruct
