import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from typing import NamedTuple

CatalogExposure = NamedTuple('CatalogExposure', [
    ('band', str),
    ('catalog', afwTable.SourceCatalog),
    ('exposure', afwImage.Exposure),
])


multibandFitBaseTemplates = {"name_input_coadd": "deep", "name_output_coadd": "deep", "name_output_cat": "fit"}


class MultibandFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates=multibandFitBaseTemplates,
):
    cat_ref = cT.PrerequisiteInput(
        doc="Reference multiband source catalog",
        name="{name_input_coadd}Coadd_ref",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )
    cats_meas = cT.PrerequisiteInput(
        doc="Deblended single-band source catalogs",
        name="{name_input_coadd}Coadd_meas",
        storageClass="SourceCatalog",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )
    coadds = cT.PrerequisiteInput(
        doc="Exposures on which to run fits",
        name="{name_input_coadd}Coadd_calexp",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap"),
    )
    #cat_output = cT.Output(
    #    doc="Measurement multi-band catalog",
    #    name="{name_output_coadd}Coadd_{name_output_cat}",
    #    storageClass="SourceCatalog",
    #   dimensions=("tract", "patch", "skymap"),
    #)
    schema_input = cT.InitInput(
        doc="Schema associated with a ref source catalog",
        storageClass="SourceCatalog",
        name="{name_input_coadd}Coadd_ref",
    )
    schema_output = cT.InitOutput(
        doc="Output of the schema used in deblending task",
        name="{name_output_coadd}Coadd_{name_output_cat}_schema",
        storageClass="SourceCatalog"
    )

    def setDefaults(self):
        super().setDefaults()


class MultibandFitSubTask(pipeBase.Task):
    _DefaultName = "multibandFitSub"
    ConfigClass = pexConfig.Config

    def run(self, cat_ref, cats_meas, coadds):
        raise RuntimeError("Not implemented")


class MultibandFitConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=MultibandFitConnections,
):
    fit_multiband = pexConfig.ConfigurableField(
        target=MultibandFitSubTask,
        doc="Task to fit sources using multiple bands",
    )


class MultibandFitTask(pipeBase.PipelineTask):
    ConfigClass = MultibandFitConfig
    _DefaultName = "multibandFit"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("fit_multiband")
        #self.outputSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        #packedId, maxBits = butlerQC.quantum.dataId.pack("tract_patch", returnMaxBits=True)
        #inputs["idFactory"] = afwTable.IdFactory.makeSource(packedId, 64 - maxBits)
        bands_cats, bands_coadds = (
            [dRef.dataId["band"] for dRef in obj]
            for obj in (inputRefs.cats_meas, inputRefs.coadds)
        )
        if bands_cats != bands_coadds:
            raise RuntimeError(f"meas SourceCatalog bands={bands_cats} != coadd exposure bands={bands_coadds}")
        inputs['bands'] = bands_cats
        outputs = self.run(**inputs)
        #butlerQC.put(outputs, outputRefs)

    def run(self, bands, cat_ref, cats_meas, coadds):
        n_data = [len(x) for x in (bands, cats_meas, coadds)]
        if not all(x == n_data[0] for x in n_data[1:]):
            raise RuntimeError('bands, cats_meas, coadds lens={n_data} not matched')
        catexps = [
            CatalogExposure(band=band, catalog=cat, exposure=exp)
            for band, cat, exp in zip(bands, cats_meas, coadds)
        ]
        cat_output = self.fit_multiband.run(catexps)
        retStruct = pipeBase.Struct(cat_output=cat_output)
        return retStruct
