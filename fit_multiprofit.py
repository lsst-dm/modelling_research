import logging
from lsst.pipe.base.testUtils import makeQuantum, runTestQuantum
from lsst.daf.butler import Butler, DatasetType
from modelling_research.fit_multiband import MultibandFitConfig, MultibandFitConnections, MultibandFitTask
from modelling_research.multiprofit_task import MultiProFitTask, MultiProFitConfig
import sys

make_mpf_task = True
logging.basicConfig(stream=sys.stdout, level=20)

dataId = dict(tract=9813, patch=40, skymap='hsc_rings_v1')
bands = ['g', 'r', 'i']

config = MultibandFitConfig()
config.connections.name_output_cat = "multiprofit" if make_mpf_task else "fit"
connections = MultibandFitConnections(config=config)

butler = Butler(
    '/project/hsc/gen3repo/rc2w02_ssw03',
    collections="HSC/runs/RC2/w_2021_02",
    run="u/dtaranu/DM-28429",
)

universe = butler.registry.dimensions
for names_output in (connections.outputs, connections.initOutputs):
    for name_output in names_output:
        ct_output = getattr(connections, name_output)
        try:
            butler.registry.getDatasetType(ct_output.name)
        except KeyError as e:
            print(f'Exception: {e}; attempting to register output datasetType: {ct_output.name}')
            dataset_type = DatasetType(ct_output.name, ct_output.dimensions if hasattr(ct_output, "dimensions") else [],
                                       ct_output.storageClass, universe=universe)
            butler.registry.registerDatasetType(dataset_type)

if make_mpf_task:
    config_mpf = MultiProFitConfig()
    config.fit_multiband.retarget(MultiProFitTask)
    config.fit_multiband.bands_fit = bands
    config.fit_multiband.idx_end = 1

config.freeze()

task = MultibandFitTask(config=config, initInputs={'cat_ref_schema': butler.get('deepCoadd_ref_schema', dataId)})

dataIds_band = []
for band in bands:
    dataId_band = dataId.copy()
    dataId_band['band'] = band
    dataIds_band.append(dataId_band)

ids = dict(cat_ref=dataId, coadds=dataIds_band, cats_meas=dataIds_band, cat_output=dataId)
quantum = makeQuantum(task, butler, dataId, ids)

for mockRun in (True, False):
    print(f'Running test quantum with mockRun={mockRun}')
    runTestQuantum(task, butler, quantum, mockRun=mockRun)