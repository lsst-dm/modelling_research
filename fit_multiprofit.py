import logging
from lsst.pipe.base.testUtils import makeQuantum, runTestQuantum
from lsst.daf.butler import Butler, DatasetType
from modelling_research.fit_multiband import MultibandFitConfig, MultibandFitConnections, MultibandFitTask
from modelling_research.multiprofit_task import MultiProFitTask, MultiProFitConfig
import sys

make_mpf_task = True
logging.basicConfig(stream=sys.stdout, level=21)

dataId = dict(tract=9813, patch=40, skymap='hsc_rings_v1')
bands = ['g', 'r', 'i']

config = MultibandFitConfig()
if make_mpf_task:
    config_mpf = MultiProFitConfig()
    config.fit_multiband.retarget(MultiProFitTask)
    config.fit_multiband.bands = bands
    config.fit_multiband.idx_end = 10

# Is this really the best way to do it? It gives a lot of freedom.
# Could name_output_cat be derived from the name of the subtask?
# Could and should name_output_bands be derived from the connections?
config.connections.name_output_cat = "multiprofit"
config.connections.name_output_bands = "".join(bands)

butler = Butler(
    '/project/hsc/gen3repo/rc2w02_ssw03',
    collections="HSC/runs/RC2/w_2021_02",
    run="u/dtaranu/DM-28429",
)

# This seems redundant with the connection, but I'm not sure how it would work without a valid dataId?
task = MultibandFitTask(config=config, initInputs={'cat_ref_schema': butler.get('deepCoadd_ref_schema', dataId)})
config.fit_multiband.freeze()
config.freeze()

# This seems to be necessary - the dataset type isn't automagically registered by runTestQuantum
# Repeated registration seems harmless but is avoided here anyway
# I originally had name_output=config.connections.cat_output but that doesn't have substitutions yet afaict
ct_output = MultibandFitConnections(config=config).cat_output
name_output = ct_output.name
try:
    butler.registry.getDatasetType(name_output)
except KeyError as e:
    print(f'Exception: {e}; attempting to register output datasetType: {name_output}')

    dataset_type = DatasetType(name_output, ct_output.dimensions, ct_output.storageClass,
                               universe=butler.registry.dimensions)
    butler.registry.registerDatasetType(dataset_type)

# Is there an easier way to build this quantum?
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