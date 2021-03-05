from modelling_research.multiprofit_task import MultiProFitTask

config.connections.name_output_cat = "multiprofit"
config.fit_multiband.retarget(MultiProFitTask)
config.fit_multiband.fitCModelExp = False
config.fit_multiband.fitSersicFromCModel = True
config.fit_multiband.computeMeasModelfitLikelihood = True
config.fit_multiband.usePriorShapeDefault = True
config.fit_multiband.priorCentroidSigma = 0.2
config.fit_multiband.priorMagBand = "i"
config.fit_multiband.psfHwhmShrink = 0.1
