import argparse
import copy
import os
import lsst.daf.persistence as dafPersist
from lsst.afw.table import SourceTable
from lsst.meas.base.measurementInvestigationLib import makeRerunCatalog, rebuildNoiseReplacer
from lsst.meas.base import (SingleFrameMeasurementConfig,
                            SingleFrameMeasurementTask)
import lsst.meas.modelfit
import lsst.log
import lsst.log.utils


def setupCModelConfig (path = "/project/dtaranu/cmodelconfigs/", stack="w_2018_14",
                       filters="HSC-R", tracts=9813, patches = '3,4'):

    configinfo = {
        "path": path,
        "stack": stack,
        'butlerdata': {'filter': filters, 'tract': tracts, 'patch': patches}
    }
    params = {
        'gradthresh': {'values': [1e-1, 1e-2, 1e-3], 'default': '1e-2', 'path': '_gradthresh_'},
        'maxinnerit': {'values': [8, 20, 50], 'default': '20', 'path': '_maxinnerit'},
        'nComps': {'values': [2, 3, 6], 'default': '3', 'path': '_nComp'},
        'nfitrad': {'values': [2, 3, 5], 'default': '3', 'path': '_nfitrad_'},
        'sr1term': {'values': [True, False], 'default': 'f', 'path': '_sr1term_'}
    }

    # Should've done this more intelligently
    for param in ['maxinnerit', 'nComps', 'nfitrad']:
        params[param]['names'] = [str(i) for i in params[param]['values']]
    params['gradthresh']['names'] = [
        '{:1.0e}'.format(i).replace("e-0", "e-") for \
        i in params['gradthresh']['values']]
    params['sr1term']['names'] = ['t', 'f']

    configinfo['params'] = params

    return configinfo


def getCModelButlers(cmodelparams, butlerdata, stack, basepath=""):

    butlers = dict()
    butlerdefault = None

    for key, value in cmodelparams.items():

        butlers[key] = dict()

        for paramstr in value['names']:

            isdefault = paramstr == value['default']
            loadbutler = (not isdefault) or (butlerdefault is None)

            if loadbutler:
                msg = "Loading non-default: "
            else:
                msg = "Skipping default: "

            pathtostack = stack + value['path'] + paramstr
            print(msg + pathtostack + " from path " + basepath)

            if loadbutler:
                butlers[key][paramstr] = {'butler':  dafPersist.Butler(os.path.join(basepath, pathtostack))}
                butler = butlers[key][paramstr]['butler']
                butlers[key][paramstr]['measures'] = butler.get('deepCoadd_meas', butlerdata)
                butlers[key][paramstr]['calexp'] = butler.get('deepCoadd_calexp', butlerdata)
                butlers[key][paramstr]['srccat'] = butler.get('deepCoadd_forced_src', butlerdata)
                # We don't bother to store a noiseReplacer because its heavy footprints get deleted every use
                # Hence the 'noiseReplacer' name, I guess...

            if isdefault:
                if butlerdefault is None:
                    butlerdefault = butlers[key][paramstr]
                else:
                    butlers[key][paramstr] = butlerdefault

    return [butlers, butlerdefault]


def getCModelButlerdefault(config = setupCModelConfig()):

    config["params"] = {'sr1term': config["params"]["sr1term"]}
    config["params"]["sr1term"]["values"] = [False]
    config["params"]["sr1term"]["names"] = ['f']
    [butlers, butlerdefault] = getCModelButlers(config["params"], config["butlerdata"], config["stack"], config["path"])
    return butlerdefault


def getCModelRerunTask(idsToRerun, measTable, calExp):

    # Plugin and butler
    plugin = "modelfit_CModel"

    # Fields to copy from old catalog, these are generally fields added outside
    # the measurement framework, that may be desirable to maintain
    fields = [plugin + "_initial_flux", "id"]

    # Set plugin and dependencies (subtasks)
    dependencies = ("modelfit_DoubleShapeletPsfApprox", "base_PsfFlux")

    # Create a new schema object, and use it to initialize a measurement task
    schema = SourceTable.makeMinimalSchema()

    # Configure any plugins at this stage.
    measConfig = SingleFrameMeasurementConfig()
    measConfig.plugins.names = tuple(dependencies) + (plugin,)

    # Hacky - replace when I find the proper way to do this
    # compound keys? (point key, quadrupole key). getAliasMap?
    for key, value in measConfig.slots.items():
        field = value
        if key.endswith("Flux"):
            field += "_flux"
        if key == "centroid" or key.endswith("Centroid"):
            field = [field + "_" + str(d) for d in ["x", "y"]]
        elif key == "shape" or key.endswith("Shape"):
            field = [field + "_" + str(d) for d in ["xx", "xy", "yy"]]
        else:
            field = [field]
        fields.extend(field)

    if True:
        for name in ["meas." + post for post in
                ["modelfit." + post2 for post2 in
                    ["AdaptiveImportanceSampler"
                        , "CModel"
                        , "integrals"
                        , "optimizer"
                        , "SoftenedLinearPrior"
                    ]] +
                ["base.psfflux"]
                ]:
            lsst.log.utils.traceSetAt(name, 5)

    measTask = SingleFrameMeasurementTask(schema, config=measConfig)
#   measTask.log.setLevel(5000)

    plugincmodel = measTask.plugins["modelfit_CModel"]
    plugincmodel.config.initial.optimizer.doSaveIterations = True

    newSrcCatalog = makeRerunCatalog(schema, measTable, idsToRerun, fields=fields)
    parentkey = newSrcCatalog.getParentKey()
    newSrcCatalog[0][parentkey] = 0

    noiseReplacer = rebuildNoiseReplacer(calExp, measTable)
    # Re-run measure on the sources selected above, using the reconstructed
    # noise replacer.
    return [measTask, noiseReplacer, newSrcCatalog]


def runCModelTargets(idsToRerun, butlerdict):

    calExp = copy.deepcopy(butlerdict['calexp'])
    measTable = butlerdict['measures']
    [measTask, noiseReplacer, newSrcCatalog] = getCModelRerunTask(idsToRerun, measTable, calExp)
    measTask.runPlugins(noiseReplacer, newSrcCatalog, calExp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test re-running LSST CModel on a single object")
    parser.add_argument("-path", help="The path to the already run deep coadds")
    parser.add_argument("-row", default=0, type=int, help="The row number of the object in the measurement table")
    parser.set_defaults(path = None, row=0)

    args = parser.parse_args()

    butler = getCModelButlerdefault()
    meastab = butler["measures"]
    idtarget = meastab["id"][args.row]
    runCModelTargets([idtarget], butler)
    [measTask, noiseReplacer, newSrcCatalog] = getCModelRerunTask(idsToRerun, measTable, calExp)
    measTask.runPlugins(noiseReplacer, newSrcCatalog, calExp)
