
# coding: utf-8

# In[1]:


import numpy as np

from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
from lsst.daf.persistence import Butler

ciHscDataPath = "" # Set this to the path to a ci-hsc data repository.
ciHscDataPath = "/ssd/nlust/repos_lsst/ci_hsc/DATA/rerun/ci_hsc/"

# Create a butler object for loading in the data.
butler = Butler(ciHscDataPath)

# Create a data Id for a single ccd.
dataId = {"visit":903334, "ccd":16, "filter":"HSC-R"}

# Load in the calibrated exposure, and the associated source catalog.
exposure = butler.get("calexp", dataId)
srcCat = butler.get("src", dataId)

# Reconstruct a noise replacer from the loaded data.
noiseReplacer = rebuildNoiseReplacer(exposure, srcCat)


# In[2]:


# Continued from the above example
from lsst.afw.table import SourceTable
from lsst.meas.base.measurementInvestigationLib import makeRerunCatalog
from lsst.meas.base import (SingleFrameMeasurementConfig,
                            SingleFrameMeasurementTask)

# Make a list of ids of objects to remeasure
idsToRerun = [775958066192449538, 775958066192449539,
              775958066192449540, 775958066192449541]

# Fields to copy from old catalog, these are generally fields added outside
# the measurement framework, that may be desirable to maintain
fields = ["deblend_nChild"]

# Create a new schema object, and use it to initialize a measurement task
schema = SourceTable.makeMinimalSchema()

# Configure any plugins at this stage.
measConfig = SingleFrameMeasurementConfig()

measTask = SingleFrameMeasurementTask(schema, config=measConfig)

# Create a Measurement catalog containing only the ids to remeasure
newSrcCatalog = makeRerunCatalog(schema, srcCat, idsToRerun, fields=fields)

# Re-run measure on the sources selected above, using the reconstructed
# noise replacer.
measTask.runPlugins(noiseReplacer, newSrcCatalog, exposure)


# In[3]:


# Get child objects
parentkey = srcCat.getParentKey()
children = np.where(srcCat[parentkey] != 0)[0]
idsToRerun = srcCat["id"][children]

# Setup again
newSrcCatalog = makeRerunCatalog(schema, srcCat, idsToRerun, fields=fields)
noiseReplacer = rebuildNoiseReplacer(exposure, srcCat)

from collections import namedtuple
timingResult = namedtuple("TimingResult",
    ['niter', 'total', 'min', 'median', 'mean'])

# Define a convenient function for timing
def timeRunPlugins(repeat=5):
    timer = timeit.Timer(
        "measTask.runPlugins(noiseReplacer, newSrcCatalog, exposure)",
        setup="from lsst.meas.base.measurementInvestigationLib "
              "import rebuildNoiseReplacer;"
              "noiseReplacer = rebuildNoiseReplacer(exposure, srcCat)",
        globals={'exposure':exposure, 'srcCat':srcCat, 'measTask': measTask,
                'newSrcCatalog':newSrcCatalog}
    )
    times = timer.repeat(repeat,1)
    result = timingResult(repeat, np.sum(times), np.min(times),
                          np.median(times), np.mean(times))
    return result


# In[4]:


import timeit
# This runs none of the children because their parents aren't in the cat
# It only takes some time to rebuild the noiseReplacer each iteration
times = timeRunPlugins()
print(times)


# In[5]:


# Set parents to zero
newSrcCatalog = makeRerunCatalog(schema, srcCat, idsToRerun,
                                 fields=fields).copy(deep=True)
newSrcCatalog[parentkey] = 0

noiseReplacer = rebuildNoiseReplacer(exposure, srcCat)


# In[6]:


# Now it actually does something
times = timeRunPlugins()
print(times)


# In[7]:


# Setup again
parents = np.unique(srcCat[parentkey][children])
idsToRerun = np.concatenate((srcCat["id"][children], parents))

newSrcCatalog = makeRerunCatalog(schema, srcCat, idsToRerun, fields=fields)
# It will not run unless it's sorted by parent key
newSrcCatalog.sort(parentkey)


# In[8]:


# It should take slightly longer sincec it's doing parents too
times = timeRunPlugins()
print(times)


# In[50]:


from lsst.afw.table import Schema, SourceCatalog
from collections import Iterable

def makeRerunCatalogFixed(schema, oldCatalog, idList, fields=None,
    resetParents=True, addParents=False, addSiblings=False):
    """ Creates a catalog prepopulated with ids
    This function is used to generate a SourceCatalog containing blank records
    with Ids specified in the idList parameter
    This function is primarily used when rerunning measurements on a footprint.
    Specifying ids in a new measurement catalog which correspond to ids in an
    old catalog makes comparing results much easier.
    Note that the new catalog will be sorted by id.
    Parameters
    ----------
    schema : lsst.afw.table.Schema
        Schema used to describe the fields in the resulting SourceCatalog
    oldCatalog : lsst.afw.table.SourceCatalog
        Catalog containing previous measurements.
    idList : iterable
        Python iterable whose values should be numbers corresponding to
        measurement ids, ids must exist in the oldCatalog
    fields : iterable
        Python iterable whose entries should be strings corresponding to schema
        keys that exist in both the old catalog and input schema. Fields listed
        will be copied from the old catalog into the new catalog.
    resetParents: boolean
        Flag to indicate that child objects should have their parents set to 0.
        Otherwise, lsst.meas.base.SingleFrameMeasurementTask.runPlugins() will
        skip these ids unless their parents are also included in idList.
    addParents: boolean
        Flag to toggle whether parents of child objects will be added to the 
        idList (if not already present).

    Returns
    -------
    measCat : lsst.afw.table.SourceCatalog
        SourceCatalog prepopulated with entries corresponding to the ids
        specified
    """
    
    if not isinstance(schema, lsst.afw.table.Schema):
        raise RuntimeError("schema must be an lsst.afw.table.Schema")
        
    if not isinstance(oldCatalog, lsst.afw.table.SourceCatalog):
        raise RuntimeError("oldCatalog must be an "
                           "lsst.afw.table.SourceCatalogiterable")
    
    if fields is None:
        fields = []
    if not isinstance(fields, Iterable):
        raise RuntimeError("fields list must be an iterable with string"
                           "elements")

    for entry in fields:
        if entry not in schema:
            schema.addField(oldCatalog.schema.find(entry).field)

    # It's likely better to convert to a list and append
    idList = list(idList)
            
    if addParents:
        lenIdList = len(idList)
        for idx in range(lenIdList):
            srcId = idList[idx]
            oldSrc = oldCatalog.find(srcId)
            parent = oldSrc.getParent()
            if parent != 0 and not parent in idList:
                idList.append(parent)

    idList.sort()

    measCat = SourceCatalog(schema)
    for srcId in idList:
        oldSrc = oldCatalog.find(srcId)
        src = measCat.addNew()
        src.setId(srcId)
        src.setFootprint(oldSrc.getFootprint())
        parent = oldSrc.getParent()
        if parent != 0 and resetParents and parent not in idList:
            parent = 0
        src.setParent(parent)
        src.setCoord(oldSrc.getCoord())
        for entry in fields:
            src[entry] = oldSrc[entry]
        
    return measCat


# In[51]:


# So the full path of lsst.afw.table is visible
import lsst

# Assume only children again
idsToRerun = srcCat["id"][children]

# What it used to do
newSrcCatalog = makeRerunCatalogFixed(
    schema, srcCat, idsToRerun, fields=fields, resetParents=False
)
print("Old (resetParents=False): ", timeRunPlugins())

# The new default resets parents
newSrcCatalog = makeRerunCatalogFixed(schema, srcCat, idsToRerun, fields=fields)
print("New (default resetParents=True): ", timeRunPlugins())

# Run the parents as well
newSrcCatalog = makeRerunCatalogFixed(
    schema, srcCat, idsToRerun, fields=fields, addParents=True,
    resetParents=False
)
print("New (addParents=True, resetParents=False): ", timeRunPlugins())

# This should be functionally identical while unnecessarily throwing away 
# parent information - not sure if it should be forbidden?
newSrcCatalog = makeRerunCatalogFixed(
    schema, srcCat, idsToRerun, fields=fields, addParents=True,
    resetParents=True
)
print("New (addParents=True, resetParents=True): ", timeRunPlugins())

