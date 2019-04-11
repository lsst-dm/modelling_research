#!/usr/bin/env python
# coding: utf-8

# # Fitting COSMOS galaxies with MultiProFit
# 
# This notebook generates a human- and machine-readable table (CSV) with results from fits to galaxies from the COSMOS survey (http://cosmos.astro.caltech.edu/) using MultiProFit (https://github.com/lsst-dm/multiprofit). For general motivation, explanation and analysis of plots generated from that table, check out other notebooks in this series - this is just about taking pickle fit results and turning them into a table. At some future time, this code may be simplified (hopefully) and included within MultiProFit or a related package.

# ## Reproducing the galaxy fits
# 
# Firstly, you'll need to install [MultiProFit](https://github.com/lsst-dm/multiprofit).
# 
# To fit true Sersic models, you'll need one of [GalSim](https://github.com/GalSim-developers/GalSim/) or [PyProFit](https://github.com/ICRAR/pyprofit/). Despite the fact that I am an author of ProFit, I suggest using GalSim as its rendering methods are faster and/or more robust in most cases. The only exception is that its more stringent accuracy requirements mean that it can be much slower or consume large amounts of RAM evaluating some parameter combinations, but that's more of an indictment of the Sersic profile than of GalSim.
# 
# For fitting HST images, these scripts make use of the [GalSim COSMOS 25.2 training sample](https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data). This catalog contains cutouts of HST-COSMOS galaxies with F814W magnitudes < 25.2 mag, designed to test algorithms for weak lensing shear measurements, which typically have stricter requirements than galaxy models for galaxy evolution science. For more information on the sample, consult [the readme](http://great3.jb.man.ac.uk/leaderboard/data/public/COSMOS_25.2_training_sample_readme.txt) or the original paper ([Leauthaud et al. 2010](http://adsabs.harvard.edu/abs/2010ApJ...709...97L)).
# 
# For fitting HSC images, these scripts make use of the LSST software pipelines (https://github.com/lsst/) to access data available on LSST servers. HSC periodically releases data publicly and you can 'quarry' (download images) from here after registering an account: https://hsc-release.mtk.nao.ac.jp/doc/index.php/tools/. The older PyProFit fork had an [example script](https://github.com/lsst-dm/pyprofit/blob/master/examples/hsc.py) using this public data, which will be ported to MultiProFit one day.
# 
# This notebook does not yet actually show the results of fitting HSC images - the multiband fits are a work in progress. The mock HSC-quality images are generated from a best-fit HST model degraded to HSC resolution using the actual i-band HSC UltraDeep coadd PSF (calling getPsf()), with added noise according to the observed HSC variance map. These measurements therefore include noise bias and model inadequacy bias for the PSF and galaxy (for galaxy models that cannot reproduce the input model). Deblending (contamination from overlapping sources) is also an issue in HSC and occasionally even in HST, but we only use images that have already been deblended (for better or worse).
# 
# The HST filter used in COSMOS ([F814W](http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=HST/WFPC2.f814w)) approximately covers HSC/SDSS i- and z-bands ([see here](http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=Subaru&gname2=HSC)), so a multi-band fit to HSC i+z should also be consistent with HST F814W. For single-band fits, HSC-I is more convenient as it is broader, tends to have better seeing and covers a larger fraction of F814W.

# ## Galaxy fitting scripts
# 
# The data analyzed here were generated using MultiProFit's [fitcosmos.py example script](https://github.com/lsst-dm/multiprofit/blob/master/examples/fitcosmos.py).
# 
# MultiProFit allows for flexible fitting workflows specified in a CSV file; the specifications used here are also [part of the repo](https://github.com/lsst-dm/modelling_research/blob/master/cosmos/modelspecs-hst-mg.csv).
# 
# fitcosmos.py was run in batches of 100 galaxies using the following invocations. The choice of batches isn't important; you could output one file per galaxy if you so desired. The exact commands run are as follows:
# 
# #Fit Sersic-family models to HST images
# python ~/src/mine/multiprofit/examples/fitcosmos.py -catalogpath ~/raid/hsc/cosmos/COSMOS_25.2_training_sample/ -catalogfile real_galaxy_catalog_25.2.fits -indices 0,100 -fithst 1 -modelspecfile ~/raid/lsst/cosmos/modelspecs-mg.csv -fileout ~/raid/lsst/cosmos/cosmos_25.2_fits_0_100_pickle.dat > ~/raid/lsst/cosmos/cosmos_25.2_fits_0_100.log 2>~/raid/lsst/cosmos/cosmos_25.2_fits_0_100.err &
# #Fit GMMs to HST images
# python ~/src/mine/multiprofit/examples/fitcosmos.py -catalogpath ~/raid/hsc/cosmos/COSMOS_25.2_training_sample/ -catalogfile real_galaxy_catalog_25.2.fits -indices 0,100 -fithst 1 -modelspecfile ~/raid/lsst/cosmos/modelspecs-mg.csv -fileout ~/raid/lsst/cosmos/cosmos_25.2_fits_0_100_pickle.dat -redo 0 > ~/raid/lsst/cosmos/cosmos_25.2_fits_0_100_mg8.log 2>~/raid/lsst/cosmos/cosmos_25.2_fits_0_100_mg8.err &
# #Fit all models to realistic mock HSC images taking the HST Sersic fit as the ground truth
# python ~/src/mine/multiprofit/examples/fitcosmos.py -catalogpath ~/raid/hsc/cosmos/COSMOS_25.2_training_sample/ -catalogfile real_galaxy_catalog_25.2.fits -indices 0,100 -fithst2hsc 1 -modelspecfile ~/raid/lsst/cosmos/modelspecs-mg.csv -redo 0 -fileout ~/raid/lsst/cosmos/cosmos_25.2_fits_0_100_pickle.dat -hst2hscmodel mgserbpx > ~/raid/lsst/cosmos/cosmos_25.2_fits_0_100_hst2hsc_ser.log 2>~/raid/lsst/cosmos/cosmos_25.2_fits_0_100_hst2hsc_ser.err &
# 
# #Experimental multi-band fit with double Gaussian PSF to a fairly well-resolved galaxy (this will take a while):
# python fitcosmos.py -catalogpath ~/raid/hsc/cosmos/COSMOS_25.2_training_sample/ -catalogfile real_galaxy_catalog_25.2.fits -hscbands 'HSC-I' 'HSC-R' 'HSC-G' -indices 1102 -fithsc 1 -modelspecfile ~/raid/lsst/cosmos/modelspecs-plot-final-psfg2.csv -plot 1 -imgplotmaxs 40.0 20.0 10.0 -imgplotmaxmulti 80 -weightsband 1 1.16 1.79
# 

# ### Import required packages

# In[1]:


import astropy as ap
from collections import OrderedDict
import galsim as gs
import glob
import multiprofit.objects as mpfobj
# Function used to compute percentile radii for Gaussian mixtures
from multiprofit.gaussutils import multigauss2drquant
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pickle


# ### Read COSMOS catalog and pickled MultiProFit results
# 
# The example script to generate these results can be found at https://github.com/lsst-dm/multiprofit/blob/master/examples/fitcosmos.py (for now).
# 
# The COSMOS catalog is from GalSim: https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy-Data

# In[2]:


path = os.path.expanduser('/project/dtaranu/cosmos/hst/COSMOS_25.2_training_sample/')
file = "real_galaxy_catalog_25.2.fits"
ccat = gs.COSMOSCatalog(file, dir=path)
rgcfits = ap.io.fits.open(os.path.join(path, file))[1].data


# In[3]:


data = []
 
files = glob.glob(os.path.expanduser(
    "~/raid/lsst/cosmos/cosmos_25.2_fits_hst*_[01234579]*99_psfg2_pickle.dat"))
files.sort()
for file in files:
    with open(file, 'rb') as f:
        data.append(pickle.load(f))


# ### Define the table column names
# 
# This (admitted ugly) section defines column names and indices for reading the COSMOS fits, followed by a number of functions to read MultiProFit fit results into a dict.

# In[4]:


# See https://github.com/GalSim-developers/GalSim/blob/8d9bc8ce568e3fa791ab658650fce592cdf03735/galsim/scene.py
# lines 615-625
# Presumably copypasta'd from the original COSMOS fit catalog table/paper

#     SERSICFIT[0]: intensity of light profile at the half-light radius.
#     SERSICFIT[1]: half-light radius measured along the major axis, in units of pixels
#                   in the COSMOS lensing data reductions (0.03 arcsec).
#     SERSICFIT[2]: Sersic n.
#     SERSICFIT[3]: q, the ratio of minor axis to major axis length.
#     SERSICFIT[4]: boxiness, currently fixed to 0, meaning isophotes are all
#                   elliptical.
#     SERSICFIT[5]: x0, the central x position in pixels.
#     SERSICFIT[6]: y0, the central y position in pixels.
#     SERSICFIT[7]: phi, the position angle in radians.  If phi=0, the major axis is
#                   lined up with the x axis of the image.

scalesources = {
    'hst': 0.03,
    'hst2hsc': 0.168,
}
paramscosmos = ["IDENT", "mag_auto", "flux_radius", "zphot", "use_bulgefit", "viable_sersic"]
colnames = ["id", "ra", "dec"] + [".".join(["cosmos", param]) for param in paramscosmos]
paramsser = ["flux", "re", "nser", "q", "phi", "x0", "y0"]
# This puts it in dev, exp order
idxparamscosmos = [[x + offset for x in [1, 2, 3, 7, 5, 6]] for offset in [8, 0]]


# In[5]:


# Some hideous code to get all of the right values in the right order, for which I apologize

# The COSMOS catalog has fewer objects than the RealGalaxyCatalog for ??? reasons - presumably some failed fits are excluded?
# Get the mapping between RGC indices (which is what I used) and CosmosCat ones (for previous fits)
indexmap = {ccat.getOrigIndex(i): i for i in range(ccat.getNObjects())}
results = {}
paramsprofit = ["chisqred", "time"]

def isflux(param):
    return isinstance(param, mpfobj.FluxParameter)


def towriteparam(name, value, fixed):
    return (
        name != 'cenx' and name != 'ceny' and name != 'rs' and (not fixed or name == 're')
        #not (name == 'nser' and fixed and value == 0.5)
    )


def towritemodel(name):
    return not name.endswith('devexptpx')


def getmgcomponentre(paramsflux, paramsre, idxs):
    fluxes = []
    res = []
    for idx in idxs:
        fluxes.append(paramsflux[idx].getvalue(transformed=False))
        res.append(paramsre[idx].getvalue(transformed=False))
    fluxes = np.array(fluxes)
    fluxes /= np.sum(fluxes)
    return multigauss2drquant(list(zip(fluxes, res)))


def readmodel(profit, results, colnamesprofit, idx, src, colnamestoadd=None, engine='galsim', verbose=False):
    profitengine = profit['fits'][engine]
    for namemodel, profitmodel in profitengine.items():
        if 'fits' in profitmodel and towritemodel(namemodel):
            profitfits = profitmodel['fits']
            for idxfit, profitfit in enumerate(reversed(profitfits)):
                fixedall = profitfit['paramsallfixed']
                namesall = [x.name for x in profitfit['params']]
                paramsall = profitfit['params']
                # the parameter objects might have different values from paramsbestall
                # and we want to check their properties, not just their values
                for param, value in zip(paramsall, profitfit['paramsbestall']):
                    param.setvalue(value, transformed=False)
                lentype = {typeofval: len(vals) for typeofval, vals in [('names', namesall), ('params', paramsall), ('fixed', fixedall)]}
                if not (lentype['params'] == lentype['names'] and lentype['params'] == lentype['fixed']):
                    print("idx={} model={} src={} value lengths don't all match: {}".format(idx, namemodel, src, lentype))
                colnames = [param for param in paramsprofit]
                values = [profitfit[param] for param in paramsprofit]
                counts = {}
                allgauss = True
                componentsmg = []
                fluxscale = 1.0
                # Always store the total flux and skip the component flux for one-component models
                if src == 'hst2hsc' and 'fluxscalehst2hsc' in profit['metadata']:
                    fluxscale = profit['metadata']['fluxscalehst2hsc']
                paramsflux = [param for param in paramsall if isflux(param)]
                colnames.append('flux')
                skipflux = len(paramsflux) == 1
                if namemodel == 'mg8devexptpx':
                    print(skipflux, namesall, fixedall)
                value = np.sum([param.getvalue(transformed=False) for param in paramsflux])
                values.append(value*fluxscale)
                paramsre = [param for param in paramsall if param.name == 're']
                for param in paramsre:
                    param.setvalue(param.getvalue(transformed=False)*scalesources[src], transformed=False)
                for nameparam, param, fixed in zip(namesall, paramsall, fixedall):
                    value = param.getvalue(transformed=False)
                    if nameparam == 'nser':
                        allgauss = allgauss and value == 0.5
                    isfluxparam = isflux(param)
                    if not (skipflux and isfluxparam) and towriteparam(nameparam, value, fixed):
                        # Check if this angle (ellipse) value is shared, indicating that it ought to be a multi-Gaussian component
                        if nameparam == 'ang':
                            # TODO: This check may need to be much smarter in the future.
                            componentsmg.append(1 + 0 if not param.inheritors else len(param.inheritors))
                        if nameparam in counts:
                            counts[nameparam] += 1
                        else:
                            counts[nameparam] = 1
                        # TODO: Revisit the choice to apply modifiers in getprofiles(), which necessitates this
                        for modifier in param.modifiers:
                            if modifier.name == "rscale":
                                value *= modifier.getvalue(transformed=False)
                        values.append(value)
                        colnames.append(('rs' if nameparam == 'rscale' else nameparam) + '.' + str(counts[nameparam]))
                if any(np.array(componentsmg) > 1):
                    if allgauss:
                        for idxcol, colname in enumerate(colnames):
                            if colname.startswith('re.'):
                                colnames[idxcol] = 'regc.' + colname[3:]
                        counts['re'] = 0
                    else:
                        raise RuntimeError(
                            'Got componentsmg={} but not allgauss because nsers={}; shared ellipse models expected to be multi-Gauss'.format(
                                componentsmg, [param.getvalue(transformed=False) for param in paramsall if param.name == 'nser']
                        ))
                    idxcomp = 0
                    # Rename the effective radii of gaussian (sub)components to 'regc' and compute re from subcomponents
                    for nsub in componentsmg:
                        if nsub > 1:
                            idxsub = range(idxcomp, idxcomp+nsub)
                            if 're' in counts:
                                counts['re'] += 1
                            else:
                                counts['re'] = 1
                            colnames.append('re.' + str(counts['re']))
                            values.append(getmgcomponentre(paramsflux, paramsre, idxsub))
                        idxcomp += nsub
                if namemodel in colnamesprofit:
                    if colnames != colnamesprofit[namemodel][0]:
                        raise RuntimeError("colnames={} for idx={} model={} don't match existing {}".format(
                            colnames, idx, namemodel, colnamesprofit[namemodel][0]))
                    if idxfit >= colnamesprofit[namemodel][1]:
                        colnamesprofit[namemodel][1] = idxfit+1
                else:
                    colnamesprofit[namemodel] = [colnames, 1]
                if namemodel not in results:
                    results[namemodel] = []
                results[namemodel].append(values)
                if verbose:
                    print('readmodel src={} idx={} idxfit={} model={} results={} colnames={}'.format(
                        src, idx, idxfit, namemodel, values, colnames))


def readfits(profit, results, colnamesprofit, idx, src, engine='galsim', verbose=False):
    hasfits = 'fits' in profit
    if hasfits:
        readmodel(profit, results, colnamesprofit, idx, src, engine=engine, verbose=verbose)
    if verbose:
        print('readfits src={} idx={} model={} profit.keys()={} hasfits={}'.format(src, idx, model, profit.keys(), hasfits))
    return hasfits


def readsrc(dataidx, results, colnamesprofit, idx, src, engine='galsim', verbose=False):
    hasfits = src in datatab[idx]
    if src not in results:
        results[src] = {}
    if hasfits:
        hasfits = readfits(dataidx[src], results[src], colnamesprofit, idx, src, engine=engine, verbose=verbose)
    if verbose:
        print('readsrc src={} idx={} datatab[idx].keys={} profit.keys()={} hasfits={}'.format(src, idx, model, profit.keys(), hasfits))
        print(src, idx, datatab[idx].keys(), model, hasfits, stage)
    return hasfits


# ### Read the MultiProFit results
# 
# Continuing the trend of unseemly code, this section parses every MultiProFit result pickle into various data types using the functions above, then combines those into a list of rows.

# In[6]:


# Read all of the results into a massive and unwieldy dict
verbose = False
printrow = False
hasaddedprofitcolnames = False
colnamesprofit = OrderedDict()
idxsubcomponentfluxes = {}
results = {}
for datatab in data:
    printrowdata = printrow
    appended = 0
    for idx in datatab:
        if isinstance(datatab[idx], dict) and idx in indexmap:
            if idx not in results:
                results[idx] = {}
            for src in scalesources:
                hasfitssrc = readsrc(datatab[idx], results[idx], colnamesprofit, idx, src, verbose=verbose or printrowdata)
                if printrowdata and hasfitssrc:
                    printrowdata = False


# In[7]:


# TODO: Actually search for it
idxtime = 1
colnamestable = (
    ["id", "ra", "dec"] + 
    [".".join(["cosmos", param]) for param in paramscosmos] +
    [".".join(['cosmos.hst.ser', param]) + ('.1' if param != 'flux' else '') for param in paramsser] +
    [".".join(['cosmos.hst.devexp', param]) for param in 
        [param + comp for comp in [".1", ".2"] for param in paramsser]]
)
ncolnamestablenonprofit = len(colnamestable)
for src in scalesources:
    for namemodel, (colnames, numfits) in colnamesprofit.items():
        if towritemodel(namemodel):
            colnamestable.append('.'.join(['profit', src, namemodel, "time"]))
            for numfit in range(numfits):
                for colname in colnames:
                    colnamestable.append('.'.join(['profit', src, namemodel, colname, str(numfit+1)]))
ncolnamestable = len(colnamestable)
rows = []
for idx, result in results.items(): #data:
    if idx in results:
        row = [idx] + list(rgcfits[idx][1:3])
        rec = ccat.getParametricRecord(indexmap[idx])
        row += [rec[param] for param in paramscosmos]
        row += [rec["flux"][0]] + list(rec["sersicfit"][idxparamscosmos[1]])
        if printrow:
            print(len(row), 'cosmos.ser')
        for offset in range(2):
            row += [rec["flux"][1+offset]] + list(rec["bulgefit"][idxparamscosmos[offset]])
            if printrow:
                print(len(row), 'cosmos.devexp.' + str(offset))
            printrow = False
        resultsidx = results[idx]
        if len(row) != ncolnamestablenonprofit:
            raise RuntimeError('Pre-ProFit len(row)={} != ncolnamestablenonprofit={} for idx={} src={}'.format(
                len(row), ncolnamestablenonprofit, idx, src))
        for src in scalesources:
            resultssrc = resultsidx[src] if src in resultsidx else None
            for namemodel, (colnames, numfits) in colnamesprofit.items():
                numfitsfound = 0
                idxtimerow = len(row)
                row.append(0)
                timetotal = 0
                if resultssrc is not None and namemodel in resultssrc:
                    for values in resultssrc[namemodel]:
                        if len(values) != len(colnames):
                            raise RuntimeError('len(values)={} != len(colnames)={}'.format(
                                len(values), len(colnames)))
                        row += values
                        numfitsfound += 1
                        timetotal += values[idxtime]
                row[idxtimerow] = timetotal
                if numfitsfound > numfits:
                    raise RuntimeError('numfitsfound={} > numfitsexpected={} for idx={} src={} model={}'.format(
                        numfitsfound, numfits, idx, src, namemodel))
                numfitsmissing = numfits - numfitsfound
                if numfitsmissing != 0:
                    print('numfitsmissing={} for idx={} src={} model={}'.format(
                        numfitsmissing, idx, src, namemodel))
                    row += list(np.repeat(np.nan, len(colnames)*numfitsmissing))
        if len(row) != ncolnamestable:
            raise RuntimeError('len(row)={} != ncolnamestable={} for idx={} src={}'.format(
                len(row), ncolnamestable, idx, src))
        rows.append(row)
print(len(rows))
print(colnamestable)


# ### Writing a table with the results
# 
# The table of results is a bit too large to be human-readable, but you can analyze it any which way you like with the column names saved.

# In[8]:


# Write to a plain old CSV, then read it back in to double-check
import csv

with open(os.path.join(path, "galfits.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([colnamestable])
    writer.writerows(rows)

