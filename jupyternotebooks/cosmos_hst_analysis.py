
# coding: utf-8

# # Fitting COSMOS galaxies with MultiProFit
# 
# This notebook plots results from fitting galaxies from the COSMOS survey (http://cosmos.astro.caltech.edu/) with MultiProFit (https://github.com/lsst-dm/multiprofit). This investigation is to help determine what kind of galaxy models LSST Data Management should fit for annual Data Releases, where we would like to fit the best possible model(s) but will be constrained in computing time by the overwhelming size of the dataset (billions of objects).

# ## Introduction and motivation
# 
# Currently, the LSST Science Pipelines fit a limited set of models to all sources: point source, exponential (exp., Sersic n=1), de Vaucouleurs (deV., Sersic n=4), and an SDSS-style cModel (the best linear combination of the exp. and deV. models). The code for this is implemented in the [meas_modelfit package](https://github.com/lsst/meas_modelfit). meas_modelfit uses a double-shapelet (perturbed Gaussians; see [Refregier 2001](https://arxiv.org/abs/astro-ph/0105178)) PSF model and multi-Gaussian approximations to Sersic profiles ([Hogg & Lang 2012](https://arxiv.org/abs/1210.6563)); this makes the convolution analytic and model evaluations quite efficient. meas_modelfit also implements a custom optimizer with a variety of settings, some of which were tested in [this notebook](https://github.com/lsst-dm/modelling_research/blob/master/jupyternotebooks/lsst_cmodel_configs.ipynb).
# 
# However, meas_modelfit does not implement many of the models used most commonly in galaxy evolution literature, including the Sersic profile and more complicated multi-component bulge + disk models. Various authors have generated catalogs of these sorts of fits using SDSS data (e.g. [Simard+2011](https://arxiv.org/abs/1107.1518)). The MultiProFit code was designed to implement alternative models of various complexity to those in meas_modelfit, and the purpose of this notebook is to determine which of these models should be used in LSST, which will have several orders of magnitude more sources to model than SDSS. For example, the Sersic profile is a non-linear model with one additional free (and very non-linear) parameter compared to the exponential/deVaucouleurs profiles. It is more expensive to fit than cModel, whose additional parameter (fracDev) is linear, but it covers a different space of possible models than cModel. Similarly, free(r) bulge + disk models have still more free parameters and are probably better parameterizations for disk galaxies, but they are much more expensive to fit and may not be a significant improvement for early-type and/or faint/poorly resolved galaxies.
# 
# A couple of other points need to be addressed before moving on:
# 
# 1. Why COSMOS? COSMOS has a full square degree of very deep, high-resolution Hubble Space Telescope (HST) imaging in the F814W band. This matches up nicely with the slightly deeper but lower resolution ground-based grizy imaging from Subaru Hyper-Suprime Cam (HSC). The approach we take here is to fit the HST images first, the higher resolution being preferable for resolved features like compact bulges and for minimizing the impact of blending (overlapping sources). The ultimate goal is to fit HSC-UltraDeep images, which are of comparable resolution and depth to 10-year LSST images (actually slightly better resolution and deeper, but that's fine). The fits to HSC data can be done in an experimental multi-band mode. An intermediate step involves taking a best-fit HST model, generating a mock HSC-quality image, and then fitting this mock image. This is meant to gauge how well the true parameters are recovered both in cases where we used the correct model - that is, when the model used to fit HSC-quality data can perfectly represent the true HST model - and also to measure the bias when fitting wrong models that can't represent the true galaxy profile exactly.
# 
# 2. All of the models tested here are either a) traditional Sersic profiles or b) Gaussian mixture models (GMMs). In fact, most of the 'Sersic' models are multi-Gaussian approximations thereof (MGAs). For more details, see [this notebook](https://github.com/lsst-dm/modelling_research/blob/master/jupyternotebooks/multigaussian_sersic1d.ipynb) on deriving optimal MGAs for the Sersic profile.

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

# ## Browsing this notebook
# 
# I recommend using jupyter's nbviewer page to browse through this notebook. For example, you can use it to open the [N=4 GMM](https://nbviewer.jupyter.org/github/lsst-dm/modelling_research/blob/master/jupyternotebooks/cosmos_hst_analysis.ipynb#COSMOS-HST:-MultiProFit-Sersic-vs-MultiProFit-MGA-Sersic-(N=4) and compare to the [N=8 GMM](https://nbviewer.jupyter.org/github/lsst-dm/modelling_research/blob/master/jupyternotebooks/cosmos_hst_analysis.ipynb#COSMOS-HST:-MultiProFit-Sersic-vs-MultiProFit-MGA-Sersic-(N=8) side-by-side.

# ### Analyze the results
# 
# Import required packages and set matplotlib/seaborn defaults for slightly nicer plots.

# In[1]:


import astropy as ap
import galsim as gs
import glob
import matplotlib as mpl
import matplotlib.colors as mplcol
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

# Setup for plotting
get_ipython().magic('matplotlib inline')

#plt.style.use('seaborn-notebook')
sns.set_style('darkgrid')
mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['image.origin'] = 'lower'
sns.set(rc={'axes.facecolor': '0.85', 'figure.facecolor': 'w'})


# ### Compute R_eff for MGA
# 
# These functions compute the effective (half-light) radius R_eff for MGA profiles. These should almost exactly equal the nominal Sersic R_eff for the Sersic MGA for values of n fitted over the full range of r/R_eff, but for n>2 the truncation at large radii and any exclusion of the inner part of the profile from the fit will change R_eff. These functions can also be used to measure R_eff for any Gaussian mixture model consisting of components with shared ellipse parameters (i.e. isophote shapes). For Gaussian mixtures with independent ellipse shapes, the definition of R_eff is more ambiguous. One could compute the isophote contain a given percentile of the flux from oversampled images, but it will not necessarily be elliptical in shape.

# In[2]:


# https://www.wolframalpha.com/input/?i=Integrate+2*pi*x*exp(-x%5E2%2F(2*s%5E2))%2F(s*sqrt(2*pi))+dx+from+0+to+r
# The fraction of the total flux of a 2D Sersic profile contained within r
# For efficiency, we can replace r with r/sigma.
# Note the trivial renormalization allows us to drop annoying sigma and pi constants - it returns (0, 1) for inputs of (0, inf)
def gauss2dint(xdivsigma):
     return 1 - np.exp(-xdivsigma**2/2.)


# Compute the fraction of the integrated flux within x for a sum of Gaussians
# x is a length in arbitrary units
# Weightsizes is a list of tuples of the weight (total flux) and size (r_eff in the same units as x) of each gaussian
# Note that gauss2dint expects x/sigma, but size is re, so we pass x/re*re/sigma = x/sigma
# 0 > quant > 1 turns it into a function that returns zero at the value of x containing a fraction quant of the total flux
# This is so you can use root finding algorithms to find x for a given quant (see below)
def multigauss2dint(x, weightsizes, quant=0):
     retosigma = np.sqrt(2.*np.log(2.))
     weightsumtox = 0
     weightsum = 0
     for weight, size in weightsizes:
         weightsumtox += weight*(gauss2dint(x/size*retosigma) if size > 0 else 1)
         weightsum += weight
     return weightsumtox/weightsum - quant


import scipy.optimize as spopt
# Compute x_quant for a sum of Gaussians, where 0<quant<1
# There's probably an analytic solution to this if you care to work it out
# Weightsizes and quant are as above
# Choose xmin, xmax so that xmin < x_quant < xmax. Ideally we'd just set xmax=np.inf but brentq doesn't work then; a very large number like 1e5 suffices.
def multigauss2drquant(weightsizes, quant=0.5, xmin=0, xmax=1e5):
    if not 0 <= quant <= 1:
        raise ValueError('Quant {} not >=0 & <=1'.format(quant, quant))
    weightsumzerosize = 0
    weightsum = 0
    for weight, size in weightsizes:
        if not (size > 0):
            weightsumzerosize += weight
        weightsum += weight
    if weightsumzerosize/weightsum >= quant:
        return 0
    return spopt.brentq(multigauss2dint, a=xmin, b=xmax, args=(weightsizes, quant))


# ### Read COSMOS catalog and pickled MultiProFit results
# 
# The example script to generate these results can be found at https://github.com/lsst-dm/multiprofit/blob/master/examples/fitcosmos.py (for now).
# 
# The COSMOS catalog is from GalSim: https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy-Data

# In[3]:


path = os.path.expanduser('~/raid/hsc/cosmos/COSMOS_25.2_training_sample/')
file = "real_galaxy_catalog_25.2.fits"
ccat = gs.COSMOSCatalog(file, dir=path)
rgcfits = ap.io.fits.open(os.path.join(path, file))[1].data


# In[4]:


data = []
 
files = glob.glob(os.path.expanduser(
    "~/raid/lsst/cosmos/cosmos_25.2_fits_*_*_pickle.dat"))
files.sort()
for file in files:
    with open(file, 'rb') as f:
        data.append(pickle.load(f))


# ### Define the table column names
# 
# This (admitted ugly) section defines column names and indices for reading both the MultiProFit fit results and the results from the COSMOS-GalSim catalog itself (using Lackner&Gunn 2012 code) for a consistency check.

# In[5]:


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

sources = ["hst"] + ["_".join(["hst2hsc", postfix]) for postfix in [
#    "",
    "mg8serbpx",
#    "_devexp"
]]
params = {
    "cosmos": ["IDENT", "mag_auto", "flux_radius", "zphot", "use_bulgefit", "viable_sersic"],
    "profit": ["chisqred", "time"],
}
# Some hideous code to get all of the column names in the right order
# See above for COSMOS fits param order
idxparamscosmos = [[x + offset for x in [1, 2, 3, 7, 5, 6]] for offset in [0, 8]]
# profit was setup to output [cenx, ceny, flux, fluxrat, re, axrat, ang, nser]
idxprofit = [2, 4, 7, 5, 6, 0, 1]
idxprofittwo = [3, 4, 7, 5, 6, 0, 1, 8, 9, 12, 10, 11, 0, 1]
idxprofitmg8 = [2, 4, 5, 6, 0, 1, 3, 8, 13, 18, 23, 28, 33, 38, 4, 9, 14, 19, 24, 29, 34, 39]
idxprofitmg4 = [2, 4, 5, 6, 0, 1, 3, 8, 13, 18, 4, 9, 14, 19]
idxprofitmg4x2 = [2, 4, 5, 6, 0, 1, 3, 8, 13, 18, 23, 28, 33, 38, 4, 9, 14, 19, 24, 29, 34, 39, 24, 25, 26]

orders = ['8', '4']
idxparamsprofit = {
    "gausspx":     idxprofit,
}
mgmodels = {
    order: ['mg' + order + model + 'px' for model in ['exp', 'n2', 'dev2', 'serb', 'serbed']]
    for order in orders
}
for order in orders:
    for model in mgmodels[order]:
        idxparamsprofit[model] = idxprofit
idxparamsprofit.update(
    {
        "serbpx":  idxprofit,
        "serb":  idxprofit,
        "mg8bpx": idxprofitmg8,
        "mg4bpx": idxprofitmg4,
        "mg4x2px": idxprofitmg4x2
    }
)
mgmodelstwo = {
    order: ['mg' + order + model + 'px' for model in ['cmodel', 'devexp', 'devexpc', 'serserb']]
    for order in orders
}
for order in orders:
    for model in mgmodelstwo[order]:
        idxparamsprofit[model] = idxprofittwo
print(idxparamsprofit)

paramsser = ["flux", "re", "n", "q", "phi", "x0", "y0"]
paramsmg8 = ["flux", "re", "q", "phi", "x0", "y0", 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8']
paramsmg4 = ["flux", "re", "q", "phi", "x0", "y0", 'f1', 'f2', 'f3', 'f4', 'r1', 'r2', 'r3', 'r4']
paramsmg4x2 = ["flux", "re", "q", "phi", "x0", "y0", 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 're2', 'q2', 'phi2']
# Keep track of the 're' field for Gaussian mixture models and the indices of the weights and sizes of each component
idxmgweightsizes = {
    '4': [(1, np.arange(6, 6+4), np.arange(10, 10+4))],
    '8': [(1, np.arange(6, 6+8), np.arange(14, 14+8))],
    '4x2': [(1, np.arange(6, 6+4), np.arange(14, 14+4)), (22, np.arange(10, 10+4), np.arange(18, 18+4))],
}
paramweightsizeskeymodel = {
    'mg8bpx': (paramsmg8, '8'),
    'mg4bpx': (paramsmg4, '4'),
    'mg4x2px': (paramsmg4x2, '4x2'),
}
models = {
    "single": {
        "cosmos": ["ser"],
        "profit": ["gausspx"] + mgmodels['8'] + mgmodels['4'] + ["serbpx", "serb"],
    },
    "double": {
        "cosmos": ["devexp"],
        "profit": mgmodelstwo['8'] + mgmodelstwo['4'],
    },
    "mg": {
        "cosmos": [],
        "profit": ["mg8bpx", "mg4bpx", "mg4x2px"],
    },
}
modellers = {
    "cosmos": [None],
    "profit": sources,
}
colnames = {
    modeller if src is None else ".".join([modeller, src]): 
        [".".join([model, param]) for model in models["single"][modeller] for param in paramsser] +
        [".".join([model, param]) for model in models["mg"][modeller] for param in paramweightsizeskeymodel[model][0]] +
        [".".join([model, param]) for model in models["double"][modeller] for param in
         [comp + "." + param for comp in ["exp", "dev"] for param in paramsser]]
    for modeller, srcs in modellers.items() for src in srcs
}
colnames = (["id", "ra", "dec"] +
            [".".join(["cosmos", param]) for param in params["cosmos"]] +
            [".".join(["profit", src, model, param, str(idx)])
             for src in sources
             for model in idxparamsprofit.keys()
             for param in params["profit"]
             for idx in range(1 + (model in models['double']['profit'] or model in models['mg']['profit']))] +
            [".".join([prefix, x]) for prefix, colnames in colnames.items() for x in colnames])

print(len(colnames), 'colnames:', colnames)


# ### Read the MultiProFit results
# 
# Continuing the trend of unseemly code, this section reads every MultiProFit result pickle and saves a row with numbers in the same order as the column names above. TBD: Combine these into one.

# In[6]:


# Some hideous code to get all of the right values in the right order, for which I apologize

# The COSMOS catalog has fewer objects than the RealGalaxyCatalog for ??? reasons - presumably some failed fits are excluded?
# Get the mapping between RGC indices (which is what I used) and CosmosCat ones (for previous fits)
indexmap = {ccat.getOrigIndex(i): i for i in range(ccat.getNObjects())}

rows = []

# Shouldn't have hardcoded this but here we are
scaleratio = 0.168/0.03
verbose = False
printrow = False
hasaddedderivedcolnames = False
idxsubcomponentfluxes = {}
for datatab in data:
    appended = 0
    for idx in datatab:
        hasfits = False
        if isinstance(datatab[idx], dict) and idx in indexmap:
            hasfits = True
            row = [idx] + list(rgcfits[idx][1:3])
            rec = ccat.getParametricRecord(indexmap[idx])
            row += [rec[param] for param in params["cosmos"]]
            for src in sources:
                stage = 0
                hasfits = hasfits and src in datatab[idx]
                if hasfits:
                    stage = 1
                    profit = datatab[idx][src]
                    hasfits = hasfits and 'fits' in profit
                    if hasfits:
                        stage = 2
                        profit = profit['fits']
                        for model in idxparamsprofit:
                            hasfits = hasfits and model in profit['galsim']
                            if hasfits:
                                stage = 3
                                profitmodel = profit['galsim'][model]
                                hasfits = hasfits and 'fits' in profitmodel
                                if hasfits:
                                    stage = 4
                                    profitmodel = profitmodel['fits']
                                    row += [profitmodelfit[param]
                                            for param in params["profit"]
                                            for profitmodelfit in profitmodel]
                                    if printrow:
                                        print(src, idx, len(row), model, params['profit'], len(profitmodel), hasfits)
                                elif verbose:
                                    print(src, idx, model, profitmodel.keys(), hasfits, stage)
                            elif verbose:
                                print(src, idx, model, profit['galsim'].keys(), hasfits, stage)
                    elif verbose:
                        print(src, idx, model, profit.keys(), hasfits, stage)
                elif verbose:
                    print(src, idx, datatab[idx].keys(), model, hasfits, stage)
                if printrow:
                    print(src, idx, datatab[idx].keys(), model, hasfits, stage)
            if hasfits:
                row += [rec["flux"][0]] + list(rec["sersicfit"][idxparamscosmos[0]])
                if printrow:
                    print(len(row), 'cosmos.ser')
                for offset in range(2):
                    row += [rec["flux"][1+offset]] + list(rec["bulgefit"][idxparamscosmos[offset]])
                    if printrow:
                        print(len(row), 'cosmos.devexp.' + str(offset))
                for src in sources:
                    profit = datatab[idx][src]['fits']
                    for model, idxs in idxparamsprofit.items():
                        twocomp = model in models['double']['profit']
                        values = np.array(profit['galsim'][model]['fits'][-1]["paramsbestall"])
                        if 'fluxscalehst2hsc' in datatab[idx][src]['metadata']:
                            values[2] /= datatab[idx][src]['metadata'][
                                'fluxscalehst2hsc']
                            values[4] *= scaleratio
                        if twocomp:
                            # Subtract first component fluxfrac
                            values[8] = 1 - values[3]
                            values[9] *= scaleratio
                            for col in [3, 8]:
                                values[col] *= values[2]
                        # This is a bit ugly - replace the placeholder re value with a proper calculation
                        if model in models['mg']['profit']:
                            idxs = np.array(idxs)
                            for idxsparams in idxmgweightsizes[paramweightsizeskeymodel[model][1]]:
                                weights = values[idxs[idxsparams[1]]]
                                weightsum = 1.0
                                for idxweight, weight in enumerate(weights):
                                    weight *= weightsum
                                    weightsum -= weight
                                    weights[idxweight] = weight
                                # Ensure that the weights sum to unity += machine eps.
                                weights[-1] = 1.0-np.sum(weights[:-1])
                                sizes = values[idxs[idxsparams[2]]]
                                reff = multigauss2drquant(list(zip(weights, sizes)))
                                if not (reff > 0):
                                    print(idxsparams)
                                    print('Unresolved component for src={}, idx={}, model={}, weightsizes: {}, {}'.format(src, idx, model, weights, sizes))
                                values[idxs[idxsparams[0]]] = reff
                        row += list(values[idxs])
                        if printrow:
                            print(len(row), model)
                    printrow = False
        if hasfits:
            # Build a list of derived summed quantities
            if not hasaddedderivedcolnames:
                colnamesarr = np.array(colnames)
                for modeller, srcs in modellers.items():
                    for src in srcs:
                        prefix = modeller if src is None else ".".join([modeller, src])
                        for model in models['double'][modeller]:
                            colname = '.'.join([prefix, model, 'flux'])
                            # Indices of all of the columns to add up to get the total flux (basically just *.exp.flux and *.dev.flux)
                            idxsubcomponentfluxes[colname] = np.array(
                                [np.where(colnamesarr == param)[0][0] for param in ['.'.join([prefix, model, comp, 'flux'])
                                                                                 for comp in ['exp', 'dev']]])
                            colnames.append(colname)
            hasaddedderivedcolnames = True
            # Actually compute the derived sums
            for fluxidxs in idxsubcomponentfluxes.values():
                row.append(np.sum(np.array([row[idx] for idx in fluxidxs])))
            rows.append(row)
            appended += 1
    listids = datatab.keys()
    print("Read {}/{} rows from {}-{}".format(
        appended, len(datatab), min(listids),
        max(listids)))

print(colnames)


# ### Writing a table with the results
# 
# The table of results is a bit too large to be human-readable, but you can analyze it any which way you like with the column names saved.

# In[7]:


# Write to a plain old CSV, then read it back in to double-check
import csv

with open(os.path.join(path, "galfits.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([colnames])
    writer.writerows(rows)
    


# In[8]:


tab = pd.read_csv(os.path.join(path, "galfits.csv"))


# ### Make joint parameter plots
# 
# This section makes an alarmingly large number of plots. Most of them check consistency between the MultiProFit fits and the COSMOS-GalSim catalog values with marginalized histograms thereof. The last set compares the results from HST fits to synthetic HSC-quality images of the same galaxy. More specifically, we take the best-fit HST F814W model (single MG8 Sersic for now), convolve it with the reconstructed HSC i-band PSF, re-normalize the magnitude to match the original HSC image since they're different resolutions and the F814W band is wider than HSC-I, and add noise based on the observed HSC inverse variance map (which isn't completely consistent with the 'true' model, but it's close enough and saves the effort of having to back out the HSC background, etc.).
# 
# Notes:
# 
# The colour coding is by log(Sersic index) from the HST true Sersic fit, such that low values (~disk-like) are blue and high values (~bulge-like) are red.
# Any linear relations between points in plots of the Sersic index are due to the different limits imposed by the various fit methods: n=[0.1, 6] in the GalSim catalog; n=[0.3, 6.3] for MultiProFit Sersic fits (set by GalSim), and n=[0.5, 6.3] for MultiProFit MGA Sersic.

# In[9]:


# How well are parameters recovered?
# First check consistency with previous fits (Sersic only for now)
# Then HSC vs HST
# TODO: Rename this: prefix -> var and var->postfix
def getname(var, prefix, postfix):
    namevar = ".".join([prefix, var]) if prefix is not None else var
    namevar = ".".join([namevar, postfix]) if postfix is not None else namevar
    return namevar


def jointplotcolorbar(jointplot, label=''):
    plt.setp(jointplot.ax_marg_x.get_yticklabels(), visible=True)
    plt.setp(jointplot.ax_marg_y.get_xticklabels(), visible=True)
    ax = jointplot.ax_marg_y
    with sns.axes_style("ticks"):
        cax = jointplot.fig.add_axes([.85, .85, .12, .05])
        cbar = plt.colorbar(cax=cax, orientation='horizontal', ticklocation='top')
        cbar.set_ticks([0.1,0.5,1,2,4,10])
        cbar.set_ticklabels([0.1,0.5,1,2,4,10])
        cbar.set_label(label, size='x-small')
        cbar.ax.tick_params(labelsize='xx-small')


def plotjoint(tab, prefixx, prefixy, varnames, varcolor=None, norm=None, varcolorname=None,
              cmap = mpl.colors.ListedColormap(sns.color_palette("RdYlBu_r", 100)),
              plotmarginals=True, hist_kws={'log': False}, postfixx=None, postfixy=None,
              plotratiosjoint=True, bins=20):
    if varcolor is None or norm is None:
        varcolorname=r"ProFit HST $n_{Ser}$"
        varcolor="profit.hst.serb.n"
        norm = mplcol.LogNorm(vmin=0.1, vmax=10)
    if varcolorname is None:
        varcolorname = varcolor
    ratios = {var: tab[getname(var, prefixy, postfixy)]/tab[getname(var, prefixx, postfixx)] for 
              var in varnames}
    for i, x in enumerate(varnames): 
        xlog = np.log10(ratios[x])
        xlog[xlog > 1] = 1
        xlog[xlog < -1] = -1
        if plotratiosjoint:
            for y in varnames[(i+1):(len(varnames)+1)]:
                ylog = np.log10(ratios[y])
                ylog[ylog > 1] = 1
                ylog[ylog < -1] = -1
                fig = sns.JointGrid(x=xlog, y=ylog)
                jointplot = fig.plot_joint(plt.scatter, c=tab[varcolor], norm=norm, cmap=cmap, marker='.',
                               edgecolor='k', s=24, zorder=2).set_axis_labels(
                    'log10({} ratio) ({}/{})'.format(x, prefixy, prefixx),
                    'log10({}_ratio) ({}/{})'.format(y, prefixy, prefixx))
                x0, x1 = jointplot.ax_joint.get_xlim()
                y0, y1 = jointplot.ax_joint.get_ylim()
                lims = [max(x0, y0), min(x1, y1)]
                jointplot.ax_joint.plot(lims, lims, '-k', zorder=1)
                jointplotcolorbar(jointplot, label=varcolorname)
                if plotmarginals:
                    fig.plot_marginals(sns.distplot, kde=False, hist_kws=hist_kws)

        ylog = np.log10(tab[getname(x, prefixx, postfixx)])
        ylog[ylog > 10] = 10
        ylog[ylog < -10] = -10

        fig = sns.JointGrid(x=ylog, y=xlog)
        jointplot = fig.plot_joint(plt.scatter, c=tab[varcolor], norm=norm, cmap=cmap, marker='.',
                       edgecolor='k', s=24, zorder=2).set_axis_labels(
            'log10({}) ({})'.format(x, prefixx),
            'log10({} ratio) ({}/{})'.format(x, prefixy, prefixx))
        x0, x1 = jointplot.ax_joint.get_xlim()
        jointplot.ax_joint.plot([x0, x1], [0, 0], '-k', zorder=1)
        jointplotcolorbar(jointplot, label=varcolorname)
        if plotmarginals:
            jointgrid = fig.plot_marginals(sns.distplot, kde=False, hist_kws=hist_kws, bins=bins)
        #cax = fig.add_axes([.94, .25, .02, .6])

varnames = ["flux", "re", "n"]


# ### COSMOS-HST: GalSim catalog Sersic fit versus MultiProfit Sersic fit
# 
# This is a sanity check comparing MultiProFit's true Sersic profile fit (albeit convolved with a 3-Gaussian PSF rather than the PSF image for direct comparison) compared to the values in the GalSim catalog.
# 
# At some point I may try true Sersic fits with the PSF images, but these are fairly slow and I was mainly interested in comparing true Sersic vs MGA Sersic using Gaussian PSFs, because we'll almost certainly use only GMMs in LSST because of how much faster they are to render.
# 
# The results are reasonably consistent, though not without systematic differences. The methods used for the GalSim catalog are described in the GREAT3 Challenge Handbook (https://arxiv.org/abs/1308.4982). As far as I can tell, they used TinyTim-generated PSF images (which I think are provided in the catalog) rather than a Gaussian mixture model thereof. Additionally, they used different limits: Sersic index 0.1≤n≤6; 0.1 and 6; axis ratio 0.05≤q≤1, and (2)R_eff≤image size. Also, their Sersic profles are cut off at a radius that varies smoothly from 4R_eff at n= 1 to 8 R_eff at n=4. Given that, at fixed n one would expect smaller R_eff from MultiProFit. However, MultiProFit tends to prefer larger R_eff, n *and* flux. It is possible that the COSMOS fits are not integrating the Sersic profiles as accurately as GalSim does - the code paper (Lackner & Gunn 2012) does not offer much detail on how their sub-pixel Sersic profile integration is done (if it is at all). Accurate sub-pixel integration is most important for large Sersic n profiles, so this could explain why there are larger differences for large n profiles than for small n.

# In[10]:


plotjoint(tab, 'profit.hst.serb', 'cosmos.ser', varnames, plotratiosjoint=False)
plotjoint(tab, 'cosmos.ser', 'profit.hst.serb', varnames)


# ### COSMOS-HST: MultiProFit Sersic vs MultiProFit MGA Sersic (N=8)
# 
# How close is the N=8 MGA to a true Sersic fit? The answer is quite close, except at n>4 where the MGA intentionally does not match the inner profile (see the introduction for more information on the derivation of MGA Sersic weights) and for n < 0.5 where the MGA can't possibly match a true Sersic (Gaussians mixtures cannot reproduce the n -> 0 limit of a top hat function in 1D; in 2D one could have many spatially offset Gaussians but this is unreasonably expensive).
# 
# It seems like there are two different tracks of outliers: objects with n>4 where the MGA has some large but not unreasonable scatter, and objects with 2.5>n>4 where the MGA has systematically lower n (somewhat more mysterious).
# 
# There is a noticeable trend for large n profiles to have smaller sizes in the MGA Sersic. For the cases where both profiles hit the upper limit of n=6.3, it's important to note that the profiles are quite different - the MGA Sersic has a much shallower cusp (less central flux) and steeper wings (marginally less outer flux). At fixed R_eff, this is also true of the Sersic profile itself - larger n profiles are cuspier and have shallower wings. R_eff also generally correlates with n for reasons that are not really obvious. I suspect it is because for a larger Sersic n, a larger R_eff stretches out the steep inner cusp, and also lowers the surface brightness of the shallow outer profile so that it blends into the sky. At any rate, one can interpret the MGA Sersic as having a smaller effective Sersic n than the true Sersic profile, and therefore it is likely to have a smaller R_eff at fixed n.

# In[11]:


plotjoint(tab, 'profit.hst.mg8serbpx', 'profit.hst.serbpx', varnames, plotratiosjoint=False)
plotjoint(tab, 'profit.hst.serbpx', 'profit.hst.mg8serbpx', varnames)


# ### COSMOS-HST: MultiProFit Sersic vs MultiProFit MGA Sersic (N=4)
# 
# How close is the N=4 MGA to a true Sersic fit? Reasonably close, but there's clearly more scatter than with an N=8 fit, particularly at n>2 where more Gaussians are needed to match the shallow outer profile.

# In[12]:


plotjoint(tab, 'profit.hst.mg4serbpx', 'profit.hst.serbpx', varnames, plotratiosjoint=False)
plotjoint(tab, 'profit.hst.serbpx', 'profit.hst.mg4serbpx', varnames)
plotjoint(tab, 'profit.hst.mg8serbpx', 'profit.hst.mg4serbpx', varnames)


# ### COSMOS-HST: MultiProFit GMM (N=8) vs MultiProFit MGA Sersic
# 
# How different is a free GMM compared to a constrained Sersic MGA? Given the discussion above, one might expect real "early-type" (Sersic n > 2.5ish) galaxies to have shallower cusps than the Sersic profile predicts, in which case Sersic fits would tend to overestimate R_eff in order to stretch out the unrealistically steep cusp. Similarly, true Sersic fits predict shallow, extended outer wings, and therefore overestimate flux if these wings don't really exist. Therefore a free Gaussian mixture could be expected to prefer smaller effective radii and fluxes for large Sersic n. As might be expected, large-n galaxies tend to smaller fluxes and sizes with a full GMM, whereas the opposite is true for low-n galaxies.

# In[13]:


plotjoint(tab, 'profit.hst.mg8bpx', 'profit.hst.mg8serbpx', ["flux", "re", "chisqred.0"], plotratiosjoint=False)
plotjoint(tab, 'profit.hst.mg8serbpx', 'profit.hst.mg8bpx', ["flux", "re"])


# ### COSMOS-HST: MultiProFit GMM (N=4) vs MultiProFit MGA Sersic and N=8 GMM
# 
# Are the N=4 GMM parameters consistent with the N=8? Pretty much, except for a few extreme outliers. The differences in reduced chi-squared are also very modest. Thus, it's not surprising that the comparison between N=4 GMM vs N=8 MGA Sersic looks about the same as N=8 GMM vs N=8 MGA Sersic, too.

# In[14]:


plotjoint(tab, 'profit.hst.mg4bpx', 'profit.hst.mg8serbpx', ["flux", "re", "chisqred.0"], plotratiosjoint=False)
plotjoint(tab, 'profit.hst.mg8serbpx', 'profit.hst.mg4bpx', ["flux", "re"])

plotjoint(tab, 'profit.hst.mg8bpx', 'profit.hst.mg4bpx', ["flux", "re", "chisqred.0"], plotratiosjoint=False)
plotjoint(tab, 'profit.hst.mg4bpx', 'profit.hst.mg8bpx', ["flux", "re"])


# ### COSMOS-HST: MultiProFit GMM vs MultiProFit MGA Sersic (N=4 x 2 Components)
# 
# How much better is a 2-component GMM (shared shape, so the radial profile is a Gaussian mixture) than a 2-component MGA Sersic? Also compare to N=8 Sersic MGA, which has fewer parameters but covers a different space from the N=4x2 GMM, since each component has 8 Gaussians.

# In[15]:


plotjoint(tab, 'profit.hst.mg4x2px', 'profit.hst.mg4serserbpx', ["flux", "chisqred.1"], plotratiosjoint=False)
plotjoint(tab, 'profit.hst.mg8serserbpx', 'profit.hst.mg4x2px', ["flux", "chisqred.1"])


# ### MultiProFit MGA Sersic (N=8): COSMOS-HST vs COSMOS-HSC
# 
# How well does a fit to synthetic, Subaru-HSC UltraDeep data recover the parameters derived from the HST fits? It should do well modulo noise bias, because the image is generated using the best-fit HST model parameters, but convolved with the actual HSC PSF for that object and with added noise from the actual HSC variance map.
# 
# In general, fluxes are reasonably robust, with the scatter increasing for fainter objects, as one might expect. Sizes are recovered well down to R_eff ~ 10 pixels but are biased low below that level. This seems rather high; I would have hoped for good size recovery down to R_eff ~3 pixels. Sersic index recovery is generally poor, with a noticeable bias to lower Sersic n. It's not yet clear why these trends exist. It appears as if there is a track of galaxies with well-recovered fluxes and seemingly random errors on Sersic n and R_eff, and another where the parameter ratios are all correlated.

# In[16]:


plotjoint(tab, 'profit.hst2hsc_mg8serbpx.mg8serbpx', 'profit.hst.mg8serbpx', varnames, plotratiosjoint=False)
plotjoint(tab, 'profit.hst.mg8serbpx', 'profit.hst2hsc_mg8serbpx.mg8serbpx', varnames)


# ### Model comparison plots (COSMOS-HST)
# 
# These plots compare the goodness of fit (reduced chi-squared in this case) of different model combinations. This mainly helps to determine which models are worth fitting and - judging by the absolute goodness of fit and the best-fit Sersic index for the single MG Sersic - what kind of galaxies each model is good for.
# 
# The first plot is probably the most interesting, as it shows that a single Sersic fit is better for most galaxies with a best-fit Sersic index n~0.5 or n~6 than a more complicated DevExp model, whereas the devExp model appears to do better for more typical galaxies with intermediate values of 1 < n < 4. This isn't completely surprising - a devExp model can't reproduce a single Gaussian, nor can a single Sersic (usually with 2 < n < 3) reproduce a pure devExp, but it's necessary to check which model is preferred by real galaxies.
# 
# Most of the galaxies for which a low-n single Sersic fit is preferred (light blue dots below 0 on the y-axis) tend to have relatively low reduced chi-squared values; these galaxies are probably not very well resolved and the fact that the devExp fit is usually not much worse suggests that they're most likely not truly Gaussian-like profiles. On the other hand, there are a significant number of galaxies preferring high-n Sersic fits to devExp; it's possible that some of these really are extended elliptical galaxies rather than bulge+disk systems.
# 
# Some of the later plots compare equivalent models where the only differences are the initial parameters; an ideal optimizer should reach identical solutions for both, albeit not necessarily in the same amount of time/iterations. Future analysis will need to devise metrics for whether a given model is worth fitting given the amount of time it takes to compute, but those measures depend on the details of the optimizer, priors and any future code performance optimizations.
# 
# Conclusions: The two-component N=4 GMM is overwhelmingly the best model, which is not surprising - it has enough freedom to reproduce any of the other models except the true Sersic. Unfortunately, it is very expensive to fit and will need some optimization to become practical for use on a large sample. Leaving it aside, there are more galaxies better-fit by devExp than by single Sersic models, but more galaxies prefer a single-component GMM to a devExp model. Based on these results, I think that none of these models should be excluded from contention. All of these models are worth testing by fitting HSC data in multi-band mode.

# In[17]:


# Which are the best models?

def getpostfix(model):
    return "0" if model in models['single']['profit'] else "1"

modelslist = [
    ('Single-component', models["single"]["profit"]),
    ('One- and two-component, no GMM', models["single"]["profit"] + 
     [model for model in models["double"]["profit"] if not model.endswith('serserbpx')]),
    ('One- and two-component, no GMMx2', models["single"]["profit"] + 
     [model for model in models["double"]["profit"] if not model.endswith('serserbpx')] + 
     [model for model in models['mg']['profit'] if not model.endswith('mg4x2px')]),
    ('All', models["single"]["profit"] + models["double"]["profit"] + models['mg']['profit']),
]
chisqredcols = {}
for modelsname, modelsprofit in modelslist:
    print('Best-fit models, ' + modelsname + ':')
    chisqredcolsprint = {}
    for model in modelsprofit:
        if model not in chisqredcols:
            chisqredcols[model] = ".".join(["profit", "hst", model, "chisqred", getpostfix(model)])
        chisqredcolsprint[model] = chisqredcols[model]
    print(chisqredcolsprint)
    modelbest = tab[list(chisqredcolsprint.values())].idxmin(axis=1)
    modelbestcounts = modelbest.value_counts()
    print(modelbestcounts)

# Plot direct model comparisons
for obs in ['hst', 'hst2hsc_mg8serbpx']:
    for colx, coly in [("mg8serbpx", "mg8devexppx"), ("mg8serbedpx", "mg8cmodelpx"), 
                       ("mg8devexppx", "mg8cmodelpx"), ("mg8serbedpx", "serb"),
                       ("mg8serbedpx", "mg8serbpx"), ("mg8serserbpx", "mg8bpx"),
                       ("mg4x2px", "mg8serserbpx")]:
        prefixx = ".".join(["profit", obs, colx])
        prefixy = ".".join(["profit", obs, coly])
        plotjoint(tab, prefixx, prefixy, ['chisqred'], varcolor="profit.hst.mg8serbpx.n",
                  cmap = mpl.colors.ListedColormap(sns.color_palette("RdYlBu_r", 100)),
                  hist_kws={'log': True}, postfixx=getpostfix(colx), postfixy=getpostfix(coly))


# ### MG Sersic fixed-n vs free-n fits
# 
# These plots compare fixed- versus free-n single Sersic fits. This is to help determine an optimal workflow for fitting increasingly complicated models. For example, the Gaussian model is the fastest to evaluate, but it's also the least likely to be the best fit out of the four fixed-n models (n=0.5, 1, 2, 4).
# 
# Another plot compares the goodness of fit for the free-n versus the best-fit fixed-n model, showing that in most cases a fixed-n model is not far off - that is, there are not many galaxies where there's a very significant benefit to fitting n=1.5 or n=3 vs n=2. The exception is the galaxies that prefer n=6 to n=4.
# 
# The last plot compares the free Sersic to an exponential, showing that an exponential is 'good enough' for most galaxies except those with best-fit n>4. This suggests that the exponential model would be a reasonable choice if we had to pick just one, which is in line with expectations that most of the galaxies in any given deep field are disky.

# In[18]:


# Now compare only single-component models: Sersic vs best fixed n
modelsfixedn = ["gausspx", "mg8exppx", 'mg8n2px', "mg8dev2px"] 
chisqredcolsfixedn = {
    model: ".".join(["profit", "hst", model, "chisqred", "0"]) 
    for model in modelsfixedn
}
modelbest = tab[list(chisqredcolsfixedn.values())].idxmin(axis=1)
print('Counts of best fixed-n Sersic model:')
print(modelbest.value_counts())
# I seriously cannot figure out how to slice with modelbest
# Surely there's a better way to do this?
modelchisqmin = tab[list(chisqredcolsfixedn.values())].min(axis=1)
for colxname in ["mg8serbpx", 'serb']: 
    labelbest = 'log10(chisqred) ({}/{})'.format(colxname, "best([gauss,exp,n2,dev]px)")
    ratiobest = tab[chisqredcols[colxname]]/modelchisqmin
    # Plots:
    # How much better is Sersic than the best [gauss/exp/dev] vs how good is the 
    # fit and vs Sersic index
    # As above but vs exp only
    colnser = colxname.join(["profit.hst.", ".n"])
    cols = [
        (tab[chisqredcols[colxname]], ratiobest,
         'log10(chisqred) ({})'.format(colxname), labelbest), 
        (tab[colnser], ratiobest, 'log10(n_ser)', labelbest),
        (tab[colnser], tab[chisqredcols[colxname]]/tab[chisqredcols["mg8exppx"]],
         'log10(n_ser)', 'log10(chisqred) ({}/exp)'.format(colxname)),
    ]
    for x, y, labelx, labely in cols:
        sns.jointplot(
            x=np.log10(x),
            y=np.log10(y),
            color="k", joint_kws={'marker': '.', 's': 4},
            marginal_kws={'hist_kws': {'log': False}},
        ).set_axis_labels(labelx, labely)


# ### Outliers and failed fits
# 
# Todo. Some of the plots have some fairly extreme outliers; these should be visually inspected for QA purposes.
# 
# Some 10% of the galaxies have at least one model that failed to fit; these should also be verified.
# 
# A small fraction of these failures are failing to find an HSC UltraDeep source within the specified tolerance distance from the nominal COSMOS astrometric coordinates. A few more are on galaxies with very large cutouts, where the true Sersic model has failed to fit because GalSim requires too large of an FFT to integrate the profile accurately. In turn, some of those large cutouts are >90% noise, indicating that there are issues with the COSMOS catalog itself.

# ### Compare GMM radial profiles to Sersic MGA
# 
# Todo. It would make sense to plot the radial profiles of the N=8 GMM and compare it to the MGA Sersic for the same galaxy. 

# ### Compare model running times
# 
# Todo. It would be useful to plot, say, runtime ratio vs chisqred ratio, and identify when and why some model fits take absurdly long times to complete.
# 
# The main caveat with this is that both quantities (runtime and goodness of fit) are optimizer-dependent, so it may take more thorough testing of optimization pathways before we can determine whether it's worth fitting models of increasing complexity.

# ### N=8 vs N=4 Sersic MGA/GMM
# 
# Todo (DM-16875). N=4x2 GMMs have been run. They are the best models in almost all cases, as they should be because they have nearly 40 free parameters and can reproduce any of the existing simpler models except true Sersic, but they are very expensive to fit. We may end up using only linear models with free weights based on e.g. the devExp fits.

# ### Multi-band COSMOS-HSC
# Todo (DM-17466). It has been tested on a handful of galaxies using gri data, but not yet all five bands or on a substantial sample. It's fairly computationally expensive, especially for GMMs with free weights since they have one flux per band.
