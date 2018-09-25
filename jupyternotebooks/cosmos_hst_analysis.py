
# coding: utf-8

# In[1]:


import astropy as ap
import galsim as gs
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd

import pickle

import seaborn as sns

# Setup for plotting
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plt.style.use('seaborn-notebook')
sns.set_style('darkgrid')
mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['image.origin'] = 'lower'
sns.set(rc={'axes.facecolor': '0.85', 'figure.facecolor': 'w'})


# In[2]:


path = os.path.expanduser('~/raid/hsc/cosmos/COSMOS_25.2_training_sample/')
file = "real_galaxy_catalog_25.2.fits"
ccat = gs.COSMOSCatalog(file, dir=path)
rgcfits = ap.io.fits.open(os.path.join(path, file))[1].data


# In[3]:


data = []
 
files = glob.glob(os.path.expanduser(
    "~/raid/lsst/cosmos/cosmos_25.2_fits_*_*_pickle.dat"))
files.sort()
for file in files:
    with open(file, 'rb') as f:
        data.append(pickle.load(f))


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

sources = ["hst"] + ["_".join(["hst2hsc", postfix]) for postfix in [
#    "",
    "mgserbgepx",
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
idxparamsprofit = {
    "gausspx":     idxprofit,
    "mgexppx":     idxprofit,
#    "mgexpgpx": idxprofit,
    "mgn2px":      idxprofit,
    "mgdevepx":    idxprofit,
    "mgserbgedpx": idxprofit,
    "mgserbgepx":  idxprofit,
    "serbpx":  idxprofit,
    "serb":  idxprofit,
    "mgcmodelpx":  idxprofittwo,
    "mgdevexppx":  idxprofittwo,
    "mgdevexpcpx": idxprofittwo,
}
paramsser = ["flux", "re", "n", "q", "phi", "x0", "y0"]
models = {
    "single": {
        "cosmos": ["ser"],
        "profit": ["gausspx", "mgexppx", "mgn2px", "mgdevepx", 
                   "mgserbgedpx", "mgserbgepx", "serbpx", "serb"],
    },
    "double": {
        "cosmos": ["devexp"],
        "profit": ["mgcmodelpx", "mgdevexppx", "mgdevexpcpx"],
    },
}
modellers = {
    "cosmos": [None],
    "profit": sources,
}
colnames = {
    modeller if src is None else ".".join([modeller, src]): 
        [".".join([model, param]) for model in models["single"][modeller] for param in paramsser] +
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
             for idx in range(1 + (model in models['double']['profit']))] +
            [".".join([prefix, x]) for prefix, colnames in colnames.items() for x in colnames])

print(colnames)


# In[5]:


# Some hideous code to get all of the right values in the right order, for which I apologize

# The COSMOS catalog has fewer objects than the RealGalaxyCatalog for ??? reasons
# Get the mapping between RGC indices (which is what I used) and CosmosCat ones (for previous fits)
indexmap = {ccat.getOrigIndex(i): i for i in range(ccat.getNObjects())}

rows = []

# Shouldn't have hardcoded this but here we are
scaleratio = 0.168/0.03
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
                hasfits = hasfits and src in datatab[idx]
                if hasfits:
                    profit = datatab[idx][src]
                    hasfits = hasfits and 'fits' in profit
                    if hasfits:               
                        profit = profit['fits']
                        for model in idxparamsprofit:
                            hasfits = hasfits and model in profit['galsim']
                            if hasfits:
                                profitmodel = profit['galsim'][model]
                                hasfits = hasfits and 'fits' in profitmodel
                                if hasfits:
                                    profitmodel = profitmodel['fits']
                                    row += [profitmodelfit[param]
                                            for param in params["profit"]
                                            for profitmodelfit in profitmodel]
            if hasfits:
                row += [rec["flux"][0]] + list(rec["sersicfit"][idxparamscosmos[0]])
                for offset in range(2):
                    row += [rec["flux"][1+offset]] + list(rec["bulgefit"][idxparamscosmos[offset]])
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
                        row += list(values[idxs])
        if hasfits:
            rows.append(row)
            appended += 1
    print("Read {}/{} rows".format(appended, len(datatab)))

    # TODO: Convert mu_re into flux for use_bulgefit=0
    #row += profit["paramsbest"]


# In[6]:


# Write to a plain old CSV, then read it back in to double-check

import csv

with open(os.path.join(path, "galfits.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([colnames])
    writer.writerows(rows)
    


# In[7]:


tab = pd.read_csv(os.path.join(path, "galfits.csv"))


# In[8]:


# How well are parameters recovered?
# First check consistency with previous fits (Sersic only for now)
# Then HSC vs HST


varnames = ["flux", "re"]
prefixes = [
    ("profit.hst.serb", "cosmos.ser", varnames),
    ("profit.hst.mgserbgedpx", "cosmos.ser", varnames),
    ("profit.hst.mgserbgedpx", "profit.hst2hsc_mgserbgepx.mgserbgedpx", 
     varnames),
#    ("profit.hst.ser", "profit.hst2hsc_devexp.ser", vars),
#    ("profit.hst.ser", "profit.hst2hsc.ser", vars)
]

cmap = mpl.colors.ListedColormap(sns.color_palette("RdYlBu_r", 100))
colors = (np.log10(tab["profit.hst.mgserbgedpx.n"])+1)/np.log10(60)
colors = cmap(np.rint(100*colors)/100)


def getname(var, prefix):
    return ".".join([prefix, var])


for prefixx, prefixy, varnames in prefixes:
    ratios = {var: tab[getname(var, prefixx)]/tab[getname(var, prefixy)] for 
              var in varnames}
    for i, x in enumerate(varnames): 
        xlog = np.log10(ratios[x])
        xlog[xlog > 1] = 1
        xlog[xlog < -1] = -1
        for y in varnames[(i+1):(len(varnames)+1)]:
            namey = getname(y, prefixy)
            ylog = np.log10(ratios[y])
            ylog[ylog > 1] = 1
            ylog[ylog < -1] = -1
            fig = sns.JointGrid(x=xlog, y=ylog)
            fig.plot_joint(plt.scatter, c=colors, marker='.',
                           edgecolor='k', s=24).set_axis_labels(
                'log10({}_ratio) ({}/{})'.format(x, prefixx, prefixy),
                'log10({}_ratio) ({}/{})'.format(y, prefixx, prefixy))
            fig.plot_marginals(sns.distplot, kde=False, hist_kws={'log': True})

        ylog = np.log10(tab[getname(x, prefixx)])
        fig = sns.JointGrid(x=ylog, y=xlog)
        fig.plot_joint(plt.scatter, c=colors, marker='.',
                       edgecolor='k', s=24).set_axis_labels(
            'log10({}) ({})'.format(x, prefixx),
            'log10({}_ratio) ({}/{})'.format(x, prefixx, prefixy))
        fig.plot_marginals(sns.distplot, kde=False, hist_kws={'log': True})
        #cax = fig.add_axes([.94, .25, .02, .6])


# In[ ]:


# Which are the best models?
modelsprofit = models["single"]["profit"] + models["double"]["profit"] 
chisqredcols = {
    model: ".".join(["profit", "hst", model, "chisqred",
                     "1" if model in models['double']['profit'] else "0"]) 
    for model in modelsprofit
}
print(chisqredcols)
modelbest = tab[list(chisqredcols.values())].idxmin(axis=1)
modelbestcounts = modelbest.value_counts()
print(modelbestcounts)
for colx, coly in [("mgserbgedpx", "mgdevexppx"), ("mgserbgedpx", "mgcmodelpx"), 
                   ("mgdevexppx", "mgcmodelpx")]:
    fig = sns.JointGrid(x=np.log10(tab[chisqredcols[colx]]),
                        y=np.log10(tab[chisqredcols[colx]]/tab[chisqredcols[coly]]))
    fig.plot_joint(plt.scatter, c=colors, marker='.',
                   edgecolor='k', s=24).set_axis_labels(
        'log10(chisqred) ({})'.format(colx),
        'log10(chisqred) ({}/{})'.format(colx, coly))


# In[17]:


# Now compare only single-component models: Sersic vs best fixed n
modelsfixedn = ["gausspx", "mgexppx", 'mgn2px', "mgdevepx"] 
chisqredcolsfixedn = {
    model: ".".join(["profit", "hst", model, "chisqred",
                     "0" if model == "cmodel" else "0"]) 
    for model in modelsfixedn
}
modelbest = tab[list(chisqredcolsfixedn.values())].idxmin(axis=1)
print(modelbest.value_counts())
# I seriously cannot figure out how to slice with modelbest
# Surely there's a better way to do this?
modelchisqmin = tab[list(chisqredcolsfixedn.values())].min(axis=1)
colxname = "mgserbgedpx"
labelbest = 'log10(chisqred) ({}/{})'.format(colxname, "best{gauss,exp,n2,dev}")
ratiobest = tab[chisqredcols[colxname]]/modelchisqmin
# Plots:
# How much better is Sersic than the best [gauss/exp/dev] vs how good is the 
# fit and vs Sersic index
# As above but vs exp only
cols = [
    (tab[chisqredcols[colxname]], ratiobest,
     'log10(chisqred) ({})'.format(colxname), labelbest), 
    (tab["profit.hst.mgserbgedpx.n"], ratiobest, 'log10(n_ser)', labelbest),
    (tab["profit.hst.mgserbgedpx.n"],
     tab[chisqredcols[colxname]]/tab[chisqredcols["mgexppx"]],
     'log10(n_ser)', 'log10(chisqred) ({}/exp)'.format(colxname)),
]
for x, y, labelx, labely in cols:
    sns.jointplot(
        x=np.log10(x),
        y=np.log10(y),
        color="k", joint_kws={'marker': '.', 's': 4},
        marginal_kws={'hist_kws': {'log': True}},
    ).set_axis_labels(labelx, labely)


# In[ ]:




