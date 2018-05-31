
# coding: utf-8

# In[1]:


import astropy.table as apt
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-notebook')
mpl.rcParams['figure.dpi'] = 400


# In[2]:


df = pd.read_table("/home/taranu/raid/lsst/fit_bench_100x100_psffwhm-3.500e+00_oversample-3.dat", delim_whitespace=True)

print(list(df))

cols = ["nser", "ang", "axrat", "re"]
valsuniq = {}
for col in cols:
    valsuniq[col] = df[col].drop_duplicates().values

colscombine = ()
for col in ["ang", "axrat", "re"]:
    colscombine += (valsuniq[col],)

lennsers = len(valsuniq["nser"])

ncols = np.min([4, np.int(np.floor(np.sqrt(lennsers)))])
nrows = np.int(np.ceil(lennsers/ncols))

print(valsuniq)
print(ncols, nrows)

valstoplot = {}
for col in ["ang", "axrat"]:
    valinds = [0, np.int(np.ceil(len(valsuniq[col])/2))-1, len(valsuniq[col])-1]
    valstoplot[col] = valsuniq[col][valinds]


# In[3]:


plottypes = {
    "time": {
        "label": "log10(t/sec)"
        , "limits": [-3.3, -0.3]
        , "columns": ["CPU_t", "CPU_GS_ft_t"]
        , "linestyles": ["-", "--"]
        , "sublabel": {"xy": [-0.25, -0.1], "va":"bottom", "ha":"left"}
    }
    , "err": {
        "label": "log10(err)"
        , "limits": [-6, 0]
        , "columns": ["CPU_e", "CPU_GS_ft_e"]
        , "linestyles": ["-", "--"]
        , "sublabel": {"xy": [0.95, -0.2], "va":"top", "ha":"right"}
    }
}

for plottype in plottypes.keys():
    plotinfo = plottypes[plottype]
    plotinfosub = plotinfo["sublabel"]
    
    fig, axesfig = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    axesplot = {"row": None, "col": []}
    for i, nser in enumerate(valsuniq["nser"]):
        col = i % ncols
        row = np.int(np.floor(i / ncols))
        subplot = axesfig[row][col]
        subplot.tick_params(axis='both', which='major', labelsize=3)
        subplot.tick_params(axis='both', which='minor', labelsize=2)
        subplot.text(plotinfosub["xy"][0], plotinfosub["xy"][1],
                     "n={:.2f}".format(nser), fontsize=4,
                     verticalalignment=plotinfosub["va"],
                     horizontalalignment=plotinfosub["ha"])
        if i == 0:
            subplot.set_ylim(plotinfo["limits"][0], plotinfo["limits"][1])
            subplot.set_xlim(-0.3, 1)
        subplot.tick_params(length=3, width=0.5)
        if col == 0:
            subplot.set_ylabel(plotinfo["label"], fontsize=5)
        if row == (nrows-1):
            subplot.set_xlabel("log10(Re/pix)", fontsize=5)
        for ang in valstoplot["ang"]:
            for axrat in valstoplot["axrat"]:
                cond = (df["nser"] == nser) & (df["ang"] == ang) & (df["axrat"] == axrat)
                for column, linestyle in zip(plotinfo["columns"], plotinfo["linestyles"]):
                    subplot.plot(np.log10(df.loc[cond, "re"]), np.log10(df.loc[cond, column]),
                                 linewidth=0.3, linestyle=linestyle)
    
    subplot = axesfig[nrows-1][ncols-1]
    subplot.plot([], [], linewidth=0.3)
    subplot.plot([], [], linewidth=0.3, linestyle="--")
    subplot.tick_params(axis='both', which='major', labelsize=3)
    subplot.tick_params(axis='both', which='minor', labelsize=2)
    subplot.set_xlabel("log10(Re/pix)", fontsize=5)
    plt.legend(["ProFit", "GS-FFT"], fontsize=5)
    plt.tight_layout(0, 0, 0)

plt.show()
plt.clf()
plt.clf()

