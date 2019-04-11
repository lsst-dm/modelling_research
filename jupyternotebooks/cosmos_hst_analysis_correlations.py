#!/usr/bin/env python
# coding: utf-8

# # MultiProFit-COSMOS: 
# 
# This notebook plots results from fitting galaxies from the COSMOS survey (http://cosmos.astro.caltech.edu/) with MultiProFit (https://github.com/lsst-dm/multiprofit). This investigation is to help determine what kind of galaxy models LSST Data Management should fit for annual Data Releases, where we would like to fit the best possible model(s) but will be constrained in computing time by the overwhelming size of the dataset (billions of objects).

# ## Introduction and motivation
# 
# See [this notebook](https://github.com/lsst-dm/modelling_research/blob/master/jupyternotebooks/cosmos_hst_analysis.ipynb) for general background and motivation.

# ## Browsing this notebook
# 
# I recommend using jupyter's nbviewer page to browse through this notebook. For example, you can use it to open the [N=4 GMM](https://nbviewer.jupyter.org/github/lsst-dm/modelling_research/blob/master/jupyternotebooks/cosmos_hst_analysis.ipynb#COSMOS-HST:-MultiProFit-Sersic-vs-MultiProFit-MGA-Sersic-(N=4) and compare to the [N=8 GMM](https://nbviewer.jupyter.org/github/lsst-dm/modelling_research/blob/master/jupyternotebooks/cosmos_hst_analysis.ipynb#COSMOS-HST:-MultiProFit-Sersic-vs-MultiProFit-MGA-Sersic-(N=8) side-by-side.

# ### Import required packages
# 
# Import required packages and set matplotlib/seaborn defaults for slightly nicer plots.

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns


# In[2]:


# Setup for plotting
get_ipython().run_line_magic('matplotlib', 'inline')

#plt.style.use('seaborn-notebook')
sns.set_style('darkgrid')
mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['image.origin'] = 'lower'
sns.set(rc={'axes.facecolor': '0.85', 'figure.facecolor': 'w'})


# ### Read the table with the results

# In[3]:


from modelling_research.plot_multiprofit_cosmos import readtable
filename = '../data/multiprofit-cosmos-fits.csv'
tab = readtable(filename)


# ### Make joint parameter plots
# 
# Something something.

# In[4]:


from modelling_research.plot_multiprofit_cosmos import plotjointsersic
varnames = ["flux", "re.1"]
plotjointsersic(tab, 'profit.hst.gausspx', 'profit.hst.mg8exppx', varnames, plotratiosjoint=True, postfixx='1', postfixy='1')
plotjointsersic(tab, 'profit.hst.mg8exppx', 'profit.hst.mg8dev2px', varnames, plotratiosjoint=True, postfixx='1', postfixy='1')
plotjointsersic(tab, 'profit.hst.gausspx', 'profit.hst.mg8dev2px', varnames, plotratiosjoint=True, postfixx='1', postfixy='1')


# In[5]:


varnamesfit = ["flux.1", "time"]
plotjointsersic(tab, 'profit.hst2hsc.gausspx', 'profit.hst2hsc.mg8exppx', varnamesfit, plotratiosjoint=False)
plotjointsersic(tab, 'profit.hst2hsc.mg8exppx', 'profit.hst2hsc.mg8dev2px', varnamesfit, plotratiosjoint=False)
plotjointsersic(tab, 'profit.hst2hsc.mg8serbpx', 'profit.hst2hsc.mg8serbedpx', varnamesfit, plotratiosjoint=False)


# ### Multi-band COSMOS-HSC
# Todo (DM-17466). It has been tested on a handful of galaxies using gri data, but not yet all five bands or on a substantial sample. It's fairly computationally expensive, especially for GMMs with free weights since they have one flux per band.
