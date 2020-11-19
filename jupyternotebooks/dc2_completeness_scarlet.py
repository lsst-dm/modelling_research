#!/usr/bin/env python
# coding: utf-8

# # DC2 v2 Simulation Measurement Completeness and Purity
# 
# This notebook plots completeness and purity for measurements of sources on the Vera Rubin Observatory/LSST Dark Energy Science Collaboration (DESC, https://lsstdesc.org/) DC2 simulations (http://lsstdesc.org/DC2-production/).

# In[1]:


# Import requirements: github.com/lsst-dm/modelling_research.
# You need that package on your python path for the plotting/dc2 imports to work
from lsst.daf.persistence import Butler
import matplotlib as mpl
import matplotlib.pyplot as plt
import modelling_research.dc2 as dc2
import modelling_research.meas_model as mrMeas
from modelling_research.plotting import plotjoint_running_percentiles
from modelling_research.plot_matches import plot_compure, plot_matches
import numpy as np
import seaborn as sns
from timeit import default_timer as timer


# In[2]:


# Setup for plotting
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['image.origin'] = 'lower'
sns.set(rc={'axes.facecolor': '0.85', 'figure.facecolor': 'w'})


# In[3]:


# Catalog matching settings
# Reference band to match on. Should be 'r' for DC2 because reasons (truth refcats are r mag limited)
band_ref = 'r'


# In[4]:


# Load the truth catalogs and count all of the rows
butler_ref = dc2.get_refcat(make=False)


# In[5]:


# Load the DC2 repo butler for 3828 only
butlers_dc2 = {
    '2.2i': Butler('/project/dtaranu/dc2/scarlet/2020-09-16/griz/'),
}


# ## Crossmatch against reference catalog
# 
# All of the plots in this notebook compare stack measurements of sources measured by the LSST Science Pipelines cross-matched against a reference catalog that is part of the Science Pipelines repository. This reference catalog contains all sources brighter than 23 mags in r, and was generated through some kind of Task that must have read the original DC2 truth tables (more on that below).

# In[6]:


# Match with the refcat using afw's DirectMatchTask
truth_path = dc2.get_truth_path()
tracts = {3828: (f'{truth_path}/scarlet/2020-09-28_mpf/', '2.2i'),}
filters_single = ('g', 'r', 'i', 'z')
filters_multi = ('griz',)
cats = dc2.match_refcat_dc2(
    butler_ref, match_afw=False, tracts=tracts, butlers_dc2=butlers_dc2, filters_single=filters_single,
    filters_multi=filters_multi,
)


# In[35]:


# Model plot setup
models = {
    desc: mrMeas.Model(desc, field, n_comps)
    for desc, field, n_comps in [
        ('PSF', 'base_PsfFlux', 1),
        ('Scarlet', 'scarlet', 0),
        ('Stack CModel', 'modelfit_CModel', 2),
    ]
}
args = dict(scatterleft=True, scatterright=True, limx=(15, 25.),)
lim_y = {
    "resolved": (-0.6, 0.4),
    "unresolved": (-0.1, 0.05),
}
models_purity = list(models.keys())


# ## Galaxy and Star Completeness and Purity
# The remainder of the notebook plots completeness and purity. The matching works as follows:
# 1. The (true and deep) reference catalog is matched to find the nearest measured source
# 2. If multiple matches are found, only the brightest one is retained
# 3. If rematch is True, the faint matches removed in step 2 are rematched if they have a
# close enough true match that wasn't already matched in step 1.
# 
# TODO: I think step 3 is first-come first-serve rather than brightness based as in step 2.
# TODO: The matching scheme should probably be replaced with a likelihood based on centroids
# and magnitudes, given measurement errors on all of the above.

# ### Galaxies, 1 pixel match radius
# This shows galaxies down to 28 mag with a fairly conservative 1 pixel (0.168") match radius.
# 
# - Completeness drops slowly to 90% at 23rd mag, 80% at 25-24 (gri), and very sharply after that.
# - Fainter than 24th mag, there are increasing numbers of matches to objects that aren't classified as galaxies, although some of these could be nan extendedness rather than definitely not extended.
# - Purity looks reasonable, although there are some matches of bright galaxies to stars. This may be CModel going awry on a bad deblend.

# In[30]:


# Galaxies
plot_matches(
    cats, True, models, filters_single, band_ref=band_ref, band_ref_multi=band_ref,
    mag_max=28, mag_max_compure=28, limy=lim_y['resolved'],
    match_dist_asec=0.168, rematch=True, models_purity=models_purity, mag_bin_complete=0.125,
    plot_diffs=False, compare_mags_psf_lim=(-2.5, 0), **args
)


# ### Galaxies, 3 pixel match radius
# These plots are as above but with a larger match radius to compensate for galaxy astrometry being harder than for stars.
# - Somewhat surprisingly, the completeness gets worse with a larger match radius and without rematching. This is likely because there are more multiple matches getting thrown out than new matches 1-3 pixels away. This does make sense if you look at the purity plots in the previous section - since the impurity is only a few percent to r~24, that sets the ceiling on new matches.
# - Unfortunately, most of the extra matches with rematch one are to objects that aren't obviously galaxies (nan or no extendedness), as shown by the 2-3% wrong type matches from 20<r<24.
# - Consequently, completeness for matches of the right type is not significantly improved; the 80% limits shift less than half a mag fainter in r band.
# - By contrast, purity is significantly improved and remains above 97% for r<25 and 95% for r<26.
# 
# I'm not sure what these last two points combined say about detection efficiency. It does suggest that either our crossmatching needs to improve, or the galaxy astrometry is poor (in which case the photometry can't be great for faint galaxies either, but we already knew that).
# 
# It's also uncertain how this picture would change with a deeper star truth table, although since this is an extragalctic field, galaxies already far outnumber stars at r=23 so probably not much.

# In[36]:


# Galaxies w/~3 pixel match radius
plot_matches(
    cats, True, models, filters_single, band_multi=filters_multi[0], band_ref=band_ref, band_ref_multi=band_ref,
    mag_max=28, mag_max_compure=28, limy=lim_y['resolved'],
    match_dist_asec=0.5, rematch=True, models_purity=models_purity, mag_bin_complete=0.125,
    plot_diffs=False, compare_mags_psf_lim=(-2.5, 0), **args
)


# ### Stars, 1 pixel match radius
# 
# Unfortunately, since the truth tables only go down to r=23 for stars, there's not much to say here other than that completeness and purity seem reasonable.
# - The increase in the fraction of matches to wrong types from r=22 to r=23 in the completeness plot is somewhat troubling. Is this indicative of bad deblending, source modelling, or crossmatching? (Yes...)
# - Purity looks fine below r<23. Fainter than that, I suppose it's actually a measure of how bad blending with galaxies is for stars.

# In[33]:


# Stars
plot_matches(
    cats, False, models, filters_single, band_ref=band_ref, band_ref_multi=band_ref,
    mag_max=25, mag_max_compure=25, limy=lim_y['unresolved'],
    match_dist_asec=0.168, rematch=True, models_purity=models_purity, mag_bin_complete=0.25,
    plot_diffs=False, **args
)


# In[ ]:


# A test plot with a nice, smoothly decreasing completeness function
mags = np.clip(-2.5*np.log10(cats[3828]['truth']['lsst_r_flux']) + 31.4, 15, None)
chance = 1 - 0.5*(1 + np.arcsinh(mags-22.5)/np.arcsinh(7.5))
matched = np.random.uniform(size=len(mags)) < chance
plot_compure(mags, matched)


# In[ ]:




