#!/usr/bin/env python
# coding: utf-8

# # Fitting DC2 v2 Simulation Data with MultiProFit
# 
# This notebook plots results from fitting sources from the Vera Rubin Observatory/LSST Dark Energy Science Collaboration (DESC, https://lsstdesc.org/) DC2 simulations (http://lsstdesc.org/DC2-production/) using MultiProFit (https://github.com/lsst-dm/multiprofit, 'MPF'). Specifically, it reads the results of using a Task to fit exposures given an existing catalog with fits from meas_modelfit (https://github.com/lsst/meas_modelfit, 'MMF'). MMF implements a variant of the SDSS CModel algorithm (https://www.sdss.org/dr12/algorithms/magnitudes/#cmodel). In additional to CModel, MultiProFit allows for multi-band fitting, as well as fitting of Sersic profiles, multi-Gaussian approximations thereof ('MG' Sersic), and non-parametric radial profiles (Gaussian mixtures model with shared ellipse parameters, effectively having a Gaussian mixture radial profile). Thus, the main results are comparisons of the two codes doing basically the same thing (single-band exponential, de Vaucouleurs, and CModel linear combination fits), followed by plots highlighting Sersic vs CModel fits, more complicated double Sersic/free-amplitude models, and MultiProFit's multi-band fits.

# In[1]:


# Import requirements
import matplotlib as mpl
import matplotlib.pyplot as plt
import modelling_research.dc2 as dc2
from modelling_research.plotting import plotjoint_running_percentiles
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
# Match radius of 0.5 arcsec


# In[4]:


# Load the truth catalogs and count all of the rows
# This allows for pre-allocation of the truth table to its full size
# ... before determining HTM indices and writing refcats below
redo_refcat = True
butler_ref = dc2.get_refcat(make=redo_refcat)


# ## Crossmatch against reference catalog
# 
# All of the plots in this notebook compare stack measurements of sources measured by the LSST Science Pipelines cross-matched against a reference catalog that is part of the Science Pipelines repository. This reference catalog contains all sources brighter than 23 mags in r, and was generated through some kind of Task that must have read the original DC2 truth tables (more on that below).

# In[5]:


# Match with the refcat using afw's DirectMatchTask
cats = dc2.match_refcat(butler_ref)


# In[6]:


# Model plot setup
filters_single = ('g', 'r', 'i')
filters_multi = ('gri',)
models = {
    'mmf CModel': ('slot_ModelFlux_mag', 0),
    'mpf CModel': ('mg8cmodelpx', 2),
    'mpf Sersic': ('mg8serbpx', 1),
    'mpf Sersic Free Amp.': ('mg8serbapx', 8),
    'mpf Sersic x 2': ('mg8serx2sepx', 2),
    'mpf Sersic x 2 Free Amp.': ('mg8serx2seapx', 16),
}
models_stars = {
    model: models[model] for model in ['mmf CModel', 'mpf CModel', 'mpf Sersic']
}
args = dict(scatterleft=True, scatterright=True, limx=(15, 25.),)
lim_y = {
    "resolved": (-0.6, 0.4),
    "unresolved": (-0.1, 0.05),
}


# In[7]:


# Model plot functions

# Sum up fluxes from each component in multi-component models
def get_total_mag(cat_band, band, name_model, n_comps, is_multiprofit):
    return (
        cat_band[name_model] if not is_multiprofit else
        cat_band[f'{name_model}_c1_{band}_mag'] if n_comps == 1 else
        -2.5*np.log10(np.sum([
            10**(-0.4*cat_band[f'{name_model}_c{comp+1}_{band}_mag'])
            for comp in range(n_comps)], axis=0
        ))
    )

# Plot model - truth for all models and for mags and colours
def plot_matches(cats, resolved, models, bands=None, band_ref=None, band_ref_multi=None,
                 band_multi=None, colors=None, mag_max=None, **kwargs):
    if mag_max is None:
        mag_max = np.Inf
    bands_is_none = bands is None
    has_multi = band_multi is not None
    obj_type = 'Galaxies' if resolved else 'Point Sources'
    for tract, cat in cats.items():
        cat_truth, cats_meas = cat['truth'], cat['meas']
        if has_multi:
            cat_multi = cats_meas[band_multi]
            
        if bands_is_none:
            bands = cats_meas.keys()
            bands = (band for band in bands if band != band_multi)
        else:
            cats_meas = {band: cats_meas[band] for band in bands}
            
        cats_meas = {band: cat for band, cat in cats_meas.items()}
        if band_ref is None:
            band_ref = bands[0]
        cat_ref = cats_meas[band_ref]
        good_ref = (cat_ref['parent'] != 0) | (cat_ref['deblend_nChild'] == 0)
        good_ref = good_ref & cat_ref['detect_isPatchInner'] & ((cat_truth['id'] > 0) == resolved)
        cat_truth = cat_truth[good_ref]
        
        for band, cat in cats_meas.items():
            cats_meas[band] = cat[good_ref]
        mags_true = {band: -2.5*np.log10(cat_truth[f'lsst_{band}_flux']) + 31.4 for band in bands}
        if has_multi:
            cat_multi = cat_multi[good_ref]
        good_mags_true = {band: mags_true[band] < mag_max for band in bands}
        
        for name, (model, n_comps) in models.items():
            is_multiprofit = n_comps > 0
            name_model = f'multiprofit_{model}' if is_multiprofit else model
            cats_type = [(cats_meas, False)]
            if band_multi is not None and is_multiprofit:
                cats_type.append((cat_multi, True))
            mags_diff = {}
            for band in bands:
                mags_diff[band] = {}
                good_band = good_mags_true[band]
                true = mags_true[band]
                for cat, multi in cats_type:
                    y = get_total_mag((cat if multi else cat[band]),
                                      band, name_model, n_comps, is_multiprofit) - true
                    mags_diff[band][multi] = y
                    x, y = true[good_band], y[good_band]
                    good = np.isfinite(y)
                    postfix = f'({band_multi})' if multi else ''
                    title = f'DC2 {tract} {obj_type} {band}-band, {name}{postfix}, N={np.sum(good)}'
                    print(title)
                    labelx = f'${band}_{{true}}$'
                    plotjoint_running_percentiles(
                        x[good], y[good], title=title,
                        labelx=labelx, labely=f'${band}_{{model}}$ - {labelx}',
                        **kwargs
                    )
                    plt.show()
            if colors is None:
                colors = list(bands)
                colors = [(colors[idx], colors[idx+1]) for idx in range(len(colors)-1)]
            elif not colors:
                continue
            for b1, b2 in colors:
                bx = b2 if band_ref_multi is None else band_ref_multi
                good_band = good_mags_true[bx]
                for _, multi in cats_type:
                    x = mags_true[bx][good_band]
                    y = mags_diff[b1][multi][good_band] - mags_diff[b2][multi][good_band]
                    good = np.isfinite(y)
                    band = f'{b1}-{b2}'
                    postfix = f'({band_multi})' if multi else ''
                    title = f'DC2 {tract} {obj_type} {band}, {name}{postfix}, N={np.sum(good)}'
                    print(title)
                    plotjoint_running_percentiles(
                        x[good], y[good], title=title,
                        labelx=f'${bx}_{{true}}$', labely=f'${band}_{{model-true}}$',
                        **kwargs)
                    plt.show()


# ## Galaxy fluxes and colours
# 
# These plots show fluxes and colours for a variety of models, separately for DC2.1.1 (tract 3832) and DC2.2 (tract 3828).
# 
# Quick summary:
# 
# 1. Bright and faint galaxies alike are a challenge, with large scatter in the former and a negative (model too bright) bias in the latter.
#     - This is not very surprising; bright galaxies are more likely to be blended, but model mismatch shouldn't be as much of an issue here. Some kind of bias and scatter at the faint end is not surprising either, though how much and in which direction is not so clear.
# 2. Colours have considerably less scatter, even for single-band fits; that is, even though bright galaxies have differences much larger than expected from noise bias alone, these are about the same in each band.
#     - This is somewhat encouraging if not unexpected. It's not entirely a trivial result for g-r as it is for r-i (for low redshift galaxies anyways).
# 3. There is at best a modest improvement from meas_modelfit (mmf) CModel to MultiProFit (mpf) CModel. The main differences here are: free centroids for mpf vs fixed in mmf; slightly different Sersic approximation (more so for dev than exp); double Gaussian vs double Shapelet PSF in mpf vs mmf (the latter ought to be better); size prior in mmf; different optimizers; analytic vs finite difference gradients in mpf vs mmf. Of all of those differences, the free centroids are most likely to provide the (modest) benefit, if any.
#     - This is a more or less neutral result; I woudln't expect much difference between the two for any of the possible reasons.
# 4. There is a slightly more substantial improvment for single-band fluxes from CModel to Sersic; in particular, the distribution of galaxies with overestimated fluxes is tighter.
#     - This is also somewhat encouraging. We know that CModel is a kludge, but it's not trivial for slightly-but-not-entirely-more-principled Sersic fits to peform better.
# 5. Nearly everything is improved by fitting gri simultaneously - magnitudes, colours, etc. all have tigher scatter no matter the model or band. While the improvements in one-sigma scatter for magnitudes are not necessarily large, the 2+sigma scatter is significantly tighter, as are colours.
#     - This is very encouraging and non-trivial; one could have imagined galaxies with non-detections in some bands to have more biased measurements in the bands with detections, but that doesn't seem to be the case very often. Of course it's at least partly expected since the morphology of the galaxy is identical in all bands in single-component models, but the fact that this improves colours substantially without making magnitudes worse is a major bonus.

# In[8]:


# Galaxies
plot_matches(cats, True, models, bands=filters_single, band_ref=band_ref, band_ref_multi=band_ref,
             band_multi=filters_multi[0], mag_max=25, limy=lim_y['resolved'], **args)


# ## Point source fluxes and colours
# 
# There's not much to say here, other than that the stars look basically fine and the choice of PSF/source model and single- vs multi-band fitting makes little difference. Someone more versed in stellar photometry might have more to say about the small biases in the medians, outliers, etc.

# In[9]:


# Stars
plot_matches(cats, False, models_stars, bands=filters_single, band_ref=band_ref, band_ref_multi=band_ref,
             band_multi=filters_multi[0], mag_max=23, limy=lim_y['unresolved'], **args)

