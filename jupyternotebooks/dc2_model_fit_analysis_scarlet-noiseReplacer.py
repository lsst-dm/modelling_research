#!/usr/bin/env python
# coding: utf-8

# # Fitting DC2 v2 Simulation Data with MultiProFit - Scarlet Edition
# 
# This notebook plots results from fitting sources from the Vera Rubin Observatory/LSST Dark Energy Science Collaboration (DESC, https://lsstdesc.org/) DC2 simulations (http://lsstdesc.org/DC2-production/) using MultiProFit (https://github.com/lsst-dm/multiprofit, 'MPF'). Specifically, it reads the results of using a Task to fit exposures given an existing catalog with fits from meas_modelfit (https://github.com/lsst/meas_modelfit, 'MMF'). MMF implements a variant of the SDSS CModel algorithm (https://www.sdss.org/dr12/algorithms/magnitudes/#cmodel). In additional to CModel, MultiProFit allows for multi-band fitting, as well as fitting of Sersic profiles, multi-Gaussian approximations thereof ('MG' Sersic), and non-parametric radial profiles (Gaussian mixtures model with shared ellipse parameters, effectively having a Gaussian mixture radial profile). Thus, the main results are comparisons of the two codes doing basically the same thing (single-band exponential, de Vaucouleurs, and CModel linear combination fits), followed by plots highlighting Sersic vs CModel fits, more complicated double Sersic/free-amplitude models, and MultiProFit's multi-band fits.

# In[15]:


# Import requirements
import functools
from lsst.daf.persistence import Butler
import matplotlib as mpl
import matplotlib.pyplot as plt
import modelling_research.dc2 as dc2
import modelling_research.meas_model as mrMeas
from modelling_research.calibrate import calibrate_catalogs
from modelling_research.plotting import plotjoint_running_percentiles
import modelling_research.plot_matches as mrPlotMatches
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


# Load the DC2 repo butlers: we'll need them later
butlers_dc2 = {
    '2.2i': Butler('/project/dtaranu/dc2/scarlet/2020-12-01/ugrizy'),
}


# In[6]:


# Match with the refcat using astropy's matcher
truth_path = dc2.get_truth_path()
tracts = {3828: (f'{truth_path}scarlet/2020-12-01_mpf-noiseReplacer/', '2.2i'),}
filters_single = ('g', 'r', 'i', 'z')
filters_multi = ('griz',)
band_multi = filters_multi[0]
patch_min, patch_max = 2, 3
patches_regex = f"[{patch_min}-{patch_max}]"
patches_regex = f"{patches_regex},{patches_regex}"
get_path_cats = functools.partial(dc2.get_path_cats, patches_regex=patches_regex)
# Calibrate catalogs: this only needs to be done once; get_cmodel_forced should only be true for single bands for reasons
calibrate_cats = True
get_multiprofit = True
get_cmodel_forced = True
get_ngmix = True
get_scarlet = True
if calibrate_cats:
    butler_scarlet = Butler(f'/project/dtaranu/dc2/scarlet/2020-12-01/ugrizy') if get_scarlet else None
    path = tracts[3828][0]
    for bands in filters_single + filters_multi:
        butler_ngmix = Butler(f'/project/dtaranu/dc2/scarlet/2020-12-01_ngmix/{bands}') if get_ngmix else None
        is_single = len(bands) == 1
        files = [
            f'{path}{bands}/mpf_dc2_{bands}_3828_{x},{y}.fits'
            for x in range(patch_min, patch_max+1) for y in range(patch_min, patch_max+1)
        ]
        calibrate_catalogs(
            files, butlers_dc2, is_dc2=True, files_ngmix=butler_ngmix,
            butler_scarlet=butler_scarlet, get_cmodel_forced=get_cmodel_forced and is_single,
            overwrite_band=None, retry_delay=15, n_retry_max=3
        )
cats = dc2.match_refcat_dc2(
    butler_ref, match_afw=False, tracts=tracts, butlers_dc2=butlers_dc2,
    filters_single=filters_single, filters_multi=filters_multi, func_path=get_path_cats,
)


# ## Crossmatch against reference catalog
# 
# All of the plots in this notebook compare stack measurements of sources measured by the LSST Science Pipelines cross-matched against a reference catalog that is part of the Science Pipelines repository. This reference catalog contains all sources brighter than 23 mags in r, and was generated through some kind of Task that must have read the original DC2 truth tables (more on that below).

# In[13]:


# Model plot setup
# ngmix takes out a pixel scale factor of 2.5*log10(0.2**2)
# TODO: remove this offset when fixed in ngmix
offset_ngmix = -3.4948500216800937
model_specs = [
    ('PSF', 'base_PsfFlux', 1),
    ('Stack CModel', 'modelfit_CModel', 2),
]
if get_cmodel_forced:
    model_specs.append(('Forced CModel', 'modelfit_forced_CModel', 2))
if get_scarlet:
    model_specs.append(('Scarlet', 'scarlet', 0))
if get_multiprofit:
    model_specs.extend([
        ('MPF Gauss', 'multiprofit_gausspx', 1),
        ('MPF CModel', 'multiprofit_mg8cmodelpx', 2),
        ('MPF Sersic', 'multiprofit_mg8serbpx', 1),
        ('MPF Sersic Free Amp.', 'multiprofit_mg8serbapx', 8),
        #('MPF Sersic x 2', 'multiprofit_mg8serx2sepx', 2),
        #('MPF Sersic x 2 Free Amp.', 'multiprofit_mg8serx2seapx', 16),
    ])
if get_ngmix:
    model_specs.append(('ngmix bd', 'ngmix_bd', 0))

models = {
    desc: mrMeas.Model(desc, field, n_comps, mag_offset=offset_ngmix if mrMeas.is_field_ngmix(field) else None)
    for desc, field, n_comps in model_specs
}

models_stars = ['PSF', 'Stack CModel'] if get_cmodel_forced else []
if get_multiprofit:
    models_stars.extend(['MPF Gauss', 'MPF CModel', 'MPF Sersic'])
if get_ngmix:
    models_stars.append('ngmix bd')
if get_scarlet:
    models_stars.append('Scarlet')

models_stars = {model: models[model] for model in models_stars}

args = dict(scatterleft=True, scatterright=True,)
args_type = {
    'resolved': {
        'limx': (14.5, 24.5),
        'limy': (-0.6, 1.4),
    },
    'unresolved': {
        'limx': (16, 23),
        'limy': (-0.08, 0.06),
    },
}
mpl.rcParams['axes.labelsize'] = 15


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

# In[18]:


import importlib
importlib.reload(mrPlotMatches)


# In[19]:


# Galaxies
select_truths = mrPlotMatches.plot_matches(
    cats, True, models, filters_single, band_ref=band_ref, band_multi=band_multi,
    band_ref_multi=band_ref, mag_max=24.5, match_dist_asec=0.168,
    plot_compure=False, rematch=True, return_select_truth=True, **args_type['resolved'], **args
)


# In[27]:


select_truth = select_truths[3828]
cats_type = cats[3828]
cat_truth, cat_meas = cats_type['truth'], cats_type['meas']['griz']

mag_max=24.5
match_dist_asec=0.168

mags_true = {band: -2.5 * np.log10(cat_truth[f'lsst_{band}_flux']) + 31.4 for band in bands}
good_mags_true = {band: mags_true[band] < mag_max for band in bands}

indices, dists = (cats_type[x] for x in ('indices1', 'dists1'))
# Cheat a little and set negatives to -1
indices = np.copy(indices)
indices[dists > match_dist_asec] = -1
# bincount only works on non-negative integers, but we want to preserve the true indices and
# don't need the total count of unmatched sources
n_matches = np.bincount(indices + 1)[1:]
matches_multi = n_matches > 1
mags_true_ref = mags_true[band_ref]

# set multiple matches to integers < -1
for idx in np.where(matches_multi)[0]:
    matches = np.where(indices == idx)[0]
    brightest = np.argmax(mags_true_ref[matches])
    indices[matches] = -idx - 2
    indices[matches[brightest]] = idx

good = indices[select_truth] >= 0

cat_truth, indices = (x[select_truth][good] for x in (cat_truth, indices))
cats_meas = {band: cat.copy(deep=True).asAstropy()[indices]
             for band, cat in cats_meas.items()}
mags_true = {band: -2.5 * np.log10(cat_truth[f'lsst_{band}_flux']) + 31.4 for band in bands}
good_mags_true = {band: mags_true[band] < mag_max for band in bands}
if has_multi:
    cat_multi = cat_multi.copy(deep=True).asAstropy()[indices]


# ## Point source fluxes and colours
# 
# There's not much to say here, other than that the stars look basically fine up to the saturation limit of ~16 and the choice of PSF/source model and single- vs multi-band fitting makes little difference. Someone more versed in stellar photometry might have more to say about the small biases in the medians, outliers, etc.

# In[9]:


# Stars
mrPlotMatches.plot_matches(
    cats, False, models_stars, filters_single, band_ref=band_ref, band_multi=band_multi,
    band_ref_multi=band_ref, mag_max=23, match_dist_asec=0.168,
    plot_compure=False, rematch=True, **args_type['unresolved'], **args
)


# In[10]:


# Compare ngmix vs mpf g-r colours. They agree almost shockingly well for ~80-90% of sources.
cat_mb = cats[3828]['meas'][band_multi]
is_good = cat_mb['detect_isPatchInner'] & cat_mb['detect_isTractInner'] & (cat_mb['parent'] != 0) & ~cat_mb['merge_footprint_sky']
cat_good = cat_mb[is_good]
if get_ngmix:
    models_gmr = ['ngmix bd', 'MPF Sersic']
    gmr = {model: models[model].get_color_total(cat_good, 'g', 'r') for model in models_gmr}
    flux = models['ngmix bd'].get_mag_total(cat_good, 'r')
    good = flux < args_type['resolved']['limx'][1]
    for model in models_gmr:
        good &= np.isfinite(gmr[model])
    plotjoint_running_percentiles(
        flux[good],
        gmr['ngmix bd'][good] - gmr['MPF Sersic'][good],
        labelx='$r_{ngmix,bd}$', labely='${g-r}_{ngmix,bd}$ - ${g-r}_{mpf,Sersic}$',
        limx=args_type['resolved']['limx'], limy=(-0.2, 0.2),
        title=f'DC2 3828 All ngmix bd vs MPF Sersic ({band_multi})',
        **args
    )


# In[11]:


# r-band size mass for all objects
# TODO: Implement getting sizes from model objects
# TODO: Get forced sizes somehow
band = 'r'
scale_pix = 0.2
sizes_model = {
    'Forced CModel': np.log10(scale_pix*np.sqrt(0.5*(
        cat_good['modelfit_CModel_ellipse_xx'] + cat_good['modelfit_CModel_ellipse_yy'])))
} if get_cmodel_forced else {}
sizes_model['MPF Sersic'] = np.log10(
    scale_pix*np.sqrt(0.5*(
        cat_good['multiprofit_mg8serbpx_c1_sigma_x']**2 
        + cat_good['multiprofit_mg8serbpx_c1_sigma_y']**2))
)
if get_ngmix:
    sizes_model['ngmix bd'] = np.log10(np.sqrt(0.5*cat_good['ngmix_bd_T']))
mag_min = args_type['resolved']['limx'][0]

for mag_max in (22.5, 24.5):
    for name_model, sizes in sizes_model.items():
        is_ngmix = name_model.startswith('ngmix')
        cat_mag = cat_good if name_model != 'Forced CModel' else cats[3828]['meas'][band][is_good]
        mag = models[name_model].get_mag_total(cat_mag, band)
        good = (mag < mag_max) & np.isfinite(sizes)
        plotjoint_running_percentiles(
            mag[good], sizes[good], limx=(mag_min, mag_max), limy=(-2.5, 2),
            labelx=f'${band}_{{{name_model}}}$',
            labely='log10($R_{eff}$/arcsec)' if not is_ngmix else 'log10($\sigma$/arcsec)',
            title=f'DC2 3828 Size-Magnitude {name_model} ({band_multi}) N={np.sum(good)}',
            **args,
        )
        plt.show()


# In[12]:


# Timing ngmix and MultiProFit
cat_mb = cats[3828]['meas'][band_multi]
times = {
    'MPF Sersic': np.log10(cat_mb['multiprofit_gausspx_time'] + cat_mb['multiprofit_mg8expgpx_time']
                       + cat_mb['multiprofit_mg8devepx_time'] + cat_mb['multiprofit_mg8serbpx_time']),
}
if get_ngmix:
    times['ngmix bd'] = np.log10(cat_mb['ngmix_time'])

model_ref = 'Stack CModel'
times_ref = None
bands_done, bands_skipped = 0, 0
for band in band_multi:
    cat = cats[3828]['meas'].get(band)
    if cat is not None:
        times_cm = cat['modelfit_CModel_exp_time'] + cat['modelfit_CModel_dev_time'] + cat['modelfit_CModel_initial_time']
        if times_ref is None:
            times_ref = times_cm
        else:
            times_ref += times_cm
        bands_done += 1
    else:
        bands_skipped += 1
times_ref *= (bands_skipped + bands_done)/bands_done
times[model_ref] = np.log10(times_ref)

lim_time = (-3.2, 1.8)
lim_y = (-2.1, 4.1)
# ha ha, how negative
good_times = None
for times_model in times.values():
    good_time = (times_model > lim_time[0]) & (times_model < lim_time[1])
    if good_times is None:
        good_times = good_time
    else:
        good_times &= good_time
       
times_ref = times[model_ref][good_times]
for model, times_model in times.items():
    print(f'{model} t_total={np.sum(10**times_model[good_times])}')
    if model != model_ref:
        plotjoint_running_percentiles(
            times_ref, times_model[good_times] - times_ref,
            limx=lim_time, limy=lim_y,
            labelx='log10($t_{cmodel}$)', labely=f'log10($t_{{{model}}}/t_{{{model_ref}}}$)',
            title=f'DC2 3828 All Timing {model} vs {model_ref} ({band_multi})',
            scatterleft=True, scatterright=True,
        )
        plt.show()


# In[ ]:




