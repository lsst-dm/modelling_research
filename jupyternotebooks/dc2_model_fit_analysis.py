#!/usr/bin/env python
# coding: utf-8

# # Fitting DC2 v2 Simulation Data with MultiProFit
# 
# This notebook plots results from fitting sources from the Vera Rubin Observatory/LSST Dark Energy Science Collaboration (DESC, https://lsstdesc.org/) DC2 simulations (http://lsstdesc.org/DC2-production/) using MultiProFit (https://github.com/lsst-dm/multiprofit, 'MPF'). Specifically, it reads the results of using a Task to fit exposures given an existing catalog with fits from meas_modelfit (https://github.com/lsst/meas_modelfit, 'MMF'). MMF implements a variant of the SDSS CModel algorithm (https://www.sdss.org/dr12/algorithms/magnitudes/#cmodel). In additional to CModel, MultiProFit allows for multi-band fitting, as well as fitting of Sersic profiles, multi-Gaussian approximations thereof ('MG' Sersic), and non-parametric radial profiles (Gaussian mixtures model with shared ellipse parameters, effectively having a Gaussian mixture radial profile). Thus, the main results are comparisons of the two codes doing basically the same thing (single-band exponential, de Vaucouleurs, and CModel linear combination fits), followed by plots highlighting Sersic vs CModel fits, more complicated double Sersic/free-amplitude models, and MultiProFit's multi-band fits.

# In[1]:


# Import requirements
from lsst.daf.persistence import Butler
import matplotlib as mpl
import matplotlib.pyplot as plt
import modelling_research.dc2 as dc2
import modelling_research.meas_model as mrMeas
from modelling_research.plotting import plotjoint_running_percentiles
from modelling_research.plot_matches import plot_matches
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
    '2.2i': Butler('/datasets/DC2/repoRun2.2i/rerun/w_2020_03/DM-22816/'),
}


# In[6]:


# Calibrate catalogs: this only needs to be done once; get_cmodel_forced should only be true for single bands for reasons
calibrate_cats = False
if calibrate_cats:
    from modelling_research.calibrate import calibrate_catalogs
    butler_ngmix_gri = Butler('/project/dtaranu/dc2/ngmix/gri')
    files = [f'/project/dtaranu/dc2/2020-01-31/i/mpf_dc2_i_3828_{x},{y}.fits' for x in range(7) for y in range (7)]
    cats = calibrate_catalogs(files, butlers_dc2, is_dc2=True, files_ngmix=butler_ngmix_gri, get_cmodel_forced=False)


# ## Crossmatch against reference catalog
# 
# All of the plots in this notebook compare stack measurements of sources measured by the LSST Science Pipelines cross-matched against a reference catalog that is part of the Science Pipelines repository. This reference catalog contains all sources brighter than 23 mags in r, and was generated through some kind of Task that must have read the original DC2 truth tables (more on that below).

# In[7]:


# Match with the refcat using astropy's matcher
truth_path = dc2.get_truth_path()
tracts = {3828: (f'{truth_path}2020-01-31/', '2.2i'),}
filters_single = ('g', 'r', 'i')
filters_multi = ('gri',)
cats = dc2.match_refcat_dc2(butler_ref, match_afw=False, tracts=tracts, butlers_dc2=butlers_dc2,
                            filters_single=filters_single, filters_multi=filters_multi)


# In[8]:


# Model plot setup
# ngmix takes out a pixel scale factor of 2.5*log10(0.2**2)
# TODO: remove this offset when fixed in ngmix
offset_ngmix = -3.4948500216800937
models = {
    desc: mrMeas.Model(desc, field, n_comps, mag_offset=offset_ngmix if mrMeas.is_field_ngmix(field) else None)
    for desc, field, n_comps in [
        ('PSF', 'base_PsfFlux', 0),
        ('stack CModel', 'modelfit_CModel', 0),
        ('forced CModel', 'modelfit_forced_CModel', 0),
        ('MPF CModel', 'mg8cmodelpx', 2),
        ('MPF Sersic', 'mg8serbpx', 1),
        ('MPF Sersic Free Amp.', 'mg8serbapx', 8),
        ('MPF Sersic x 2', 'mg8serx2sepx', 2),
        ('MPF Sersic x 2 Free Amp.', 'mg8serx2seapx', 16),
        ('ngmix bd', 'ngmix_bd', 0),
    ]
}
models_stars = {
    model: models[model] for model in ['PSF', 'stack CModel', 'MPF CModel', 'MPF Sersic', 'ngmix bd']
}
args = dict(scatterleft=True, scatterright=True,)
args_type = {
    'resolved': {
        'limx': (14.5, 24.5),
        'limy': (-0.6, 0.4),
    },
    'unresolved': {
        'limx': (16, 23),
        'limy': (-0.08, 0.06),
    },
}
band_multi = filters_multi[0]
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

# In[9]:


# Galaxies
plot_matches(
    cats, True, models, filters_single, band_ref=band_ref, band_multi=band_multi,
    band_ref_multi=band_ref, mag_max=24.5, match_dist_asec=0.168,
    plot_compure=False, rematch=True, **args_type['resolved'], **args
)


# ## Point source fluxes and colours
# 
# There's not much to say here, other than that the stars look basically fine up to the saturation limit of ~16 and the choice of PSF/source model and single- vs multi-band fitting makes little difference. Someone more versed in stellar photometry might have more to say about the small biases in the medians, outliers, etc.

# In[10]:


# Stars
plot_matches(
    cats, False, models_stars, filters_single, band_ref=band_ref, band_multi=band_multi,
    band_ref_multi=band_ref, mag_max=23, match_dist_asec=0.168,
    plot_compure=False, rematch=True, **args_type['unresolved'], **args
)


# In[11]:


# Compare ngmix vs mpf g-r colours. They agree almost shockingly well for ~80-90% of sources.
cat_mb = cats[3828]['meas']['gri']
cat_good = cat_mb[cat_mb['detect_isPatchInner'] & cat_mb['detect_isPrimary']]
models_gmr = ['ngmix bd', 'MPF Sersic']
gmr = {model: models[model].get_total_color(cat_good, 'g', 'r') for model in models_gmr}
flux = models['ngmix bd'].get_total_mag(cat_good, 'r')
good = flux < args_type['resolved']['limx'][1]
for model in models_gmr:
    good &= np.isfinite(gmr[model])
plotjoint_running_percentiles(
    flux[good],
    gmr['ngmix bd'][good] - gmr['MPF Sersic'][good],
    labelx='$r_{ngmix,bd}$', labely='${g-r}_{ngmix,bd}$ - ${g-r}_{mpf,Sersic}$',
    limx=args_type['resolved']['limx'], limy=(-0.2, 0.2),
    title='DC2 3828 All ngmix bd vs MPF Sersic (gri)',
    **args
)


# In[12]:


# r-band size mass for all objects
# TODO: Implement getting sizes from model objects
# TODO: Get forced sizes somehow
scale_pix = 0.2
sizes_model = {
    'stack CModel': np.log10(scale_pix*np.sqrt(0.5*(
        cat_good['modelfit_CModel_ellipse_xx'] + cat_good['modelfit_CModel_ellipse_yy']))),
    'MPF Sersic': np.log10(scale_pix*np.sqrt(0.5*(
        cat_good['multiprofit_mg8serbpx_c1_sigma_x']**2 + 
        cat_good['multiprofit_mg8serbpx_c1_sigma_y']**2))),
    'ngmix bd': np.log10(np.sqrt(0.5*cat_good['ngmix_bd_T'])),
}
for name_model, sizes in sizes_model.items():
    is_ngmix = name_model.startswith('ngmix')
    mag = models[name_model].get_total_mag(cat_good, 'r')
    good = (mag < 24.5) & np.isfinite(sizes)
    plotjoint_running_percentiles(
        mag[good], sizes[good], limx=args_type['resolved']['limx'], limy=(-2.5, 2),
        labelx='$r_{model}$', labely='log10($R_{eff}$/arcsec)' if not is_ngmix else 'log10($\sigma$/arcsec)',
        title=f'DC2 3828 Size-Magnitude {name_model} ({band_multi}) N={np.sum(good)}',
        **args,
    )
    plt.show()


# In[13]:


# Timing ngmix and MultiProFit
cat_mb = cats[3828]['meas'][band_multi]
times = {
    'ngmix': np.log10(cat_mb['ngmix_time']),
    'mpf': np.log10(cat_mb['multiprofit_gausspx_time'] + cat_mb['multiprofit_mg8expgpx_time'] + 
                    cat_mb['multiprofit_mg8devepx_time'] + cat_mb['multiprofit_mg8serbpx_time']),
}

model_ref = 'cmodel'
times_ref = None
for band in band_multi:
    cat = cats[3828]['meas'][band]
    times_cm = cat['modelfit_CModel_exp_time'] + cat['modelfit_CModel_dev_time'] + cat['modelfit_CModel_initial_time']
    if times_ref is None:
        times_ref = times_cm
    else:
        times_ref += times_cm
times[model_ref] = np.log10(times_cm)

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

