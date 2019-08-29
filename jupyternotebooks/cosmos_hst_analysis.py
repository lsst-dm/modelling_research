#!/usr/bin/env python
# coding: utf-8

# # Analyzing MultiProFit galaxy model performance on COSMOS data
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

sns.set_style('darkgrid')
mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['image.origin'] = 'lower'
sns.set(rc={'axes.facecolor': '0.85', 'figure.facecolor': 'w'})


# ### Read the table with the results

# In[3]:


from modelling_research.plot_multiprofit_cosmos import readtable
from multiprofit.gaussutils import reff_to_sigma
filename = '../data/multiprofit-cosmos-fits.csv'
tab = readtable(filename)
# There's at least one point with 1e-15 flux
tab["cosmos.hst.ser.flux"] = np.clip(tab["cosmos.hst.ser.flux"], 5e-1, np.Inf)
prefixes_models = [
    (["mg4", "mg8"], ["serbpx", "sermpx", "faserpx"]),
    ([""], ["serb"]),
]
for prefixes, models in prefixes_models:
    for src in ["hst","hst2hsc"]:
        for prefix in prefixes:
            for model in models:
                name_column = f"profit.{src}.{prefix}{model}.re.1.1"
                values = tab[name_column]
                if prefix == "":
                    values *= reff_to_sigma(1)
                else:
                    values /= reff_to_sigma(1)
                tab[name_column] = np.clip(values, 1e-2, np.Inf)


# ### Make joint parameter plots
# 
# This section makes an alarmingly large number of plots. Most of them check consistency between the MultiProFit fits and the COSMOS-GalSim catalog values with marginalized histograms thereof. The last set compares the results from HST fits to synthetic HSC-quality images of the same galaxy. More specifically, we take the best-fit HST F814W model (single MG8 Sersic for now), convolve it with the reconstructed HSC i-band PSF, re-normalize the magnitude to match the original HSC image since they're different resolutions and the F814W band is wider than HSC-I, and add noise based on the observed HSC inverse variance map (which isn't completely consistent with the 'true' model, but it's close enough and saves the effort of having to back out the HSC background, etc.).
# 
# Notes:
# 
# The colour coding is by log(Sersic index) from the HST true Sersic fit, such that low values (~disk-like) are blue and high values (~bulge-like) are red.
# Any linear relations between points in plots of the Sersic index are due to the different limits imposed by the various fit methods: n=[0.1, 6] in the GalSim catalog; n=[0.3, 6.3] for MultiProFit Sersic fits (set by GalSim), and n=[0.5, 6.3] for MultiProFit MGA Sersic.

# In[4]:


from modelling_research.plot_multiprofit_cosmos import plotjointsersic
varnames = ["flux", "re.1", "nser.1"]


# ### COSMOS-HST: GalSim catalog Sersic fit versus MultiProfit Sersic fit
# 
# This is a sanity check comparing MultiProFit's true Sersic profile fit (albeit convolved with a 3-Gaussian PSF rather than the PSF image for direct comparison) compared to the values in the GalSim catalog.
# 
# At some point I may try true Sersic fits with the PSF images, but these are fairly slow and I was mainly interested in comparing true Sersic vs MGA Sersic using Gaussian PSFs, because we'll almost certainly use only GMMs in LSST because of how much faster they are to render.
# 
# The results are reasonably consistent, though not without systematic differences. The methods used for the GalSim catalog are described in the GREAT3 Challenge Handbook (https://arxiv.org/abs/1308.4982). As far as I can tell, they used TinyTim-generated PSF images (which I think are provided in the catalog) rather than a Gaussian mixture model thereof. Additionally, they used different limits: Sersic index 0.1≤n≤6; 0.1 and 6; axis ratio 0.05≤q≤1, and (2)R_eff≤image size. Also, their Sersic profles are cut off at a radius that varies smoothly from 4R_eff at n= 1 to 8 R_eff at n=4. Given that, at fixed n one would expect smaller R_eff from MultiProFit. However, MultiProFit tends to prefer larger R_eff, n *and* flux. It is possible that the COSMOS fits are not integrating the Sersic profiles as accurately as GalSim does - the code paper (Lackner & Gunn 2012) does not offer much detail on how their sub-pixel Sersic profile integration is done (if it is at all). Accurate sub-pixel integration is most important for large Sersic n profiles, so this could explain why there are larger differences for large n profiles than for small n.

# In[5]:


_ = plotjointsersic(tab, 'profit.hst.serb', 'cosmos.hst.ser', varnames, plotratiosjoint=False, postfixx='1')
_ = plotjointsersic(tab, 'cosmos.hst.ser', 'profit.hst.serb', varnames, postfixy='1')


# ### COSMOS-HST: MultiProFit Sersic vs MultiProFit MGA Sersic (N=8)
# 
# How close is the N=8 MGA to a true Sersic fit? The answer is quite close, except at n>4 where the MGA intentionally does not match the inner profile (see the introduction for more information on the derivation of MGA Sersic weights) and for n < 0.5 where the MGA can't possibly match a true Sersic (Gaussians mixtures cannot reproduce the n -> 0 limit of a top hat function in 1D; in 2D one could have many spatially offset Gaussians but this is unreasonably expensive).
# 
# It seems like there are two different tracks of outliers: objects with n>4 where the MGA has some large but not unreasonable scatter, and objects with 2.5>n>4 where the MGA has systematically lower n (somewhat more mysterious).
# 
# There is a noticeable trend for large n profiles to have smaller sizes in the MGA Sersic. For the cases where both profiles hit the upper limit of n=6.3, it's important to note that the profiles are quite different - the MGA Sersic has a much shallower cusp (less central flux) and steeper wings (marginally less outer flux). At fixed R_eff, this is also true of the Sersic profile itself - larger n profiles are cuspier and have shallower wings. R_eff also generally correlates with n for reasons that are not really obvious. I suspect it is because for a larger Sersic n, a larger R_eff stretches out the steep inner cusp, and also lowers the surface brightness of the shallow outer profile so that it blends into the sky. At any rate, one can interpret the MGA Sersic as having a smaller effective Sersic n than the true Sersic profile, and therefore it is likely to have a smaller R_eff at fixed n.

# In[6]:


_ = plotjointsersic(tab, 'profit.hst.mg8serbpx', 'profit.hst.sermpx', varnames, plotratiosjoint=False, postfixx='1', postfixy='1')
_ = plotjointsersic(tab, 'profit.hst.sermpx', 'profit.hst.mg8serbpx', varnames, postfixx='1', postfixy='1')


# ### COSMOS-HST: MultiProFit Sersic vs MultiProFit MGA Sersic (N=4)
# 
# How close is the N=4 MGA to a true Sersic fit? Reasonably close, but there's clearly more scatter than with an N=8 fit, particularly at n>2 where more Gaussians are needed to match the shallow outer profile.

# In[7]:


_ = plotjointsersic(tab, 'profit.hst.mg8sermpx', 'profit.hst.sermpx', varnames, plotratiosjoint=False, postfixx='1', postfixy='1')
_ = plotjointsersic(tab, 'profit.hst.sermpx', 'profit.hst.mg4sermpx', varnames, postfixx='1', postfixy='1')
_ = plotjointsersic(tab, 'profit.hst.mg8sermpx', 'profit.hst.mg4sermpx', varnames, postfixx='1', postfixy='1')


# ### COSMOS-HST: MultiProFit GMM (N=8) vs MultiProFit MGA Sersic
# 
# How different is a free GMM compared to a constrained Sersic MGA? Given the discussion above, one might expect real "early-type" (Sersic n > 2.5ish) galaxies to have shallower cusps than the Sersic profile predicts, in which case Sersic fits would tend to overestimate R_eff in order to stretch out the unrealistically steep cusp. Similarly, true Sersic fits predict shallow, extended outer wings, and therefore overestimate flux if these wings don't really exist. Therefore a free Gaussian mixture could be expected to prefer smaller effective radii and fluxes for large Sersic n. As might be expected, large-n galaxies tend to smaller fluxes and sizes with a full GMM, whereas the opposite is true for low-n galaxies.

# In[8]:


_ = plotjointsersic(tab, 'profit.hst.mg8faserpx', 'profit.hst.mg8sermpx', ["flux", "re.1", "chisqred"], plotratiosjoint=False, postfixx='1', postfixy='1')
_ = plotjointsersic(tab, 'profit.hst.mg8sermpx', 'profit.hst.mg8faserpx', ["flux", "re.1"], postfixx='1', postfixy='1')


# ### COSMOS-HST: MultiProFit GMM (N=4) vs MultiProFit MGA Sersic and N=8 GMM
# 
# Are the N=4 GMM parameters consistent with the N=8? Pretty much, except for a few extreme outliers. The differences in reduced chi-squared are also very modest. Thus, it's not surprising that the comparison between N=4 GMM vs N=8 MGA Sersic looks about the same as N=8 GMM vs N=8 MGA Sersic, too.

# In[9]:


_ = plotjointsersic(tab, 'profit.hst.mg4faserpx', 'profit.hst.mg8sermpx', ["flux", "re.1", "chisqred"], plotratiosjoint=False, postfixx='1', postfixy='1')
_ = plotjointsersic(tab, 'profit.hst.mg8sermpx', 'profit.hst.mg4faserpx', ["flux", "re.1"], postfixx='1', postfixy='1')


# ### COSMOS-HST: MultiProFit GMM vs MultiProFit MGA Sersic (N=4 x 2 Components)
# 
# How much better is a 2-component GMM (shared shape, so the radial profile is a Gaussian mixture) than a 2-component MGA Sersic? Also compare to N=8 Sersic MGA, which has fewer parameters but covers a different space from the N=4x2 GMM, since each component has 8 Gaussians.
# Note that previously this model had a free scale radius, i.e. the fit went double Sersic -> fit amplitudes -> re-fit scale radius; since I haven't implemented Jacobian computation for scale radii, this version skips that final scale radius fitting step.

# In[10]:


_ = plotjointsersic(tab, 'profit.hst.mg4fax2px', 'profit.hst.mg4serserbpx', ["flux", "chisqred"], plotratiosjoint=False, postfixx='1', postfixy='1')
_ = plotjointsersic(tab, 'profit.hst.mg8serserbpx', 'profit.hst.mg4fax2px', ["flux", "chisqred"], postfixx='1', postfixy='1')


# ### MultiProFit MGA Sersic (N=8): COSMOS-HST vs COSMOS-HSC
# 
# How well does a fit to synthetic, Subaru-HSC UltraDeep data recover the parameters derived from the HST fits? It should do well modulo noise bias, because the image is generated using the best-fit HST model parameters, but convolved with the actual HSC PSF for that object and with added noise from the actual HSC variance map.
# 
# In general, fluxes are reasonably robust, with the scatter increasing for fainter objects, as one might expect. Sizes are recovered well down to R_eff ~ 10 pixels but are biased low below that level. This seems rather high; I would have hoped for good size recovery down to R_eff ~3 pixels. Sersic index recovery is generally poor, with a noticeable bias to lower Sersic n. It's not yet clear why these trends exist. It appears as if there is a track of galaxies with well-recovered fluxes and seemingly random errors on Sersic n and R_eff, and another where the parameter ratios are all correlated.

# In[11]:


fluxes = tab['profit.hst2hsc.mg8sermpx.flux.1']
flux_scales = tab['profit.hst2hsc.mg8sermpx.flux_scale_hst2hsc.1']
print(flux_scales)
tab['profit.hst2hsc.mg8sermpx.flux.1'] /= flux_scales
_ = plotjointsersic(tab, 'profit.hst2hsc.mg8sermpx', 'profit.hst.mg8sermpx', varnames, plotratiosjoint=False, postfixx='1', postfixy='1')
_ = plotjointsersic(tab, 'profit.hst.mg8sermpx', 'profit.hst2hsc.mg8sermpx', varnames, postfixx='1', postfixy='1')
fluxes = tab['profit.hst2hsc.mg8sermpx.flux.1']


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

# In[12]:


# Which are the best models?

ordersmg = ['4', '8']

modelsmgser = {
    order: ['mg' + order + model + 'px' for model in ['expm', 'expg', 'deve', 'devg', 'devm', 'serm', 'serb']]
    for order in ordersmg
}
modelsmgsertwo = {
    order: ['mg' + order + model + 'px' for model in ['cmodel', 'devexp', 'devexpc', 'devexp2', 'serserb']]
    for order in ordersmg
}
modelsmgw = {
    order: ['mg' + order + 'fa' + model + 'px' for model in ['exp', 'dev', 'ser']]
    for order in ordersmg
}
modelsmgwr = {
    order: []#'mg' + order + 'wr' + model + 'px' for model in (['exp', 'dev'] if order == '4' else []) + ['ser']]
    for order in ordersmg
}
modelsprofit = {
    "single": ["gausspx"] + modelsmgser['4'] + modelsmgser['8'] + ["sermpx", "serb"],
    "double": modelsmgsertwo['4'] + modelsmgsertwo['8'],
    "mgw": modelsmgw['4'] + modelsmgw['8'],
    "mgwr": modelsmgwr['4'] + modelsmgwr['8'],
#    "mga": ["mg8bpx", "mg4bpx", "mg4x2px"],
}

def getpostfix(model):
    return "1"

modelslist = [
    ('Single-component Sersic', modelsprofit["single"]),
    ('Single-component Sersic + linear N=4 GMM', modelsprofit["single"] + modelsmgw['4']),
    ('Single-component Sersic + rscale N=4 GMM', modelsprofit["single"] + modelsmgw['4'] + modelsmgwr['4']),
    ('Single-component Sersic + rscale GMM', modelsprofit["single"] + modelsprofit['mgwr']),
    ('One- and two-component, no GMM', modelsprofit["single"] + 
     [model for model in modelsprofit["double"] if not model.endswith('serserbpx')]),
    ('One- and two-component, no GMMx2', modelsprofit["single"] + 
     [model for model in modelsprofit["double"] if not model.endswith('serserbpx')] + 
     [model for model in modelsprofit['mgw']]),
    ('All', modelsprofit["single"] + modelsprofit["double"] + modelsprofit['mgw']),
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
plotmodels = True
if plotmodels:
    for obs in ['hst', 'hst2hsc']:
        for colx, coly in [("mg8serbpx", "mg8devexppx"), ("mg8serbedpx", "mg8cmodelpx"), 
                           ("mg8devexppx", "mg8cmodelpx"), ("mg8serbedpx", "serb"),
                           ("mg8serbedpx", "mg8serbpx"), ("mg8serggpx", "mg8serbpx"),
                           ("mg8serserbpx", "mg8faserpx"), ("mg4fax2px", "mg8serserbpx")]:
            prefixx = ".".join(["profit", obs, colx])
            prefixy = ".".join(["profit", obs, coly])
            plotjointsersic(
                tab, prefixx, prefixy, ['chisqred'], columncolor="profit.hst.mg8serbpx.nser.1.1",
                cmap = mpl.colors.ListedColormap(sns.color_palette("RdYlBu_r", 100)),
                hist_kws={'log': True}, postfixx=getpostfix(colx), postfixy=getpostfix(coly))


# ### MG Sersic fixed-n vs free-n fits
# 
# These plots compare fixed- versus free-n single Sersic fits. This is to help determine an optimal workflow for fitting increasingly complicated models. For example, the Gaussian model is the fastest to evaluate, but it's also the least likely to be the best fit out of the four fixed-n models (n=0.5, 1, 2, 4).
# 
# Another plot compares the goodness of fit for the free-n versus the best-fit fixed-n model, showing that in most cases a fixed-n model is not far off - that is, there are not many galaxies where there's a very significant benefit to fitting n=1.5 or n=3 vs n=2. The exception is the galaxies that prefer n=6 to n=4.
# 
# The last plot compares the free Sersic to an exponential, showing that an exponential is 'good enough' for most galaxies except those with best-fit n>4. This suggests that the exponential model would be a reasonable choice if we had to pick just one, which is in line with expectations that most of the galaxies in any given deep field are disky.

# In[13]:


# Now compare only single-component models: Sersic vs best fixed n
modelsfixedn = ["gausspx", "mg8expmpx", "mg8devmpx"] 
chisqredcolsfixedn = {
    model: ".".join(["profit", "hst", model, "chisqred", "1"]) 
    for model in modelsfixedn
}
modelbest = tab[list(chisqredcolsfixedn.values())].idxmin(axis=1)
print('Counts of best fixed-n Sersic model:')
print(modelbest.value_counts())
# I seriously cannot figure out how to slice with modelbest
# Surely there's a better way to do this?
modelchisqmin = tab[list(chisqredcolsfixedn.values())].min(axis=1)
for colxname in ["mg8serbpx", "sermpx"]: 
    labelbest = 'log10(chisqred) ({}/{})'.format(colxname, "best([gauss,exp,n2,dev]px)")
    ratiobest = tab[chisqredcols[colxname]]/modelchisqmin
    # Plots:
    # How much better is Sersic than the best [gauss/exp/dev] vs how good is the 
    # fit and vs Sersic index
    # As above but vs exp only
    colnser = colxname.join(["profit.hst.", ".nser.1.1"])
    cols = [
        (tab[chisqredcols[colxname]], ratiobest,
         'log10(chisqred) ({})'.format(colxname), labelbest), 
        (tab[colnser], ratiobest, 'log10(n_ser)', labelbest),
        (tab[colnser], tab[chisqredcols[colxname]]/tab[chisqredcols["mg8expmpx"]],
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
