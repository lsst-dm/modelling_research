#!/usr/bin/env python
# coding: utf-8

# # Fitting DC2 v2 Simulation Data with MultiProFit
# 
# This notebook plots results from fitting sources from the Vera Rubin Observatory/LSST Dark Energy Science Collaboration (DESC, https://lsstdesc.org/) DC2 simulations (http://lsstdesc.org/DC2-production/) using MultiProFit (https://github.com/lsst-dm/multiprofit, 'MPF'). Specifically, it reads the results of using a Task to fit exposures given an existing catalog with fits from meas_modelfit (https://github.com/lsst/meas_modelfit, 'MMF'). MMF implements a variant of the SDSS CModel algorithm (https://www.sdss.org/dr12/algorithms/magnitudes/#cmodel). In additional to CModel, MultiProFit allows for multi-band fitting, as well as fitting of Sersic profiles, multi-Gaussian approximations thereof ('MG' Sersic), and non-parametric radial profiles (Gaussian mixtures model with shared ellipse parameters, effectively having a Gaussian mixture radial profile). Thus, the main results are comparisons of the two codes doing basically the same thing (single-band exponential, de Vaucouleurs, and CModel linear combination fits), followed by plots highlighting Sersic vs CModel fits, more complicated double Sersic/free-amplitude models, and MultiProFit's multi-band fits.

# In[1]:


# Import requirements
import esutil
import glob
from lsst.afw.table import Schema, SourceCatalog
from lsst.daf.persistence import Butler
from lsst.meas.algorithms import ingestIndexReferenceTask as ingestTask, IndexerRegistry
from lsst.meas.astrom import DirectMatchTask, DirectMatchConfig 
import matplotlib as mpl
import matplotlib.pyplot as plt
from modelling_research.plotting import plotjoint_running_percentiles
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3
from timeit import default_timer as timer


# In[2]:


# Setup for plotting
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['image.origin'] = 'lower'
sns.set(rc={'axes.facecolor': '0.85', 'figure.facecolor': 'w'})


# In[3]:


# Where I keep DC2 stuff
dc2_dst_path = '/project/dtaranu/dc2/'


# In[4]:


# DC2/LSST bands
bands_dc2 = ['u', 'g', 'r', 'i', 'z', 'y']


# In[5]:


# Catalog matching settings
# Reference band to match on. Should be 'r' for DC2 because reasons (truth refcats are r mag limited)
band_ref = 'r'
# Match radius of 0.5 arcsec
config = DirectMatchConfig(matchRadius=0.5)


# In[6]:


# One-line convenience function to print things
def time_print(time, format_time='.1f', prefix='', postfix=''):
    time_new = timer()
    timing = '' if time is None else (
        f'{time_new-time:{format_time}}s' if (format_time is not None) else f'{time_new-time}s')
    print(f'{prefix}{timing}{postfix}')
    return time_new


# In[7]:


# Load the truth catalogs and count all of the rows
# This allows for pre-allocation of the truth table to its full size
# ... before determining HTM indices and writing refcats below
redo_refcat = True
if redo_refcat:
    time = None
    files = glob.glob(f'{dc2_dst_path}truth_cats/truth_summary_hp*.sqlite3')
    truth_cats = {}
    n_files = len(files)
    for idx, file in enumerate(files):
        with sqlite3.connect(file) as db: 
            healpix = file.split('truth_summary_hp')[1].split('.sqlite3')[0]
            prefix = f'Counted in  ' if idx > 0 else 'Counting starting'
            time = time_print(time, format_time='.2f', prefix=prefix, postfix=f'; counting {healpix} ({idx+1}/{n_files}) file={file}')
            cursor = db.cursor() 
            n_rows = cursor.execute('SELECT COUNT(*) from truth_summary').fetchone()[0]
            truth_cats[healpix] = (file, n_rows)
    time = time_print(time, format_time='.2f', prefix=f'Counted in  ', postfix='; finished counting all truth catalogs')


# In[8]:


# Read/write the refcat
deg2rad = np.pi/180.

dc2_dst_refcat_path = f'{dc2_dst_path}ref_cats/gal_ref_cat/'
butler = Butler(dc2_dst_refcat_path)
butler_stars = Butler('/datasets/DC2/repoRun2.2i/')
if redo_refcat:
    time_start = timer()
    schema = Schema()
    overrides = {
        'ra': 'coord_ra',
        'dec': 'coord_dec',
    }
    flags = ['is_variable', 'is_pointsource']
    ignore = ['host_galaxy', 'is_variable', 'is_pointsource']
    flags_good = [flag for flag in flags if flag not in ignore]
    schema.addField('id', type="L", doc='DC2 id')
    for coord in ('ra', 'dec'):
        schema.addField(overrides[coord], type="Angle", doc=f'Sky {coord} position', units='rad')
    schema.addField('parent', type="L", doc='Parent id')
    for flag in flags:
        if flag not in ignore:
            schema.addField(flag, type="Flag", doc=f'Is source {flag}')
    schema.addField('redshift', type="D", doc='Redshift')
    for postfix, doc in (('', '(extincted)'), ('_noMW', '(unextincted)')):
        for band in bands_dc2:
            name = f'lsst_{band}{postfix}_flux'
            overrides[f'flux_{band}{postfix}'] = name
            schema.addField(name, type="D", doc=f'True LSST {band} flux {doc}', units='nJy')

    datasetConfig = ingestTask.DatasetConfig(format_version=1)
    indexer = IndexerRegistry[datasetConfig.indexer.name](datasetConfig.indexer.active)
    cat = SourceCatalog(schema)
    dataId = indexer.makeDataId('master_schema', datasetConfig.ref_dataset_name)
    ingestTask.addRefCatMetadata(cat)
    butler.put(cat, 'ref_cat', dataId=dataId)

    n_rows = np.sum([x[1] for x in truth_cats.values()])
    cat.resize(n_rows)
    sub = np.repeat(False, n_rows)
    row_begin = 0
    row_end = 0
    ras = np.zeros(n_rows)
    decs = np.zeros(n_rows)

    time = None
    for idx, (healpix, (file, n_rows)) in enumerate(truth_cats.items()):
        time = time_print(time, prefix=f'Assigned in ' if idx > 0 else 'Loading underway',
                          postfix=f';loading {healpix} ({idx+1}/{n_files}) nrows={n_rows} file={file}')
        with sqlite3.connect(file) as db: 
            truth = pd.read_sql_query("SELECT * from truth_summary", db)
        time = time_print(time, prefix=f'Loaded in ', postfix='; assigning underway')
        row_end += n_rows
        # It's easier to keep track of the coordinates for indices in arrays than to convert Angles
        ras[row_begin:row_end] = truth['ra']
        decs[row_begin:row_end] = truth['dec']
        # The output needs to be in radians
        truth['ra'] *= deg2rad
        truth['dec'] *= deg2rad
        sub[row_begin:row_end] = True
        columns = truth.columns 
        for source in truth.columns:
            if source not in ignore:
                name = overrides.get(source, source)
                cat[name][sub] = truth[source]
        if flags_good:
            assert(False, 'Flag setting is too slow; find a solution first')
            for i, row in enumerate(cat):
                row_src = truth.iloc[i]
                for flag in flags:
                    row[flag][sub] = row_src[flag]
        sub[row_begin:row_end] = False
        row_begin = row_end
    time = time_print(time, prefix=f'Assigned in ', postfix='; computing indices')
    
    indices = np.array(indexer.indexPoints(ras, decs))
    # Break up the pixels using a histogram
    h, rev = esutil.stat.histogram(indices, rev=True)
    time = time_print(time, prefix=f'Computed indices in ', postfix='; writing refcats')
    gd, = np.where(h > 0)
    for i in gd:
        within = rev[rev[i]: rev[i + 1]]
        sub[within] = True
        index_htm = indices[within[0]]
        # Write the individual pixel
        dataId = indexer.makeDataId(index_htm, datasetConfig.ref_dataset_name)
        cat_put = cat[sub]
        try:
            # Add stars to the catalog - they're already binned by the same htm pix
            cat_stars = butler_stars.get('ref_cat', dataId)
            # Only stars: the galaxies are wrong
            cat_stars = cat_stars[~cat_stars['resolved']]
            cat_extend = SourceCatalog(schema)
            cat_extend.resize(len(cat_stars))
            cat_extend['id'] = -cat_stars['id']
            for column in ('coord_ra', 'coord_dec'):
                cat_extend[column] = cat_stars[column]
            for band in bands_dc2:
                cat_extend[f'lsst_{band}_flux'] = cat_stars[f'lsst_{band}_smeared_flux']
            cat_put.extend(cat_extend)
        except Exception as e:
            print(f"Failed to find stars/extend ref_cat for index={index_htm} due to {e}")
        butler.put(cat_put, 'ref_cat', dataId=dataId)
        time = time_print(time, prefix=f'Wrote refcat {index_htm} in ')
        sub[within] = False
    
    print(f'Finished writing refcats in {time - time_start:.1f}s')
    # And save the dataset configuration
    dataId = indexer.makeDataId(None, datasetConfig.ref_dataset_name)
    butler.put(datasetConfig, 'ref_cat_config', dataId=dataId)


# ## Crossmatch against reference catalog
# 
# All of the plots in this notebook compare stack measurements of sources measured by the LSST Science Pipelines cross-matched against a reference catalog that is part of the Science Pipelines repository. This reference catalog contains all sources brighter than 23 mags in r, and was generated through some kind of Task that must have read the original DC2 truth tables (more on that below).

# In[9]:


# Load MultiProFit catalogs and concat them. Note 3828/9 = 2.2i, 3832 et al. = 2.1.1i
dc2_tracts = {
    3828: (f'{dc2_dst_path}2020-01-31/', '2.2i'),
    3832: (f'{dc2_dst_path}2020-01-31/', '2.1.1i'),
}
flux_match = f'lsst_{band_ref}'
filters_single = ('g', 'r', 'i')
filters_multi = ('gri',)
filters_all = filters_single + filters_multi
filters_order = [band_ref] + [band for band in filters_all if band != band_ref]
cats = {}
for tract, (path, run_dc2) in dc2_tracts.items():
    task = DirectMatchTask(butler, config=config)
    cats[tract] = {'meas': {}}
    matched_ids_src = {}
    schema_truth, truth_full = None, None
    for band in filters_order:
        print(f'Loading tract {tract} band {band}')
        files = np.sort(glob.glob(f'{path}{band}/mpf_dc2_{band}_{tract}_[0-9],[0-9]_mag.fits'))
        cat_full = None
        n_files = len(files)
        time = timer()
        for idx, file in enumerate(files):
            # This entire bit of aggravating code is a tedious way to get matched catalogs
            # in different bands all matched on the same reference band
            patch = file.split('_')[-2]
            matches = matched_ids_src.get(patch, None)
            has_match = matches is not None
            cat = SourceCatalog.readFits(file)
            if cat_full is None:
                assert(idx == 0)
                cat_full = SourceCatalog(cat.schema)
            if not has_match:
                assert(band == band_ref)
                matches = task.run(cat, filterName=flux_match)
                if truth_full is None:
                    schema_truth = matches.refCat.schema
                matches = matches.matches
            n_matches = len(matches)
            cat_full.reserve(n_matches)
            if has_match:
                n_good = 0
                for id_src in matches:
                    # See below - we saved the id of the src but sadly couldn't get the row index (I think)
                    src = cat.find(id_src)
                    cat_full.append(src)
                    good_src = np.isfinite(src[f'multiprofit_gausspx_c1_{band if band in filters_single else band_ref}_mag'])
                    n_good += good_src
            else:
                truth_patch = SourceCatalog(schema_truth)
                truth_patch.reserve(n_matches)
                match_ids = np.argsort([match.second.getId() for match in matches])
                matched_ids_src_patch = np.zeros(n_matches, dtype=cat_full['id'].dtype)
                # Loop through matches sorted by meas cat id
                # Add them to the full truth/meas cats
                # Save the id for other bands to find by
                # (If there were a way to find row index by id that would probably be better,
                # since it would only need to be done once in the ref_band)
                for idx_save, idx_match in enumerate(match_ids):
                    match = matches[idx_match]
                    matched_ids_src_patch[idx_save] = match.second.getId()
                    cat_full.append(match.second)
                    truth_patch.append(match.first)
                assert((idx_save + 1) == len(matched_ids_src_patch))
                matched_ids_src[patch] = matched_ids_src_patch
                cat_matched = cat_full[len(cat_full)-n_matches:]
                if truth_full is None:
                    assert(idx == 0)
                    truth_full = truth_patch
                else:
                    truth_full.extend(truth_patch)
            time = time_print(
                time, prefix=f'Loaded in ', postfix=f'; loading {patch} ({idx+1}/{n_files})'
                f'{" and matching" if not has_match else ""} file={file};'
                f' len(cat,truth)={len(cat_full) if cat_full is not None else 0},'
                f'{len(truth_full) if truth_full is not None else 0}'
            )
            cats[tract]['meas'][band] = cat_full.copy(deep=True)
        cats[tract]['truth'] = truth_full.copy(deep=True)


# In[10]:


# Model plot setup
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


# In[11]:


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
            mags_true_good = {}
            mags_model_good = {}
            cats_type = [(cats_meas, False)]
            if band_multi is not None and is_multiprofit:
                cats_type.append((cat_multi, True))
            mags_diff = {}
            for band in bands:
                mags_diff[band] = {}
                good_band = good_mags_true[band]
                true = mags_true[band]
                for cat, multi in cats_type:
                    y = get_total_mag((cat if multi else cat[band]), band, name_model, n_comps, is_multiprofit) - true
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

# In[12]:


# Galaxies
plot_matches(cats, True, models, bands=filters_single, band_ref=band_ref, band_ref_multi=band_ref,
             band_multi=filters_multi[0], mag_max=25, limy=lim_y['resolved'], **args)


# ## Point source fluxes and colours
# 
# There's not much to say here, other than that the stars look basically fine and the choice of PSF/source model and single- vs multi-band fitting makes little difference. Someone more versed in stellar photometry might have more to say about the small biases in the medians, outliers, etc.

# In[13]:


# Stars
plot_matches(cats, False, models_stars, bands=filters_single, band_ref=band_ref, band_ref_multi=band_ref,
             band_multi=filters_multi[0], mag_max=23, limy=lim_y['unresolved'], **args)

