#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import lsst.daf.butler as dafButler
import lsst.geom as geom

import modelling_research.meas_model as mrMeas
from modelling_research.plot_matches import plot_matches
import numpy as np
import pandas as pd
import sys


# In[2]:


# Setup for plotting
plot = True
if plot:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    get_ipython().run_line_magic('matplotlib', 'inline')
    sns.set_style('darkgrid')
    mpl.rcParams['figure.dpi'] = 160
    mpl.rcParams['image.origin'] = 'lower'
    sns.set(rc={'axes.facecolor': '0.85', 'figure.facecolor': 'w'})


# In[3]:


# Load data and setup column lists
weekly = 'w_2022_26'
ticket = 'DM-35344'

butler = dafButler.Butler(
    '/repo/dc2',
    collections=[
        '2.2i/truth_summary',
        f'2.2i/runs/test-med-1/{weekly}/{ticket}',
    ],
    skymap="DC2",
)

tracts = (3828, 3829)
band_ref = 'r'
mag_tot_min, mag_tot_max = 0., 27.
mag_zeropoint = 31.4
name_bands = 'ugrizy'
bands = tuple(x for x in name_bands)
model_target = 'cModel'

use_fluxes = True

columns_fluxes_target = [f'{band}_{model_target}Flux' for band in bands]
columns_data_target = ['x', 'y']
if use_fluxes:
    columns_data_target.extend(columns_fluxes_target)
columns_errors_target = [f'{col}Err' for col in columns_data_target]
columns_all_target = ['objectId', 'merge_peak_sky', 'detect_isPrimary', 'patch']
columns_all_target.extend(columns_data_target)
columns_all_target.extend(columns_errors_target)
if not use_fluxes:
    columns_all_target.extend(columns_all_target)

columns_fluxes_ref = [f'flux_{band}' for band in bands]
columns_data_ref = ['ra', 'dec']
if use_fluxes:
    columns_data_ref += columns_fluxes_ref
columns_all_ref = ['id']
columns_all_ref.extend(columns_data_ref)
columns_all_ref.extend(columns_fluxes_ref)

if plot:
    columns_all_ref.append('is_pointsource')
    columns_all_target.append('refExtendedness')
    columns_all_target.extend([f'{band}_psfFlux{suffix}' for band in bands for suffix in ('', 'Err')])

# We only measure chi^2 from x, y because we don't have ra, dec errors (yet)
columns_data_ref[:2] = 'x', 'y'

match_dist_max = geom.Angle(0.5, geom.arcseconds)
# Two coordinates and one band
n_finite_min = 2 + use_fluxes

print(f'n_columns_ref: {len(columns_all_ref)}, n_columns_target: {len(columns_all_target)}')


# In[4]:


# Load matched catalogs
def get_xy(ras, decs, wcs):
    radec_true = [geom.SpherePoint(ra, dec, geom.degrees) for ra, dec in zip(ras, decs)]
    xy_true = wcs.skyToPixel(radec_true)
    return np.array([xy[0] for xy in xy_true]), np.array([xy[1] for xy in xy_true])

name_skymap = 'DC2'
skymap = butler.get('skyMap', name=name_skymap)

cats_in, cats_out = {}, {}

columns_match_ref = ('match_row', 'match_candidate', 'match_chisq', 'match_n_chisq_finite')

n_target = 0

for tract in tracts:
    print(f'Getting tract {tract}')
    cat_target, cat_ref, cat_match_ref = (
        butler.get(
            dataset,
            tract=tract,
            skymap=name_skymap,
            parameters={"columns": columns},
        )
        for dataset, columns in (
            ('objectTable_tract', columns_all_target),
            ('truth_summary', columns_all_ref),
            ('match_ref_truth_summary_objectTable_tract', columns_match_ref),
        )
    )
    x, y = get_xy(cat_ref['ra'], cat_ref['dec'], wcs=skymap[tract].getWcs())
    cat_ref['x'], cat_ref['y'] = x, y
    for column in columns_match_ref:
        cat_ref[column] = cat_match_ref[column]
    
    if n_target > 0:
        cat_ref.loc[cat_ref['match_row'] >= 0, 'match_row'] += n_target
        cat_ref_all = pd.concat([cat_ref_all, cat_ref], axis=0, ignore_index=True)
        cat_target_all = pd.concat([cat_target_all, cat_target], axis=0, ignore_index=True)
    else:
        cat_ref_all = cat_ref
        cat_target_all = cat_target
        
    n_target += len(cat_target)
    
del cat_target, cat_ref, cat_match_ref, x, y
match_max = np.max(cat_ref_all['match_row'])
if match_max >= len(cat_target_all):
    raise RuntimeError(f'match_max={match_max} >= len(cat_target_all)={len(cat_target_all)}')


# In[5]:


# Concat tracts

select_target=(~cat_target_all['merge_peak_sky'] & cat_target_all['detect_isPrimary']).values,

n_matched = np.sum(cat_ref_all['match_row'] >= 0)
n_matched_uniq = len(set(cat_ref_all['match_row']))

# There's always a min_int sentinel, unless everything was matched (that would be strange!)
if n_matched_uniq != n_matched + 1:
    print(np.where(np.unique(cat_ref_all['match_row'], return_counts=True)[1] > 1)[0])
    raise RuntimeError(f'n_matched={n_matched} != n_matched_uniq={n_matched_uniq}')
print(f'Matched {n_matched}/{len(cat_ref_all)}')


# In[6]:


# Plot basics
import matplotlib.pyplot as plt
chisq = cat_ref_all['match_chisq']
n_match = cat_ref_all['match_n_chisq_finite']
mag_ref = -2.5*np.log10(cat_ref_all[f'flux_{band_ref}']) + mag_zeropoint

_ = plt.hist(np.clip(np.log10((chisq[chisq > 0]/n_match[chisq > 0])), -1.5, 2.5), bins=100)

mag_plot_min, mag_plot_max = 15, mag_tot_max + 0.5
n_bins = int(np.round(10*(mag_plot_max - mag_plot_min)))
bins = np.linspace(mag_plot_min, mag_plot_max, num=n_bins + 1)
n_obj = np.zeros(n_bins, dtype=int)
n_matched = np.zeros(n_bins, dtype=int)

for idx in range(n_bins):
    within = (mag_ref > bins[idx]) & (mag_ref < bins[idx+1])
    n_obj[idx] = np.sum(within)
    n_matched[idx] = np.sum(chisq[within] > 0)

plt.figure()
plt.plot(bins[:-1], n_matched/n_obj)

del chisq, n_match, mag_ref

mag_psf_ref = -2.5*np.log10(cat_target_all[f'{band_ref}_psfFlux'].values) + mag_zeropoint
matched = np.zeros(len(cat_target_all), dtype=bool)
matched[cat_ref_all['match_row'].values[cat_ref_all['match_row'] >= 0]] = True
is_primary = cat_target_all['detect_isPrimary'] & ~cat_target_all['merge_peak_sky']

for idx in range(n_bins):
    within = (mag_psf_ref > bins[idx]) & (mag_psf_ref < bins[idx+1]) & is_primary
    n_obj[idx] = np.sum(within)
    n_matched[idx] = np.sum(matched[within])

del matched, within, mag_psf_ref, is_primary
    
plt.figure()
plt.plot(bins[:-1], n_matched/n_obj)


# In[7]:


# Check memory usage

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
memusg = list(sorted(
    [
        (x, sys.getsizeof(globals().get(x))) for x in dir()
        if not x.startswith('_') and x not in sys.modules and x not in ipython_vars
    ],
    key=lambda x: x[1], reverse=True
))
print(memusg)


# In[8]:


# Plot matches nicely
models = {
    desc: mrMeas.Model(
        desc, field, n_comps,
        column_flux='Flux',
        column_separator='',
        column_band_prefixed=True,
        prefix_centroid_default='',
    )
    for desc, field, n_comps in [
        ('PSF', 'psf', 1),
        ('Stack CModel', 'cModel', 2),
    ]
}

select_target = ~cat_target_all['merge_peak_sky'].values & cat_target_all['detect_isPrimary'].values
select_ref = cat_ref_all['match_candidate']

kwargs = {
    'scatterleft': True,
    'scatterright': True,
    'fluxes_true': {band: cat_ref_all.loc[select_ref, f'flux_{band}'].values for band in bands},
    'centroids_ref': {'x': 'x', 'y': 'y'},
    'limx': (mag_plot_min, mag_tot_max - 1),
    'limits_y_chi': (-10., 10.),
    'limits_y_dist': (0., 2.5),
    'match_dist_asec': match_dist_max.asArcseconds(),
    'mag_bin_complete': 0.125,
    'mag_zeropoint_ref': mag_zeropoint,
    'mag_max': mag_plot_max,
    'models': models,
    'models_purity': ('PSF', 'Stack CModel'),
    'models_diff': ('Stack CModel',),
    'models_dist': ('Stack CModel',),
    'select_target': select_target,
    'title': f'DC2 {",".join(str(tract) for tract in tracts) } {model_target}',
    'compare_mags_psf_lim': (-2.45, 0.05),
    'kwargs_get_mag': {'zeropoint': mag_zeropoint},
}

lim_y = {
    "resolved": (-0.6, 0.4),
    "unresolved": (-0.1, 0.05),
}

_ = plot_matches(
    cat_ref_all[select_ref],
    cat_target_all,
    resolved=True,
    plot_chi=True,
    limits_y_diff=lim_y['resolved'],
    limits_y_color_diff=lim_y['resolved'],
    **kwargs
)


# In[9]:


# Plot probable point sources
_ = plot_matches(
    cat_ref_all[select_ref],
    cat_target_all,
    resolved=False,
    plot_chi=True,
    limits_y_diff=lim_y['unresolved'],
    limits_y_color_diff=lim_y['unresolved'],
    **kwargs
)


# In[10]:


# Check memory usage

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
memusg = list(sorted(
    [
        (x, sys.getsizeof(globals().get(x))) for x in dir()
        if not x.startswith('_') and x not in sys.modules and x not in ipython_vars
    ],
    key=lambda x: x[1], reverse=True
))
print(memusg)


# In[ ]:




