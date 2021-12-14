#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import lsst.daf.butler as dafButler
import lsst.geom as geom

import logging
from modelling_research.match_prob import ProbabilisticMatcherConfig, ProbabilisticMatcher
import modelling_research.meas_model as mrMeas
from modelling_research.plot_matches import plot_matches
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq


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
weekly = 'w_2021_36'
ticket = 'DM-31699'

butler = dafButler.Butler(
    '/repo/dc2',
    collections=[
        '2.2i/truth_summary',
        f'2.2i/runs/test-med-1/{weekly}/{ticket}',
    ]
)

tracts = (3828, 3829)
band_ref = 'r'
mag_tot_min, mag_tot_max = 0., 27.
mag_zeropoint = 31.4
name_bands = 'ugrizy'
bands = tuple(x for x in name_bands)
model_target = 'CModel'
#masks_bad = ['BAD', 'EDGE', 'SAT', 'NO_DATA']

use_fluxes = True

columns_fluxes_target = [f'{band}{model_target}Flux' for band in bands]
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
    columns_all_target.extend([f'{band}PsFlux{suffix}' for band in bands for suffix in ('', 'Err')])

# We only measure chi^2 from x, y because we don't have ra, dec errors (yet)
columns_data_ref[:2] = 'x', 'y'

match_dist_max = geom.Angle(0.5, geom.arcseconds)
# Two coordinates and one band
n_finite_min = 2 + use_fluxes

skymap = butler.get('skyMap')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('match')


# In[4]:


# Re-match or use previous?
rerun = False
write = False


# In[5]:


# Load and match and save
def get_xy(ra, dec, wcs):
    skymap_tract = skymap[tract]
    wcs = skymap_tract.getWcs()
    ra_ref = cat_ref['ra'].copy(deep=True)
    dec_ref = cat_ref['dec'].copy(deep=True)
    radec_true = [geom.SpherePoint(ra, dec, geom.degrees) for ra, dec in zip(ra_ref, dec_ref)]
    xy_true = wcs.skyToPixel(radec_true)
    return np.array([xy[0] for xy in xy_true]), np.array([xy[1] for xy in xy_true])

def get_filename_match(name_model, name_cat, bands, tract, postfix=''):
    return f'match_{model_target}_{name_cat}_{bands}_{tract}{postfix}.parq'

path_match = f'/project/dtaranu/dc2_gen3/match/{weekly}'
if not os.path.exists(path_match):
    os.mkdir(path_match)
    
cats_in, cats_out = {}, {}

for tract in tracts:
    cat_target, cat_ref = (
        butler.get(
            dataset,
            tract=tract,
            parameters={"columns": columns},
        )
        for dataset, columns in (
            ('objectTable_tract', columns_all_target),
            ('truth_summary', columns_all_ref)
        )
    )
    cats_in[tract] = (cat_ref, cat_target)

    if rerun or plot:
        cat_ref['x'], cat_ref['y'] = get_xy(cat_ref['ra'], cat_ref['dec'], skymap[tract].getWcs())

postfix = '' if use_fluxes else '_nofluxes'
        
if rerun:
    for tract in tracts:
        cat_ref, cat_target = cats_in[tract]
        wcs = skymap[tract].getWcs()
        flux_tot = np.nansum(cat_ref.loc[:,[f'flux_{band}' for band in bands]].values, axis=1)
        cat_ref['flux'] = flux_tot
        mag_tot = -2.5*np.log10(flux_tot) + mag_zeropoint

        radec_obj = wcs.pixelToSky([geom.Point2D(x, y) for x, y in zip(cat_target['x'], cat_target['y'])])
        cat_target['ra'] = np.array([radec[0].asDegrees() for radec in radec_obj])
        cat_target['dec'] = np.array([radec[1].asDegrees() for radec in radec_obj])

        config = ProbabilisticMatcherConfig(
            column_order = 'flux',
            column_ref_coord1 = 'ra',
            column_ref_coord2 = 'dec',
            column_target_coord1 = 'ra',
            column_target_coord2 = 'dec',
            columns_ref_meas = columns_data_ref,
            columns_target_meas = columns_data_target,
            columns_target_err = columns_errors_target,
            logging_n_rows = 50000,
            match_dist_max = match_dist_max.asDegrees(),
            match_n_finite_min = n_finite_min,
        )
        matcher = ProbabilisticMatcher(config)
        catalog_out_ref, catalog_out_target, exceptions = matcher.match(
            cat_ref,
            cat_target,
            select_ref=(mag_tot < mag_tot_max) & (mag_tot > mag_tot_min),
            select_target=(~cat_target['merge_peak_sky'] & cat_target['detect_isPrimary']).values,
            logger=logger,
        )

        for band in bands:
            exptime = butler.get("deepCoadd_exposure_time_map_sum", tract=tract, band=band)
            catalog_out_ref[f'exposure_time_{band}'] = exptime.get_values_pos(cat_ref['ra'].values, cat_ref['dec'].values)

        cats_out[tract] = (catalog_out_ref, catalog_out_target)
        if write:
            pq.write_table(pa.Table.from_pandas(catalog_out_ref),
                           f'{path_match}{get_filename_match(name_model=model_target, name_cat="ref", bands=name_bands, tract=tract, postfix=postfix)}')
            pq.write_table(pa.Table.from_pandas(catalog_out_target),
                           f'{path_match}{get_filename_match(name_model=model_target, name_cat="target", bands=name_bands, tract=tract, postfix=postfix)}')
else:
    for tract in tracts:
        cats_out[tract] = (
            pq.read_table(f'{path_match}{get_filename_match(name_model=model_target, name_cat="ref", bands=name_bands, tract=tract, postfix=postfix)}').to_pandas(),
            pq.read_table(f'{path_match}{get_filename_match(name_model=model_target, name_cat="target", bands=name_bands, tract=tract, postfix=postfix)}').to_pandas(),
        )


# In[6]:


# Concat tracts
import pandas as pd

# Keep track of row ranges for each ref cat
cat_ref_all = [cat for cat, _ in cats_out.values()]
n_rows_refs = [len(cat) for cat in cat_ref_all]
cat_ref_all = pd.concat(cat_ref_all, axis=0, ignore_index=True)

# ... and row ranges for target cats
cat_target_all = [cat for _, cat in cats_in.values()]
n_rows_targets = [len(cat) for cat in cat_target_all]
cat_target_all = pd.concat(cat_target_all, axis=0, ignore_index=True)

# Increment indices to point to the right index in the merged table
for n_rows_ref, n_rows_target in zip(n_rows_refs[:-1], n_rows_targets[:-1]):
    cat_ref_all.loc[n_rows_ref:, 'match_row'] += n_rows_target * (cat_ref_all.iloc[n_rows_ref:]['match_row'] >= 0)

idx_match_count = np.bincount(cat_ref_all['match_row'][cat_ref_all['match_row'] >= 0])

# Merge and concat target cat (containing row indices) with full inputs
cat_ref_in = pd.concat([cat for cat, _ in cats_in.values()], axis=0, ignore_index=True)
cat_ref_all = pd.concat([cat_ref_all, cat_ref_in], axis=1)

select_target=(~cat_target_all['merge_peak_sky'] & cat_target_all['detect_isPrimary']).values,

n_matched = np.sum(cat_ref_all['match_row'] >= 0)
n_matched_uniq = len(set(cat_ref_all['match_row']))

# There's always a min_int sentinel, unless everything was matched (that would be strange!)
if n_matched_uniq != n_matched + 1:
    raise RuntimeError(f'n_matched={n_matched} != n_matched_uniq={n_matched_uniq}')
print(f'Matched {n_matched}/{len(cat_ref_all)}')


# In[7]:


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

mag_psf_ref = -2.5*np.log10(cats_in[tract][1][f'{band_ref}PsFlux'].values) + mag_zeropoint
matched = np.zeros(len(cats_in[tract][1]), dtype=bool)
matched[cats_out[tract][0]['match_row'].values[cats_out[tract][0]['match_row'] >= 0]] = True

for idx in range(n_bins):
    within = (mag_psf_ref > bins[idx]) & (mag_psf_ref < bins[idx+1]) & cats_in[tract][1]['detect_isPrimary'] & ~cats_in[tract][1]['merge_peak_sky']
    n_obj[idx] = np.sum(within)
    n_matched[idx] = np.sum(matched[within])

plt.figure()
plt.plot(bins[:-1], n_matched/n_obj)


# In[8]:


# Plot matches nicely
models = {
    desc: mrMeas.Model(
        desc, field, n_comps,
        column_band_separator='',
        column_flux='Flux',
        column_separator='',
        column_band_prefixed=True,
        prefix_centroid_default='',
    )
    for desc, field, n_comps in [
        ('PSF', 'Ps', 1),
        ('Stack CModel', 'CModel', 2),
    ]
}

select_target = ~cat_target_all['merge_peak_sky'].values & cat_target_all['detect_isPrimary'].values
select_ref = cat_ref_all['match_candidate']

kwargs = {
    'scatterleft': True,
    'scatterright': True,
    'fluxes_true': {band: cat_ref_all.loc[select_ref, f'flux_{band}'].values for band in bands},
    'centroids_true': {'x': 'x', 'y': 'y'},
    'limx': (mag_plot_min, mag_tot_max - 1),
    'limits_y_chi': (-10., 10.),
    'limits_y_dist': (0., 2.5),
    'match_dist_asec': match_dist_max.asArcseconds(),
    'mag_bin_complete': 0.125,
    'mag_ref_zeropoint': mag_zeropoint,
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


# In[ ]:




