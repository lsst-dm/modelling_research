# This file is part of modelling_research.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import esutil
import glob
import lsst.afw.table as afwTable
from lsst.daf.persistence import Butler
from lsst.meas.algorithms import ingestIndexReferenceTask as ingestTask, IndexerRegistry
from .match_refcat import match_refcat
import numpy as np
import pandas as pd
import sqlite3
from .timing import time_print


def get_butlers():
    return {version: Butler(path) for version, path in get_repos().items()}


def get_filters():
    return ['u', 'g', 'r', 'i', 'z', 'y']


def get_filter_ref():
    return 'r'


def get_path_cats(prefix, band, tract, patches_regex=None):
    if patches_regex is None:
        patches_regex = "[0-7],[0-7]"
    path = f'{prefix}{band}/mpf_dc2_{band}_{tract}_{patches_regex}_mag.fits'
    files = glob.glob(path)
    print(f'Loading {len(files)} files from path={path}')
    return np.sort(files)


def get_refcat(truth_path=None, refcat_path=None, truth_summary_path=None, make=False):
    if truth_path is None:
        truth_path = get_truth_path()
    if refcat_path is None:
        refcat_path = f'{truth_path}ref_cats/gal_ref_cat/'
    butler = Butler(refcat_path)
    if make:
        if truth_summary_path is None:
            truth_summary_path = f'{truth_path}truth_cats/truth_summary_hp*.sqlite3'
        files = glob.glob(truth_summary_path)
        make_refcat(butler, files)
    return butler


def get_slurm_patches():
    return [
        [f'{x},{y}' for x in range(3) for y in range(7)] + [f'3,{y}' for y in range(3)],
        [f'3,{y}' for y in range(4, 7)] + [f'{x},{y}' for x in range(4, 7) for y in range(7)],
    ]


def get_repos():
    return {
        '2.1.1i': '/datasets/DC2/repoRun2.1.1i/rerun/w_2019_34/',
        '2.2i': '/datasets/DC2/repoRun2.2i/rerun/w_2020_03/DM-22816/'
    }


def get_tracts():
    truth_path = get_truth_path()
    tracts = {
        3828: (f'{truth_path}2020-01-31/', '2.2i'),
        3832: (f'{truth_path}2020-01-31/', '2.1.1i'),
    }
    return tracts


def get_truth_path():
    return '/project/dtaranu/dc2/'


def make_refcat(butler, files, filters=None, butler_stars=None):
    if filters is None:
        filters = get_filters()
    if butler_stars is None:
        butler_stars = Butler('/datasets/DC2/repoRun2.2i/')
    time = None
    truth_cats = {}
    n_files = len(files)
    for idx, file in enumerate(files):
        with sqlite3.connect(file) as db:
            healpix = file.split('truth_summary_hp')[1].split('.sqlite3')[0]
            prefix = f'Counted in  ' if idx > 0 else 'Counting starting'
            time = time_print(time, format_time='.2f', prefix=prefix,
                              postfix=f'; counting {healpix} ({idx+1}/{n_files}) file={file}')
            cursor = db.cursor()
            n_rows = cursor.execute('SELECT COUNT(*) from truth_summary').fetchone()[0]
            truth_cats[healpix] = (file, n_rows)
    time_start = time_print(time, format_time='.2f', prefix=f'Counted in  ',
                            postfix='; finished counting all truth catalogs')
    schema = afwTable.Schema()
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
        for band in filters:
            name = f'lsst_{band}{postfix}_flux'
            overrides[f'flux_{band}{postfix}'] = name
            schema.addField(name, type="D", doc=f'True LSST {band} flux {doc}', units='nJy')

    datasetConfig = ingestTask.DatasetConfig(format_version=1)
    indexer = IndexerRegistry[datasetConfig.indexer.name](datasetConfig.indexer.active)
    cat = afwTable.SourceCatalog(schema)
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
    deg2rad = np.deg2rad(1.)
    for idx, (healpix, (file, n_rows)) in enumerate(truth_cats.items()):
        time = time_print(time, prefix=f'Assigned in ' if idx > 0 else 'Loading underway',
                          postfix=f';loading {healpix} ({idx + 1}/{n_files}) nrows={n_rows} file={file}')
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
        for source in truth.columns:
            if source not in ignore:
                name = overrides.get(source, source)
                cat[name][sub] = truth[source]
        if flags_good:
            assert (False, 'Flag setting is too slow; find a solution first')
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
            cat_extend = afwTable.SourceCatalog(schema)
            cat_extend.resize(len(cat_stars))
            cat_extend['id'] = -cat_stars['id']
            for column in ('coord_ra', 'coord_dec'):
                cat_extend[column] = cat_stars[column]
            for band in filters:
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


def match_refcat_dc2(
        butler_refcat, kwargs_get, tracts=None, butlers_dc2=None, filter_ref=None, match_afw=True, filters_single=None,
        filters_multi=None, func_path=None, **kwargs
):
    """Load DC2 catalogs and Match catalogs to a reference catalog.

    Parameters
    ----------
    butler_refcat : `lsst.daf.persistence.Butler`
        A butler with a reference catalog.
    kwargs_get : `dict`
        Keyword arguments for butler.get calls to pass to `match_refcat`.
    tracts : iterable [`int`]
        A list of tract numbers.
    butlers_dc2 : `dict` [`str`, `lsst.daf.persistence.Butler`]
        A dict of butlers keyed by DC2 run name.
    filter_ref : `str`
        The name of the reference filter to match on.
    match_afw : `bool`
        Whether to use afw's DirectMatchTask to match or not
    filters_single : iterable [`str`]
        A list of filters to load single-band catalogs for. Default ('g', 'r', 'i').
    filters_multi : iterable [`str`]
        A list of names to load multi-band catalogs for. Default empty.
    func_path : callable
        A function that takes `prefix_file_path`, filter name and tract number as arguments and returns
        filenames of catalogs in that path. See `get_path_cats` for an example.
    kwargs : additional arguments passed to `match_refcat`.

    Returns
    -------
    See `modelling_research.match_refcat.match_refcat` for return value.

    Notes
    -----
    Many of the input arguments are passed unchanged to `modelling_research.match_refcat.match_refcat` and are listed 
    """
    # Load MultiProFit catalogs and concat them. Note 3828/9 = 2.2i, 3832 et al. = 2.1.1i
    if tracts is None:
        tracts = get_tracts()
    if butlers_dc2 is None:
        butlers_dc2 = get_butlers()
    if filter_ref is None:
        filter_ref = get_filter_ref()
    if filters_single is None:
        filters_single = ('g', 'r', 'i', 'z')
    if filters_multi is None:
        filters_multi = ('griz',)
    if func_path is None:
        func_path = get_path_cats
    if kwargs_get is None:
        kwargs_get = {}

    cats = {}
    for tract, (path, run_dc2) in tracts.items():
        cats[tract] = match_refcat(
            butler_refcat, butlers_dc2[run_dc2], [tract], filter_ref, func_path, kwargs_get=kwargs_get,
            match_afw=match_afw, prefix_flux_match='lsst_', prefix_file_path=path,
            filters_single=filters_single, filters_multi=filters_multi, **kwargs
        )[tract]

    return cats
