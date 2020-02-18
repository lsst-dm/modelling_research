import astropy.coordinates as coord
import astropy.units as u
import esutil
import glob
import lsst.afw.table as afwTable
from lsst.daf.persistence import Butler
from lsst.geom import Box2D
from lsst.meas.algorithms import ingestIndexReferenceTask as ingestTask, IndexerRegistry
from lsst.meas.astrom import DirectMatchTask, DirectMatchConfig
import numpy as np
import pandas as pd
import sqlite3
from timeit import default_timer as timer
from .timing import time_print


def get_filters():
    return ['u', 'g', 'r', 'i', 'z', 'y']


def get_filter_ref():
    return 'r'


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


class Model:
    def get_total_mag(self, cat, band):
        return (
            cat[self.field] if not self.is_multiprofit else
            cat[f'{self.field}_c1_{band}_mag'] if self.n_comps == 1 else
            -2.5 * np.log10(np.sum([
                10 ** (-0.4 * cat[f'{self.field}_c{comp + 1}_{band}_mag'])
                for comp in range(self.n_comps)], axis=0
            ))
        )

    def __init__(self, desc, field, n_comps):
        self.desc = desc
        self.is_multiprofit = n_comps > 0
        self.n_comps = n_comps
        self.field = f'multiprofit_{field}' if self.is_multiprofit else field


def _get_bbox_corners(bbox, wcs):
    corners = wcs.pixelToSky(bbox.getCorners())
    ra = np.sort([x.getRa() for x in corners])
    dec = np.sort([x.getDec() for x in corners])
    return [np.mean(x) for x in (ra[0:2], ra[2:4], dec[0:2], dec[2:4])]


def _get_corner_cat(ra_min, ra_max, dec_min, dec_max, cat=None):
    if cat is None:
        cat = afwTable.SourceCatalog(afwTable.SourceTable.makeMinimalSchema())
        cat.resize(4)
    cat['coord_ra'][[0, 3]] = ra_min
    cat['coord_ra'][1:3] = ra_max
    cat['coord_dec'][0:2] = dec_min
    cat['coord_dec'][2:4] = dec_max
    return cat


def _get_refcat_bbox(task, bbox, wcs, filterName=None, cat_corner=None):
    """
    Get all of the objects from a reference catalog within a bounding box.

    Parameters
    ----------
    task : `lsst.meas.astrom.directMatch.DirectMatchTask`
        A matching task to load reference objects with.
    bbox : `lsst.geom.Box2D`
        A pixel bounding box; only objects within the box are returned.
    wcs : `lsst.afw.geom.SkyWcs`
        A WCS solution to transform the bbox.
    filterName : `str`
        The reference filter to load reference objects for.
    cat_corner : `lsst.afw.table.SourceCatalog`
        A SourceCatalog, the first four elements of which will be set to the coordinates of `bbox` and
        then used to select reference objects within a circumscribing circle.

    Returns
    -------

    """
    ra_min, ra_max, dec_min, dec_max = _get_bbox_corners(bbox, wcs)
    cat = _get_corner_cat(ra_min, ra_max, dec_min, dec_max, cat=cat_corner)
    circle = task.calculateCircle(cat)
    refcat = task.refObjLoader.loadSkyCircle(circle.center, circle.radius, filterName=filterName).refCat
    ra, dec = refcat['coord_ra'], refcat['coord_dec']
    return refcat[(ra > ra_min) & (ra < ra_max) & (dec > dec_min) & (dec < dec_max)]


def match_refcat(butler, tracts=None, butlers_dc2=None, config=None, filter_ref=None, match_afw=True):
    # Load MultiProFit catalogs and concat them. Note 3828/9 = 2.2i, 3832 et al. = 2.1.1i
    if tracts is None:
        truth_path = get_truth_path()
        tracts = {
            3828: (f'{truth_path}2020-01-31/', '2.2i'),
            3832: (f'{truth_path}2020-01-31/', '2.1.1i'),
        }
    if butlers_dc2 is None:
        butlers_dc2 = {
            '2.2i': Butler('/datasets/DC2/repoRun2.2i/rerun/w_2020_03/DM-22816/'),
            '2.1.1i': Butler('/datasets/DC2/repoRun2.1.1i/rerun/w_2019_34/'),
        }
    if not match_afw:
        skymaps = {version: butler_dc2.get('deepCoadd_skyMap') for version, butler_dc2 in butlers_dc2.items()}
    if config is None:
        config = DirectMatchConfig(matchRadius=0.5)
    if filter_ref is None:
        filter_ref = get_filter_ref()
    flux_match = f'lsst_{filter_ref}'
    filters_single = ('g', 'r', 'i')
    filters_multi = ('gri',)
    filters_all = filters_single + filters_multi
    filters_order = [filter_ref] + [band for band in filters_all if band != filter_ref]
    task = DirectMatchTask(butler, config=config)

    if not match_afw:
        items_extra = [f'{t}{idx}' for t in ('dists', 'indices') for idx in range(1, 3)]

    cats = {}
    for tract, (path, run_dc2) in tracts.items():
        cats[tract] = {'meas': {}}
        if not match_afw:
            skymap_tract = skymaps[run_dc2][tract]
            wcs = skymap_tract.getWcs()
            for item in items_extra:
                cats[tract][item] = []
        matched_ids_src = {}
        schema_truth, truth_full = None, None
        # Store a bool mask of the valid children per patch in the band used for matching and apply consistently
        # For some reason detect_isPatchInner isn't completely consistent in each filter
        is_primary = {}
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
                cat = afwTable.SourceCatalog.readFits(file)
                if cat_full is None:
                    assert (idx == 0)
                    cat_full = afwTable.SourceCatalog(cat.schema)
                if not has_match:
                    assert (band == filter_ref)
                    is_primary[patch] = \
                    butlers_dc2[run_dc2].get('deepCoadd_ref', {'tract': tract, 'patch': patch})[
                        'detect_isPrimary']
                cat = cat[is_primary[patch]]
                if not has_match:
                    if match_afw:
                        matches = task.run(cat, filterName=flux_match)
                        if truth_full is None:
                            schema_truth = matches.refCat.schema
                        matches = matches.matches
                    else:
                        bbox = Box2D(skymap_tract[tuple(int(x) for x in patch.split(','))].getInnerBBox())
                        truth_patch = _get_refcat_bbox(task, bbox, wcs, filterName=flux_match).copy(deep=True)
                        skyCoords = [
                            coord.SkyCoord(x['coord_ra'], x['coord_dec'], unit=u.rad)
                            for x in (truth_patch.asAstropy(), cat.copy(deep=True).asAstropy())
                        ]
                        n_truth, n_meas = len(truth_full) if truth_full is not None else 0, len(cat_full)
                        for refcat_first in (True, False):
                            postfix = 1 + (not refcat_first)
                            indices, dists, _ = coord.match_coordinates_sky(skyCoords[~refcat_first],
                                                                            skyCoords[refcat_first])
                            cats[tract][f'dists{postfix}'].append(dists.arcsec)
                            offset = n_meas if refcat_first else n_truth
                            if offset:
                                indices += offset
                            cats[tract][f'indices{postfix}'].append(indices)
                        matched_ids_src[patch] = True
                if match_afw:
                    n_matches = len(matches)
                    cat_full.reserve(n_matches)
                    if has_match:
                        n_good = 0
                        for id_src in matches:
                            # See below - we saved the id of the src but sadly couldn't get the row index (right?)
                            src = cat.find(id_src)
                            cat_full.append(src)
                            good_src = np.isfinite(
                                src[f'multiprofit_gausspx_c1_'
                                    f'{band if band in filters_single else filter_ref}_mag'])
                            n_good += good_src
                    else:
                        truth_patch = afwTable.SourceCatalog(schema_truth)
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
                        assert ((idx_save + 1) == len(matched_ids_src_patch))
                        matched_ids_src[patch] = matched_ids_src_patch
                else:
                    cat_full.extend(cat)
                if not has_match:
                    if truth_full is None:
                        assert (idx == 0)
                        truth_full = truth_patch
                    else:
                        truth_full.extend(truth_patch)
                time = time_print(
                    time, prefix=f'Loaded in ',
                    postfix=f'; loading {patch} ({idx + 1}/{n_files})'
                            f'{" and matching" if not has_match else ""} file={file};'
                            f' len(cat,truth)={len(cat_full) if cat_full is not None else 0},'
                            f'{len(truth_full) if truth_full is not None else 0}'
                )
            cats[tract]['meas'][band] = cat_full.copy(deep=True)
        cats[tract]['truth'] = truth_full.copy(deep=True)
        if not match_afw:
            for item in items_extra:
                cats[tract][item] = np.concatenate(cats[tract][item])
    return cats