import astropy.coordinates as coord
import astropy.units as u
from .tables import read_split_cat_fits
import lsst.afw.table as afwTable
from lsst.geom import Box2D
from lsst.meas.astrom import DirectMatchTask, DirectMatchConfig
import numpy as np
from timeit import default_timer as timer
from .timing import time_print


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


def match_refcat(
        butler_refcat, butler_data, tracts, filter_ref, func_path, match_afw=True, skymap=None,
        prefix_flux_match=None, prefix_file_path=None, filters_single=None, filters_multi=None, config=None,
        *args, **kwargs
):
    """Match catalogs to a reference catalog.

    Parameters
    ----------
    butler_refcat : `lsst.daf.persistence.Butler`
        A butler with a reference catalog.
    butler_data : `lsst.daf.persistence.Butler`
        A butler to get a skymap and deepCoadd_ref source catalog from.
    tracts : iterable [`int`]
        A list of tract numbers.
    filter_ref : `str`
        A reference filter to match on.
    func_path : callable
        A function that takes `prefix_file_path`, filter name and tract number as arguments and returns
        filenames of catalogs in that path. See `modelling_research.dc2.get_path_cats` for an example.
    match_afw : `bool`
        Whether to match using `lsst.meas.astrom.DirectMatchTask`; otherwise,
        `astropy.coordinates.match_coordinates_sky` is used.
    skymap : `lsst.skymap.BaseSkyMap`
        A skymap to use for matching with astropy. Defaults to `butler_data.get('deepCoadd_skyMap')`.
    prefix_flux_match : `str`
        A field name prefix for truth fluxes. Default 'lsst_'.
    prefix_file_path : `str`
        A file path prefix to pass to `func_path`. Default empty.
    filters_single : iterable [`str`]
        A list of filters to load single-band catalogs for. Default ('g', 'r', 'i').
    filters_multi : iterable [`str`]
        A list of names to load multi-band catalogs for. Default empty.
    config : `lsst.meas.astrom.DirectMatchConfig`
        Configuration for `lsst.meas.astrom.DirectMatchTask`. Ignored if not `match_afw`.

    Returns
    -------
    cats : `dict`
        A dict of results by tract, each with the following entries:

        ``dists1``, ``dists2``
            Distances in arcsecs from each true source to the nearest measured source (1),
            and vice versa (2).
        ``indices1``, ``indice2``
            Indices (row numbers) of the measured/true (1, 2) matched source.
        ``meas``
            `dict` [`str`, `lsst.afw.table.SourceCatalog`] of measured sources in each filter.
        ``truth``
            `lsst.afw.table.SourceCatalog` of reference sources.
    Notes
    -----
    TODO: The default behaviour should be to match on deepCoadd_ref, not a single reference filter.

    `match_afw` only returns distances within the specified match radius and does not compute dists2 or
    indices2.

    """
    if not match_afw and skymap is None:
        skymap = butler_data.get('deepCoadd_skyMap')
    if config is None:
        config = DirectMatchConfig(matchRadius=0.5)
    if filters_single is None:
        filters_single = ('g', 'r', 'i')
    if filters_multi is None:
        filters_multi = ()
    if prefix_flux_match is None:
        prefix_flux_match = 'lsst_'
    if prefix_file_path is None:
        prefix_file_path = ''

    flux_match = f'{prefix_flux_match}{filter_ref}'
    filters_all = filters_single + filters_multi
    filters_order = [filter_ref] + [band for band in filters_all if band != filter_ref]
    task = DirectMatchTask(butler_refcat, config=config)

    if not match_afw:
        items_extra = [f'{t}{idx}' for t in ('dists', 'indices') for idx in range(1, 3)]

    cats = {}
    for tract in tracts:
        cats[tract] = {'meas': {}}
        if not match_afw:
            skymap_tract = skymap[tract]
            wcs = skymap_tract.getWcs()
            for item in items_extra:
                cats[tract][item] = []
        matched_ids_src = {}
        schema_truth, truth_full = None, None
        # Store a bool mask of the valid children per patch in the band used for matching; apply consistently
        # For some reason detect_isPatchInner isn't completely consistent in each filter
        is_primary = {}
        for band in filters_order:
            print(f'Loading tract {tract} band {band}')
            files = func_path(prefix_file_path, band, tract, *args, **kwargs)
            cat_full = None
            n_files = len(files)

            time = timer()
            for idx, file in enumerate(files):
                # This entire bit of aggravating code is a tedious way to get matched catalogs
                # in different bands all matched on the same reference band
                patch = file.split('_')[-2]
                matches = matched_ids_src.get(patch, None)
                has_match = matches is not None
                cat = read_split_cat_fits(file)
                if cat_full is None:
                    assert (idx == 0)
                    cat_full = afwTable.SourceCatalog(cat.schema)
                    colnames_full = [x.field.getName() for x in cat_full.schema]
                    units = [x.field.getUnits() for x in cat_full.schema]
                    set_colnames_full = set(colnames_full)
                else:
                    colnames = [x.field.getName() for x in cat.schema]
                    matched_schema = {
                        'names': colnames == colnames_full,
                        'units': units == [x.field.getUnits() for x in cat_full.schema],
                    }
                    if not all(matched_schema.values()):
                        matched_fails = ','.join(k for k, v in matched_schema.items() if not v)
                        set_matched = set(colnames) == set_colnames_full
                        raise RuntimeError(
                            f"File {file} column {matched_fails} don't match original {files[0]}; "
                            f"set matched={set_matched} but ")

                if not has_match:
                    assert (band == filter_ref)
                    is_primary[patch] = butler_data.get(
                        'deepCoadd_ref', {'tract': tract, 'patch': patch}
                    )['detect_isPrimary']
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
                            # See below - we saved the id of the src
                            # but sadly couldn't get the row index (right?)
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