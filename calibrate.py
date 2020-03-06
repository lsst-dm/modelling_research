from . import dc2
from lsst.afw.table import Schema, SourceCatalog
from lsst.daf.persistence import Butler
from .timing import time_print


def calibrate_catalog(catalog, photoCalibs_filter, filter_ref=None):
    """Calibrate a catalog with filter-dependent instFlux columns.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        A SourceCatalog with instFlux columns.
    photoCalibs_filter : `dict` [`str`, `lsst.afw.image.PhotoCalib`]
        A dict of PhotoCalibs by filter.
    filter_ref : `str`
        The filter to use for non-multiband fluxes; default first key in list(photoCalibs_filter).

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        A SourceCatalog with instFlux columns calibrated to mag columns.

    Notes
    -------
    The main purpose of this method is to calibrate a catalog with multi-band MultiProFit columns as
    generated by this task; however, it will work on any catalog as long as the columns match one of
    [*fit_*instFlux, *Flux_instFlux], and the filter-dependent columns are named *{filter}_instFlux.
    """
    filters = photoCalibs_filter.keys()
    if filter_ref is None:
        filter_ref = list(filters)[0]
    # See DM-23766: catalog.schema.getNames() returns a set (unordered) whereas we want a list
    columns = [x.field.getName() for x in catalog.schema]
    fluxes = ['_'.join(name.split('_')[0:-1]) for name in columns if
              ('fit_' in name and name.endswith('_instFlux')) or name.endswith('Flux_instFlux')
              or name.endswith('SdssShape_instFlux')]
    fluxes_filter = {band: [] for band in filters}
    for flux in fluxes:
        last = flux.split('_')[-1]
        fluxes_filter[last if last in filters else filter_ref].append(flux)
    for band, photoCalib in photoCalibs_filter.items():
        catalog = photoCalib.calibrateCatalog(catalog, fluxes_filter[band])
    return catalog


def calibrate_catalogs(files, butler, func_dataId=None, is_dc2=False, return_cat=False, write=True,
                       postfix='_mag.fits'):
    """Calibrate FITS source catalogs from know repos.

    Parameters
    ----------
    files : iterable of `str`
        An iterable of paths to catalogs to calibrate.
    butler : `lsst.daf.persistence.Butler` or container thereof
        A `dict` of butlers if `dc2`, else a single butler to obtain photoCalibs from.
    func_dataId : callable
        A function that takes a catalog filename sans FITS extension and returns bands, patch, tract.
    is_dc2 : `bool`
        Whether this is a DC2 simulation repo on lsst-dev.
    return_cat : `bool`
        Whether to return the calibrate catalogs; they might be very large so the default is False.
    write : `bool`
        Whether to write the calibrated catalogs to disk, postfixed by `postfix.
    postfix : `str`
        The postfix to add to filenames when writing calibrate catalogs; default '_mag.fits'.

    Returns
    -------
    cats : `list` [`lsst.afw.table.SourceCatalog`]
        Calibrate catalogs for each file if `return_cat`.

    """
    if func_dataId is None:
        func_dataId = parse_multiprofit_dataId
    if is_dc2:
        tracts_dc2 = dc2.get_tracts()
        repos = dc2.get_repos()
        if butler is None:
            butler = {}

    time = None
    cats = []

    for file in files:
        time = time_print(time, prefix=f'Calibrated in  ' if time is not None else 'Calibrating starting',
                          postfix=f', now on {file}')
        cat = SourceCatalog.readFits(file)
        filename = file.split('.fits')[0]
        bands, tract, patch = func_dataId(filename)

        tract = int(tract)
        if is_dc2:
            version = tracts_dc2[tract][1]
            if version not in butler:
                path = repos[version]
                print(f'DC2 butler {path} for {tract} not found; loading...')
                butler[version] = Butler(path)
                time_print(time, prefix=f'Loaded butler in ')
            butler_cal = butler[version]
        else:
            butler_cal = butler

        photoCalibs = {
            band: butler_cal.get('deepCoadd_photoCalib', {'tract': tract, 'patch': patch, 'filter': band})
            for band in bands
        }
        cat_calib = calibrate_catalog(cat, photoCalibs)
        if return_cat:
            cats.append(cat_calib)
        if write:
            cat_calib.writeFits(f'{filename}{postfix}')


def is_field_multiprofit(field):
    return field.startswith('multiprofit_')


def parse_multiprofit_dataId(filename):
    bands, tract, patch = filename.split('/')[-1].split('_')[-3:]
    return bands, tract, patch


def reorder_fields(cat, filters=None, func_field=None):
    """Reorder filter-dependent instFlux fields in a catalog.

    Parameters
    ----------
    cat : `lsst.afw.table.Catalog`
        A catalog with fields to be re-ordered.
    filters : iterable of `str`
        The filters in their desired order. Default ('i', 'r', 'g').
    func_field : callable
        A function that takes a string field name and returns whether it should be re-ordered. Default
        `is_field_multiprofit`.

    Returns
    -------
    A catalog with re-ordered fields.

    Notes
    -----
    If no re-ordering is necessary, the function will return cat. Otherwise, it will make a deep copy of cat
    before re-assigning the out-of-order columns. This may not be the optimal.
    """
    if filters is None:
        filters = ('i', 'r', 'g')
    filters_order = {idx: band for idx, band in enumerate(filters)}

    schema = cat.schema
    schema_new = Schema()
    fields = [x.field.getName() for x in cat.schema]
    fields_added = {}
    fields_remap = set()

    for field in fields:
        field_toadd = field
        if is_field_multiprofit(field) and field.endswith('_instFlux'):
            split = field.split('_')
            name_field = '_'.join(split[:-2])
            band = split[-2]
            if name_field not in fields_added:
                fields_added[name_field] = 0
            else:
                fields_added[name_field] += 1

            band_ordered = filters_order[fields_added[name_field]]

            if band != band_ordered:
                field_ordered = f'{name_field}_{band_ordered}_{split[-1]}'
                fields_remap.add(field)
                field_toadd = field_ordered
        schema_new.addField(schema.find(field_toadd).field)

    if not fields_remap:
        return cat

    schema_new.setAliasMap(schema.getAliasMap())
    cat_new = type(cat)(schema_new)
    cat_new.extend(cat, deep=True)

    for field in fields_remap:
        cat_new[field] = cat[field]

    return cat_new