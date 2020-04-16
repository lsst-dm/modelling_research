from . import dc2
from lsst.afw.table import Schema, SchemaMapper, SourceCatalog
from lsst.daf.persistence import Butler
from . import meas_model as mm
from . import tables
from .timing import time_print


def calibrate_catalog(catalog, photoCalibs_filter, filter_ref=None, func_field=None):
    """Calibrate a catalog with filter-dependent instFlux columns.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        A SourceCatalog with instFlux columns.
    photoCalibs_filter : `dict` [`str`, `lsst.afw.image.PhotoCalib`]
        A dict of PhotoCalibs by filter.
    filter_ref : `str`
        The filter to use for non-multiband fluxes; default first key in list(photoCalibs_filter).
    func_field : callable
        A function that takes a field name and returns true if it should be calibrated. Defaults

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        A SourceCatalog with instFlux columns calibrated to mag columns.

    Notes
    -------
    The main purpose of this method is to calibrate a catalog with multi-band MultiProFit columns as
    generated by this task; however, it will work on any catalog as long as the columns match one of
    [*fit_*instFlux, *Flux_instFlux], and the filter-dependent columns are named *{filter}_instFlux.

    Only fields ending in '_instFlux' are calibrated; this is in addition to passing the `func_field` check.
    """
    if func_field is None:
        func_field = mm.is_field_fit
    filters = photoCalibs_filter.keys()
    if filter_ref is None:
        filter_ref = list(filters)[0]
    # See DM-23766: catalog.schema.getNames() returns a set (unordered) whereas we want a list
    columns = [x.field.getName() for x in catalog.schema]
    fluxes = ['_'.join(name.split('_')[0:-1]) for name in columns if
              (func_field(name) and name.endswith('_instFlux'))
              or name.endswith('Flux_instFlux')
              or name.endswith('SdssShape_instFlux')]
    fluxes_filter = {band: [] for band in filters}
    for flux in fluxes:
        last = flux.split('_')[-1]
        fluxes_filter[last if last in filters else filter_ref].append(flux)
    for band, photoCalib in photoCalibs_filter.items():
        catalog = photoCalib.calibrateCatalog(catalog, fluxes_filter[band])
    return catalog


def calibrate_catalogs(files, butler, func_dataId=None, is_dc2=False, return_cats=False, write=True,
                       files_ngmix=None, datasetType_ngmix=None, postfix='_mag.fits',
                       type_cat=None, type_calib=None, get_cmodel_forced=False, func_field=None):
    """Calibrate FITS source measurement catalogs derived from data in a given repo.

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
    return_cats : `bool`
        Whether to return the calibrate catalogs; they might be very large so the default is False.
    write : `bool`
        Whether to write the calibrated catalogs to disk, postfixed by `postfix.
    files_ngmix : iterable of `str`
        An iterable of paths to ngmix catalogs to calibrate. Must be same len as `files`.
    datasetType_ngmix : `str`
        The butler dataset type to retrieve for ngmix files.
    postfix : `str`
        The postfix to add to filenames when writing calibrate catalogs; default '_mag.fits'.
    type_cat: type
        A type of catalog to read; default `lsst.afw.table.SourceCatalog`.
    type_calib : `str`
        The type of calibration to use; default "deepCoadd_photoCalib".
    get_cmodel_forced: `bool`
        Whether to add cmodel forced photometry columns.
    func_field : callable
        A function to select fields; passed to `calibrate_catalog`.

    Returns
    -------
    cats : `list` [`lsst.afw.table.SourceCatalog`]
        Calibrate catalogs for each file if `return_cat`.

    Notes
    -----
    CModel forced photometry has a fixed shape in each band but different amplitudes.
    """
    if func_dataId is None:
        func_dataId = parse_multiprofit_dataId
    if is_dc2:
        tracts_dc2 = dc2.get_tracts()
        repos = dc2.get_repos()
        if butler is None:
            butler = {}
    if type_calib is None:
        type_calib = 'deepCoadd_photoCalib'

    if files_ngmix is not None:
        is_ngmix_butler = isinstance(files_ngmix, Butler)
        if is_ngmix_butler:
            if datasetType_ngmix is None:
                datasetType_ngmix = 'deepCoadd_ngmix_deblended'
        elif not (len(files_ngmix) == len(files)):
            raise RuntimeError(f'len(files_ngmix)={len(files_ngmix)} != len(files)={len(files)}')
    else:
        is_ngmix_butler = False
    if type_cat is None:
        type_cat = SourceCatalog

    time = None
    cats = []

    for idx, file in enumerate(files):
        time = time_print(time, prefix=f'Calibrated in  ' if time is not None else 'Calibrating starting',
                          postfix=f', now on {file}')
        cat = type_cat.readFits(file)
        filename = file.split('.fits')[0]
        bands, tract, patch = func_dataId(filename)
        dataId = {'tract': tract, 'patch': patch}

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

        if files_ngmix or get_cmodel_forced:
            mapper = SchemaMapper(cat.schema)
            mapper.addMinimalSchema(cat.schema, True)
            mapper.editOutputSchema().setAliasMap(cat.schema.getAliasMap())

        fields_cat_new = []
        if files_ngmix:
            cat_ngmix = files_ngmix.get(datasetType_ngmix, {'tract': tract, 'patch': patch}) if \
                is_ngmix_butler else type_cat.readFits(files_ngmix[idx])
            schema_ngmix = cat_ngmix.schema
            names = cat.schema.getNames()
            fields_new = {}
            for key in schema_ngmix:
                field = key.field
                name = field.getName()
                if name not in names:
                    name_split = name.split('_')
                    n_split = len(name_split)
                    is_flux = name_split[-2] == "flux"
                    is_flux_err = (n_split > 2) and (name_split[-3] == 'flux') and (name_split[-2] == 'err')
                    is_psf = name_split[1] == 'psf'
                    if is_flux:
                        band = name_split[-1]
                        name_new = f'{"_".join(name_split[:-2])}_{band}_instFlux' if band in bands else None
                    elif is_flux_err:
                        band = name_split[-1]
                        name_new = f'{"_".join(name_split[:-3])}_{band}_instFluxErr'\
                            if band in bands else None
                    elif is_psf and not (
                            ((n_split > 3) and (name_split[3] == 'mean')) or (
                            (n_split > 2) and (name_split[2] == 'flags'))):
                        band = name_split[2]
                        name_new = name if band in bands else None
                    else:
                        name_new = name
                    if name_new is not None:
                        fields_new[name] = name_new
                        mapper.editOutputSchema().addField(field.copyRenamed(name_new))
            fields_cat_new.append((fields_new, cat_ngmix))

        if get_cmodel_forced:
            schema = None
            fields_out = []
            for band in bands:
                forced = butler_cal.get('deepCoadd_forced_src', set_dataId_band(dataId, band))

                if schema is None:
                    schema = forced.schema
                    for key in schema:
                        field = key.field
                        if field.dtype != 'Flag':
                            name_field = field.getName()
                            if mm.is_field_modelfit(name_field):
                                name_split = name_field.split('_')[1:]
                                fields_out.append((
                                    name_field,
                                    f'modelfit_forced_{"_".join(name_split[:-1])}_',
                                    f'_{name_split[-1]}'
                                ))
                elif forced.schema != schema:
                    raise RuntimeError(f'Schema {forced.schema} for dataId {dataId} deepCoadd_forced_src '
                                       f'differs from filter {bands[0]} schema {schema}')

                fields_new = {}
                for field_in, prefix_field, postfix_field in fields_out:
                    name_new = f'{prefix_field}{band}{postfix_field}'
                    fields_new[field_in] = name_new
                    mapper.editOutputSchema().addField(
                        forced.schema.find(field_in).field.copyRenamed(name_new))
                fields_cat_new.append((fields_new, forced))

        if fields_cat_new:
            cat_new = SourceCatalog(mapper.getOutputSchema())
            cat_new.reserve(len(cat))
            cat_new.extend(cat, mapper)
            for fields_new, cat_in in fields_cat_new:
                for field_in, field_out in fields_new.items():
                    cat_new[field_out] = cat_in[field_in]
            cat = cat_new

        n_columns = len(cat.schema.getNames())
        if n_columns > tables.n_columns_max:
            raise RuntimeError(f'pre-calib cat has {n_columns}>max={tables.n_columns_max}')

        photoCalibs = {band: butler_cal.get(type_calib, set_dataId_band(dataId, band))
                       for band in bands}

        cat_calib = calibrate_catalog(cat, photoCalibs, func_field=func_field)
        if return_cats:
            cats.append(cat_calib)
        if write:
            filename_out = f'{filename}{postfix}'
            n_columns = len(cat_calib.schema.getNames())
            if n_columns > tables.n_columns_max:
                tables.write_split_cat_fits(filename_out, cat, cat_calib)
            else:
                cat_calib.writeFits(filename_out)
    if return_cats:
        return cats


def get_cat(cat, type_cat=None):
    """ Get or validate a catalog from an input.

    Parameters
    ----------
    cat : `str` or type
        A path to a FITS catalog to load, or an already-loaded catalog of type `type_cat`.
    type_cat
        The type of catalog to load or expect.

    Returns
    -------
    cat : type
        The catalog.

    Notes
    -----
    Meow.
    """
    if type_cat is None:
        type_cat = SourceCatalog
    if isinstance(cat, type_cat):
        return cat
    elif isinstance(cat, str):
        return type.readFits(str)
    raise RuntimeError(f'Unexpected type {type(cat)}!={type_cat} for cat {cat}')


def parse_multiprofit_dataId(filename):
    bands, tract, patch = filename.split('/')[-1].split('_')[-3:]
    return bands, int(tract), patch


def parse_multiprofit_dataId_Hsc(filename):
    bands, tract, patch = parse_multiprofit_dataId(filename)
    bands = tuple(f'HSC-{b.upper()}' for b in bands)
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
        `modelling_research.meas_model.is_field_multiprofit`.

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
    if func_field is None:
        func_field = mm.is_field_multiprofit
    filters_order = {idx: band for idx, band in enumerate(filters)}

    schema = cat.schema
    schema_new = Schema()
    fields = [x.field.getName() for x in cat.schema]
    fields_added = {}
    fields_remap = set()

    for field in fields:
        field_toadd = field
        if func_field(field) and field.endswith('_instFlux'):
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


def set_dataId_band(dataId, band):
    dataId['filter'] = band
    return dataId
