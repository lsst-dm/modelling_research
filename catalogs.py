from lsst.afw.table import Schema, SourceCatalog
from .meas_model import is_field_multiprofit
from .tables import read_split_cat_fits
from .timing import time_print


def read_source_fits_as_astropy(file, rows_expect=None, log=False, return_time=False, preprint=None,
                                read_split_cat=False, **kwargs):
    if preprint is not None:
        print(preprint, end='')
    table = read_split_cat_fits(file) if read_split_cat else SourceCatalog.readFits(file)
    if rows_expect and len(table) != rows_expect:
        raise RuntimeError(f'Loaded file {file} len={len(table)} != expected={rows_expect}')
    table = table.asAstropy()
    if log:
        if 'prefix' in kwargs:
            prefix = kwargs['prefix']
            del kwargs['prefix']
        else:
            prefix = f'Loaded in '
        time = time_print(prefix=prefix, **kwargs)
    if return_time:
        return table, time
    return table


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
    if func_field is None:
        func_field = is_field_multiprofit
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
