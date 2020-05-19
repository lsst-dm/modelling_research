from lsst.afw.table import SchemaMapper, SourceCatalog
import os

# This is a limitation in the FITS format
n_columns_max = 999


def read_split_cat_fits(filename, type_cat=None):
    """Read a catalog split across multiple FITS files.

    Parameters
    ----------
    filename : `str`
        A file path to the first file in a potential series.
    type_cat : `class`
        The class type of the catalog to read; default `lsst.afw.table.SourceCatalog`.

    Returns
    -------
    cat_merged : `type_cat`
        The merged catalog.

    Notes
    -----
    This is motivated by the FITS limitation of 1000 columns per table.
    """
    if type_cat is None:
        type_cat = SourceCatalog
    prefix = filename[:-5]
    cat = type_cat.readFits(f'{prefix}.fits')
    idx = 1
    filename_next = f'{prefix}{idx}.fits'
    if not os.path.exists(filename_next):
        return cat
    names = cat.schema.getNames()
    mapper = SchemaMapper(cat.schema)
    mapper.addMinimalSchema(cat.schema, True)
    mapper.editOutputSchema().setAliasMap(cat.schema.getAliasMap())
    fields_cats_new = []
    while os.path.exists(filename_next):
        cat_next = type_cat.readFits(filename_next)
        names_new = []
        for key in cat_next.schema:
            field = key.field
            name = field.getName()
            if name not in names:
                names.add(name)
                names_new.append(name)
                mapper.editOutputSchema().addField(field)
        fields_cats_new.append((names_new, cat_next))
        idx += 1
        filename_next = f'{prefix}{idx}.fits'
    cat_merged = type_cat(mapper.getOutputSchema())
    cat_merged.extend(cat, mapper)
    for names_new, cat_in in fields_cats_new:
        for name in names_new:
            cat_merged[name] = cat_in[name]
    return cat_merged


def write_split_cat_fits(filename, cat, cat_calib):
    """ Write a catalog into as many FITS files as necessary.

    Parameters
    ----------
    filename : `str`
        A file path to the first file in a potential series.
    cat : `lsst.afw.table.BaseCatalog`
        A catalog with instrumental fluxes that can be written to a FITS table.
    cat_calib : `lsst.afw.table.BaseCatalog`
        A second catalog with calibrated fluxes of the same type as `cat`.
    """
    prefix = filename[:-5]
    names = [x.field.getName() for x in cat.schema]
    schema_new = cat.table.makeMinimalSchema()
    names_new = []
    for key in cat_calib.schema:
        field = key.field
        name = field.getName()
        if name not in names:
            if field.dtype == 'Flag':
                raise RuntimeError(f"Field {name} is flag; can't copy")
            schema_new.addField(field.copyRenamed(name))
            names_new.append(name)
    cat_new = type(cat)(schema_new)
    cat_new.resize(len(cat))
    for name in names_new:
        cat_new[name] = cat_calib[name]
    cat.writeFits(f'{prefix}.fits')
    cat_new.writeFits(f'{prefix}1.fits')
