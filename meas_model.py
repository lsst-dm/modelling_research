import numpy as np


def is_field_fit(field):
    return is_field_modelfit(field) or is_field_multiprofit(field) or is_field_ngmix(field)


def is_field_instFlux(field):
    return field.endswith('_instFlux')


def is_field_modelfit(field):
    return field.startswith('modelfit_')


def is_field_modelfit_forced(field):
    return field.startswith('modelfit_forced_')


def is_field_multiprofit(field):
    return field.startswith('multiprofit_')


def is_field_ngmix(field):
    return field.startswith('ngmix_')


class Model:
    """A class for models used to measure sources in MultiProFit catalogs.
    """
    def get_total_color(self, cat, band1, band2):
        """Return a single total color.

        Parameters
        ----------
        cat : `dict` [`str`, array-like]
            A table-like with equal-length array-likes of magnitudes of each component.
        band1 : `str`
            A filter name.
        band2 : `str`
            The name of the filter to subtract from `band1`.

        Returns
        -------
        colors: array-like
            Total `band1` - `band2` color for all sources.
        """
        return self.get_total_mag(cat, band1) - self.get_total_mag(cat, band2)

    # TODO: add apCorr field
    def get_total_mag(self, cat, band):
        """Return the total magnitude in one band.

        Parameters
        ----------
        cat : `dict` [`str`, array-like]
            A table-like with equal-length array-likes of magnitudes of each component.
        band : `str`
            A filter name.

        Returns
        -------
        mags: array-like
            Total model magnitude for all sources.
        """
        if self.is_multiprofit:
            mags = cat[f'{self.field}_c1_{band}_mag'] if self.n_comps == 1 else (
                -2.5 * np.log10(np.sum([
                    10 ** (-0.4 * cat[f'{self.field}_c{comp + 1}_{band}_mag'])
                    for comp in range(self.n_comps)], axis=0
                ))
            )
        else:
            postfix = f'_{band}' if (self.is_ngmix or self.is_modelfit_forced) else ''
            mags = cat[f'{self.field}{postfix}_mag']
        if self.mag_offset is not None:
            mags = np.copy(mags)
            mags += self.mag_offset
        return mags

    def __init__(self, desc, field, n_comps, mag_offset=None):
        """Describe a model and its relevant fields.

        Parameters
        ----------
        desc : `str`
            A longer descriptive name for the model.
        field : `str`
            The prefix for this model's fields in catalogs.
        n_comps : `int`
            The number of explicit components with magnitudes stored by this model,
            or zero if only a total magnitude is stored.
        mag_offset : `float`
            An additive magnitude offset to apply when returning mags. Default None (zero).

        Notes
        -----

        All base_ and meas_ models including PsfFlux and CModel should have n_comps set to zero because
        they only store total magitudes, not individual component magnitudes. CModel stores fracDev
        separately.

        """

        self.desc = desc
        self.is_multiprofit = n_comps > 0
        self.is_ngmix = is_field_ngmix(field)
        self.is_modelfit_forced = is_field_modelfit_forced(field)
        self.multiband = self.is_multiprofit or self.is_ngmix
        self.n_comps = n_comps
        self.field = f'multiprofit_{field}' if self.is_multiprofit else field
        self.mag_offset = mag_offset
