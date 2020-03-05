import numpy as np


class Model:
    """A class for models used to measure sources in MultiProFit catalogs.
    """
    def get_total_mag(self, cat, band):
        """

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
        return (
            cat[self.field] if not self.is_multiprofit else
            cat[f'{self.field}_c1_{band}_mag'] if self.n_comps == 1 else
            -2.5 * np.log10(np.sum([
                10 ** (-0.4 * cat[f'{self.field}_c{comp + 1}_{band}_mag'])
                for comp in range(self.n_comps)], axis=0
            ))
        )

    def __init__(self, desc, field, n_comps):
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

        Notes
        -----

        All base_ and meas_ models including PsfFlux and CModel should have n_comps set to zero because
        they only store total magitudes, not individual component magnitudes. CModel stores fracDev
        separately.

        """

        self.desc = desc
        self.is_multiprofit = n_comps > 0
        self.n_comps = n_comps
        self.field = f'multiprofit_{field}' if self.is_multiprofit else field
