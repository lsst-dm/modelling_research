from multiprofit.gaussutils import covar_to_ellipse
from multiprofit.utils import flux_to_mag, mag_to_flux
import numpy as np


def get_prefix_comp_multiprofit(prefix, comp):
    return f'{prefix}_c{comp}'


def is_field_fit(field):
    return is_field_modelfit(field) or is_field_multiprofit(field) or is_field_ngmix(field) or is_field_scarlet(field)


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


def is_field_scarlet(field):
    return field.startswith('scarlet')


class Model:
    """A class for models used to measure sources in MultiProFit catalogs.
    """
    def get_cen(self, cat, axis, comp=None):
        if self.is_multiprofit:
            return cat[f'{get_prefix_comp_multiprofit(self.name, comp)}_cen{axis}']
        return None

    def get_color_total(self, cat, band1, band2):
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
        return self.get_mag_total(cat, band1) - self.get_mag_total(cat, band2)

    def get_corr_terms(self, cat, band='', comp=None):
        x = self.get_moment(cat, 'x', band=band, comp=comp)
        y = self.get_moment(cat, 'y', band=band, comp=comp)
        if self.is_multiprofit:
            rho = self.get_rho(cat, band=band, comp=comp)
        else:
            rho = self.get_moment_xy(cat, band=band, comp=comp)/(x*y)
        return x, y, rho

    def get_covar_terms(self, cat, band='', comp=None):
        if self.is_multiprofit:
            x, y, rho = self.get_corr_terms(cat, band=band, comp=comp)
            xx, yy, xy = x*x, y*y, rho*x*y
        else:
            xx = self.get_moment2(cat, 'x', band=band, comp=comp)
            yy = self.get_moment2(cat, 'y', band=band, comp=comp)
            xy = self.get_moment_xy(cat, band=band, comp=comp)
        return xx, yy, xy

    def get_ellipse_terms(self, cat, band='', comp=None):
        terms_covar = self.get_covar_terms(cat, band=band, comp=comp)
        if any([x is None for x in terms_covar]):
            return None
        return covar_to_ellipse(terms_covar, use_method_eigen=False)

    def get_field_prefix(self, band=None, comp=None):
        """ Return the mandatory prefix for all model fields.

        Parameters
        ----------
        band
        comp

        Returns
        -------

        """
        prefix = self.name
        if (self.is_psf and self.is_multiprofit) or self.is_ngmix or self.is_modelfit_forced or self.is_scarlet:
            prefix = f'{prefix}_{band}'
        if comp is not None:
            return self.get_prefix_comp(prefix, comp)
        return prefix

    def get_moment_xy(self, cat, band='', comp=None):
        if self.is_multiprofit:
            self.get_rho(cat) * \
                self.get_moment(cat, 'x', band=band, comp=comp) * \
                self.get_moment(cat, 'y', band=band, comp=comp)
        else:
            return cat[f'{self.get_field_prefix(band=band, comp=comp)}_{self.prefix_ellipse}xy']

    def get_moment(self, cat, axis, band='', comp=None):
        if self.is_multiprofit:
            return cat[
                f'{self.get_field_prefix(band=band, comp=comp)}_sigma_{axis}'
            ]
        else:
            return np.sqrt(self.get_moment2(cat, axis))

    def get_moment2(self, cat, axis, band='', comp=None):
        if self.is_multiprofit:
            return self.get_moment(cat, axis, band=band, comp=comp)**2
        else:
            return cat[
                f'{self.get_field_prefix(band=band, comp=comp)}_{self.prefix_ellipse}{axis}{axis}'
            ]

    def get_prefix_comp(self, prefix, comp):
        if self.is_multiprofit:
            return get_prefix_comp_multiprofit(prefix, comp)
        elif self.is_psf:
            return f'{prefix}_{comp}'
        return prefix

    def get_rho(self, cat, band='', comp=None):
        if self.is_multiprofit:
            return cat[f'{self.get_field_prefix(band=band, comp=comp)}_rho']
        else:
            return self.get_moment_xy(cat, band=band, comp=comp)/(
                self.get_moment(cat, 'x', band=band, comp=comp) *
                self.get_moment(cat, 'y', band=band, comp=comp)
            )

    def get_flux_total(self, cat, band, flux=None):
        """Get total model flux.

        Parameters
        ----------
        cat : `dict` [`str`, array-like]
            A table-like with equal-length array-likes of fluxes of each component.
        band : `str`
            A filter name.
        flux : `str`
            The name of the type of flux to sum; default "flux".
        Returns
        -------
        fluxes : array-like
            Total model magnitude for all sources.
        """
        if self.is_psf:
            return None
        if flux is None:
            flux = 'flux'
        if self.is_multiprofit:
            postfix = f'_{band}_{flux}'
            data = [
                cat[f'{self.get_prefix_comp(self.name, comp + 1)}{postfix}']
                for comp in range(self.n_comps)
            ]
            if self.n_comps == 1:
                return data[0]
            else:
                return np.sum(data, axis=0)
        return cat[f'{self.name}_{flux}']

    def get_mag_comp(self, cat, band, comp):
        """Get a single component's magnitude.

        Parameters
        ----------
        cat : `dict` [`str`, array-like]
            A table-like with equal-length array-likes of fluxes of each component.
        band : `str`
            A filter name.
        comp : `str`
            A component identifier.

        Returns
        -------
        mags : array-like
            Component magnitude for all sources.
        """
        if self.is_psf:
            return None
        elif self.is_multiprofit:
            return cat[f'{get_prefix_comp_multiprofit(self.name, comp=comp)}_{band}_mag']
        else:
            #TODO: Implement for modelfit if necessary
            return None

    def get_mag_total(self, cat, band):
        """Get total model magnitude.

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
        if self.is_psf:
            return None
        elif self.is_multiprofit:
            mags = self.get_mag_comp(cat=cat, band=band, comp=1)
            if self.n_comps > 1:
                mags = mag_to_flux(mags)
                for idx in range(2, self.n_comps + 1):
                    mags += mag_to_flux(self.get_mag_comp(cat=cat, band=band, comp=idx))
                mags = flux_to_mag(mags)
        else:
            mags = cat[f'{self.get_field_prefix(band=band)}_mag']
        if self.mag_offset is not None:
            mags = np.copy(mags)
            mags += self.mag_offset
        return mags

    def __init__(self, desc, name, n_comps, is_psf=False, mag_offset=None):
        """Describe a model and enable retrieval of its parameters.

        Parameters
        ----------
        desc : `str`
            A longer descriptive name for the model.
        name : `str`
            The prefix for this model's fields in catalogs.
        n_comps : `int`
            The number of explicit components with magnitudes stored by this model,
            or zero if only a total magnitude is stored.
        is_psf : `bool`
            Whether this is a PSF model; default False.
        mag_offset : `float`
            An additive magnitude offset to apply when returning mags. Default None (zero).

        Notes
        -----

        All base_ and meas_ models including PsfFlux and CModel should have n_comps set to zero because
        they only store total magitudes, not individual component magnitudes. CModel stores fracDev
        separately.

        """

        self.n_comps = n_comps
        self.is_psf = is_psf
        self.desc = desc
        self.n_comps = n_comps
        self.name = name
        self.is_modelfit = is_field_modelfit(name)
        self.is_modelfit_model = self.is_modelfit and not self.is_psf
        self.is_modelfit_forced = is_field_modelfit_forced(name)
        self.is_multiprofit = is_field_multiprofit(name)
        self.is_ngmix = is_field_ngmix(name)
        self.is_scarlet = is_field_scarlet(name)
        self.multiband = self.is_multiprofit or self.is_ngmix or self.is_scarlet
        self.prefix_ellipse = 'ellipse_' if self.is_modelfit_model else ''
        self.mag_offset = mag_offset
