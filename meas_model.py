from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from multiprofit.gaussutils import covar_to_ellipse
from multiprofit.utils import flux_to_mag, mag_to_flux
from collections import namedtuple
import numpy as np


# Classes for row-wise measurements
Centroid = namedtuple('Centroid', ['x', 'y'])
Shape = namedtuple('Shape', ['r_maj', 'r_min', 'ang'])
Ellipse = namedtuple('Ellipse', ['centroid', 'shape'])
Measurement = namedtuple('Measurement', ['mag', 'ellipse'])
Source = namedtuple('Source', ['idx_row', 'measurements'])
BlendDatum = namedtuple('BlendDatum', ['img', 'siginv', 'models'])


class Blend:
    def make_plots(
        self, name_model, bands_weights, plot_sig=False, data_residual_factor=1, bands=None,
        sources_true=None, sources=None, measmodels=None, chi_clip=3, residual_scale=1,
        label_data=None, label_model=None, offsetxy_texts=None, color_true=None, show=True,
        **kwargs
    ):
        if bands is None:
            bands = 'irg'
        if sources is None:
            sources = []
        if label_data is None:
            label_data = ""
        if label_model is None:
            label_model = ""
        if offsetxy_texts is None:
            offsetxy_texts = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        if color_true is None:
            color_true = 'pink'
        img_rgb = make_lupton_rgb(*(self.data[b].img * bands_weights.get(b) for b in bands), **kwargs)
        img_model_rgb = make_lupton_rgb(
            *(self.data[b].models[name_model] * bands_weights.get(b) for b in bands), **kwargs)
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img_rgb)
        bands = list(bands_weights.keys())
        label_bands = ''.join(bands)
        ax[0].set_title(f'{label_bands} {label_data} data')
        ax[1].imshow(img_model_rgb)
        n_y, n_x, n_c = img_rgb.shape
        cxs, cys, mags, ellipses = [], [], [], []
        for source in (sources_true if sources_true is not None else []):
            measure = source.measurements[0]
            cxs.append(measure.ellipse.centroid[0])
            cys.append(measure.ellipse.centroid[1])
            mags.append(measure.mag)
        for cx, cy, mag in zip(cxs, cys, mags):
            for ax_i in ax:
                ax_i.annotate(f'{mag:.1f}', (cx, cy), color=color_true, fontsize=5, ha='right', va='top')
        for ax_i in ax:
            ax_i.scatter(cxs, cys, marker='o', color=color_true, s=0.5)

        for idx_model, (model, specs) in enumerate(measmodels.items()):
            cxs, cys, mags, ellipses = [], [], [], []
            offsets = specs.get('offset', (0, 0))
            offsetxy_text = offsetxy_texts[idx_model]
            for source in sources:
                measure = source.measurements.get(model)
                if measure is not None:
                    ellipse = measure.ellipse
                    cx, cy = ellipse.centroid
                    cx += offsets[0]
                    cy += offsets[1]
                    if (cx > 0) & (cx < n_x) & (cy > 0) & (cy < n_y):
                        cxs.append(cx)
                        cys.append(cy)
                        mags.append(measure.mag)
                        shape = ellipse.shape
                        if shape is not None:
                            ellipses.append(((cx, cy), shape.r_maj, shape.r_min, shape.ang))
            for ax_i in ax:
                scatter_opts = specs.get('scatter_opts', {})
                ax_i.scatter(cxs, cys, **scatter_opts)
                for spec in ('color', 'edgecolors', 'facecolors'):
                    color = scatter_opts.get(spec)
                    if color is not None:
                        break
                if color is None:
                    color = 'white'
                for cx, cy, mag in zip(cxs, cys, mags):
                    ax_i.annotate(f'{mag:.1f}', (cx + offsetxy_text[0], cy + offsetxy_text[1]), color=color,
                                  fontsize=5)
            for ellipse in ellipses:
                for ax_i in ax:
                    ell_patch = patches.Ellipse(*ellipse, fill=False)
                    ax_i.add_patch(ell_patch)
        ax[1].set_title(f'{label_bands} {label_model} (neighb.*{data_residual_factor:.1f})')

        if plot_sig:
            fig, ax = plt.subplots(ncols=2)
            chi_rgb = np.zeros_like(img_rgb)
            res_rgb = np.zeros_like(img_rgb)

            for idx, band in enumerate(bands):
                residual = self.data[band].models[name_model] - data_residual_factor * self.data[band].img
                chi = residual * self.data[band].siginv
                chi_rgb[:, :, idx] = 256 * np.clip(chi / (2 * chi_clip) + 0.5, 0, 1)
                res_rgb[:, :, idx] = 256 * np.clip(residual / (2 * residual_scale) + 0.5, 0, 1)
            ax[0].imshow(res_rgb)
            ax[0].set_title(f'{label_bands} Residuals (clipped +/- {residual_scale:.2f})')
            ax[1].imshow(chi_rgb)
            ax[1].set_title(f'{label_bands} Chi (clipped +/- {chi_clip:.2f})')
        if show:
            plt.show()

    def __init__(self, blenddata, cat_ref, idx_parent):
        data = {}
        parent = cat_ref[idx_parent]
        bbox = parent.getFootprint().getBBox()
        for band, blenddatum in blenddata:
            img = blenddatum.img


def get_sources_meas(cat_meas, cat_ref, band_ref, idx_children, models_meas):
    sources = []
    for idx, idx_row in enumerate(idx_children):
        child = cat_meas[idx_row]
        child_ref = cat_ref[idx_row]
        measures = {}
        cxo, cyo = child_ref.getFootprint().getBBox().getBegin()
        for name_model, model in models_meas.items():
            is_mpf = name_model.startswith('MPF')
            cen = Centroid(
                x=model.get_cen(child, 'x', comp=1) + (cxo if is_mpf else 0),
                y=model.get_cen(child, 'y', comp=1) + (cyo if is_mpf else 0),
            )
            mag = model.get_mag_total(child, band=band_ref)
            # This is stupid, I know, but necessary for now... sorry
            if model.n_comps > 0:
                # It can't be a single child but must be a table for some reason
                r_maj, axrat, ang = model.get_ellipse_terms(cat_meas[idx_row:(idx_row+1)], comp=1)
                shape = Shape(r_maj=r_maj[0], r_min=r_maj[0]*axrat[0], ang=ang[0])
            else:
                shape = None
            measures[name_model] = Measurement(mag=mag, ellipse=Ellipse(centroid=cen, shape=shape))

        sources.append(Source(idx_row=idx_row, measurements=measures))
    return sources


# Classes for column-wise measurements
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
        return cat[f'base_SdssCentroid_{axis}']

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
