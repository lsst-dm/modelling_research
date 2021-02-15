# This file is part of modelling_research.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from astropy.visualization import make_lupton_rgb
import dataclasses as dc
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import matplotlib.patches as patches
import matplotlib.patheffects as pathfx
import matplotlib.pyplot as plt
from multiprofit.gaussutils import covar_to_ellipse
from multiprofit.utils import flux_to_mag, mag_to_flux
from typing import Dict, NamedTuple, Sequence
import numpy as np


# Classes for row-wise measurements
Centroid = NamedTuple('Centroid', [('x', float), ('y', float)])
Shape = NamedTuple('Shape', [('r_maj', float), ('r_min', float), ('ang', float)])
Ellipse = NamedTuple('Ellipse', [('centroid', Centroid), ('shape', Shape)])
Measurement = NamedTuple('Measurement', [('mag', float), ('ellipse', Ellipse)])
Source = NamedTuple('Source', [('idx_row', int), ('measurements', Sequence[Measurement])])
CatExp = NamedTuple('CatExp', [
    ('band', str),
    ('cat', afwTable.SourceCatalog),
    ('img', afwImage.Image),
    ('model', afwImage.Image),
    ('siginv', afwImage.Image),
])


def get_source_points(sources=None):
    cxs, cys, mags = [], [], []
    if sources:
        for source in sources:
            measure = source.measurements[0]
            cxs.append(measure.ellipse.centroid[0])
            cys.append(measure.ellipse.centroid[1])
            mags.append(measure.mag)
    return cxs, cys, mags


# TODO: Allow addition to existing image
def get_spanned_image(footprint, bbox=None):
    spans = footprint.getSpans()
    bbox_is_none = bbox is None
    if bbox_is_none:
        bbox = footprint.getBBox()
    if not (bbox.getHeight() > 0 and bbox.getWidth() > 0):
        return None, bbox
    bbox_fp = bbox if bbox_is_none else footprint.getBBox()
    img = afwImage.Image(bbox_fp, dtype='D')
    spans.setImage(img, 1)
    img.array[img.array == 1] = footprint.getImageArray()
    if not bbox_is_none:
        img = img.subset(bbox)
    return img.array, bbox


def plot_sources(ax, cxs, cys, mags, kwargs_annotate=None, kwargs_scatter=None, path_effects=None):
    if kwargs_annotate is None:
        kwargs_annotate = {}
    if kwargs_scatter is None:
        kwargs_scatter = {}
    for cx, cy, mag in zip(cxs, cys, mags):
        for ax_i in ax:
            text = ax_i.annotate(f'{mag:.1f}', (cx, cy), **kwargs_annotate)
            if path_effects is not None:
                text.set_path_effects(path_effects)
    handles = []
    for ax_i in ax:
        handles.append(ax_i.scatter(cxs, cys, **kwargs_scatter))
    return handles


@dc.dataclass
class Deblend:
    cat_ref: afwTable.SourceCatalog
    children: afwTable.SourceCatalog = dc.field(init=False, repr=False)
    data: Dict[str, CatExp]
    idx_children: np.ndarray = dc.field(init=False, repr=False)
    idx_parent: int
    is_child: np.ndarray = dc.field(init=False, repr=False)
    parent: afwTable.SourceRecord = dc.field(init=False, repr=False)
    name_deblender: str

    def plot(
        self, bands_weights, bbox=None, plot_sig=False, data_residual_factor=1, bands=None,
        sources=None, sources_true=None, sources_sig=None, measmodels=None, chi_clip=3, residual_scale=1,
        label_data=None, label_model=None, offsetxy_texts=None, color_true=None, show=True, idx_children_sub=None,
        ax_legend=1, **kwargs
    ):
        if bands is None:
            if len(bands_weights) == 3:
                bands = bands_weights.keys()
            else:
                bands = 'irg'
        if sources is None:
            sources = []
        if label_data is None:
            label_data = ""
        if label_model is None:
            label_model = self.name_deblender
        if offsetxy_texts is None:
            offsetxy_texts = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        if color_true is None:
            color_true = 'pink'
        imgs, models, weights = {}, {}, {}
        for b in bands:
            datum = self.data[b]
            weight = bands_weights.get(b, 1.)
            weights[b] = weight
            img, model = (
                (x.subset(bbox) if bbox is not None else x)
                for x in (datum.img, datum.model)
            )
            if idx_children_sub is not None:
                model = model.clone()
                bbox_img = img.getBBox()
                cat = self.data[b].cat
                for idx_child in idx_children_sub:
                    fp_child = cat[idx_child].getFootprint()
                    bbox_intersect = fp_child.getBBox().clippedTo(bbox_img)
                    if bbox_intersect.area > 0:
                        model_child, bbox_extra = get_spanned_image(fp_child, bbox=bbox_intersect)
                        model.subset(bbox_intersect).array -= model_child
            imgs[b] = weight * img.array
            models[b] = model
        img_rgb = make_lupton_rgb(*(imgs.values()), **kwargs)
        img_model_rgb = make_lupton_rgb(*(i.array*w for i, w in zip(models.values(), weights.values())), **kwargs)
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img_rgb)
        bands = list(bands_weights.keys())
        label_bands = ''.join(bands)
        ax[0].set_title(f'{label_bands} {label_data} data')
        ax[1].imshow(img_model_rgb)
        n_y, n_x, n_c = img_rgb.shape
        if sources_true is not None:
            handle = plot_sources(
                ax, *get_source_points(sources_true),
                kwargs_annotate=dict(color=color_true, fontsize=5, ha='right', va='top'),
                kwargs_scatter=dict(marker='o', color=color_true, s=0.5)
            )[ax_legend]
            handle.set_label('True Mags')

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
            if cxs and cys:
                for idx_ax, ax_i in enumerate(ax):
                    scatter_opts = specs.get('scatter_opts', {})
                    handle = ax_i.scatter(cxs, cys, **scatter_opts)
                    if idx_ax == ax_legend:
                        handle.set_label(model)
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
        legend = ax[ax_legend].legend(facecolor='k')
        for text, handle in zip(legend.get_texts(), legend.legendHandles):
            color = handle.get_edgecolor()
            if color is None:
                color = handle.get_facecolor()
            # Don't ask why but sometimes one or the other functions returns a list of length one
            text.set_color(color[0] if isinstance(color, list) else color[0])

        if plot_sig:
            fig_sig, ax_sig = plt.subplots(ncols=2)
            chi_rgb = np.zeros_like(img_rgb)
            res_rgb = np.zeros_like(img_rgb)

            for idx, band in enumerate(bands):
                data_band = self.data[band]
                model_b, img_b, siginv_b = (
                    x if bbox is None else x.subset(bbox)
                    for x in (models[band], data_band.img, data_band.siginv)
                )
                residual = model_b.array - data_residual_factor * img_b.array
                chi = residual * siginv_b.array
                chi_rgb[:, :, idx] = 256 * np.clip(chi / (2 * chi_clip) + 0.5, 0, 1)
                res_rgb[:, :, idx] = 256 * np.clip(residual / (2 * residual_scale) + 0.5, 0, 1)
            ax_sig[0].imshow(res_rgb)
            ax_sig[0].set_title(f'{label_bands} Residuals (clipped +/- {residual_scale:.2f})')
            ax_sig[1].imshow(chi_rgb)
            ax_sig[1].set_title(f'{label_bands} Chi (clipped +/- {chi_clip:.2f})')

            if sources_sig is not None:
                handle = plot_sources(
                    ax_sig, *get_source_points(sources_sig),
                    # Can also try bbox=dict(facecolor='black', pad=1) but I find that it obscures the image too much
                    kwargs_annotate=dict(color='w', fontsize=4.5, ha='right', va='top'),
                    #kwargs_scatter=dict(marker='o', facecolor='w', edgecolor='k', s=9),
                    kwargs_scatter=dict(marker='o', facecolor='none', edgecolor='none', s=0),
                    path_effects=[pathfx.withStroke(linewidth=1.5, foreground='k')],
                )[ax_legend]
                handle.set_label('True Mags')
                legend = ax[ax_legend].legend(facecolor='k')
                for text, handle in zip(legend.get_texts(), legend.legendHandles):
                    color = handle.get_edgecolor()
                    if color is None:
                        color = handle.get_facecolor()
                    # Don't ask why but sometimes one or the other functions returns a list of length one
                    text.set_color(color[0] if isinstance(color, list) else color[0])
        else:
            fig_sig, ax_sig = None, None
        if show:
            plt.show()
        return fig, ax, fig_sig, ax_sig

    def __post_init__(self):
        self.parent = self.cat_ref[self.idx_parent]
        self.is_child = self.cat_ref['parent'] == self.parent['id']
        self.idx_children = np.nonzero(self.is_child)[0].astype(np.int16)
        self.children = self.cat_ref[self.is_child]


class Blend:
    def __init__(self, deblends: Sequence[Deblend]):
        maxs = np.iinfo(np.int32)
        mins = np.array((maxs.max, maxs.max))
        maxs = np.array((maxs.min, maxs.min))

        for deblend in deblends:
            for band, datum in deblend.data.items():
                if band != 'ref':
                    bbox = datum.cat[deblend.idx_parent].getFootprint().getBBox()
                    np.minimum(mins, bbox.getBegin(), out=mins)
                    np.maximum(maxs, bbox.getEnd(), out=maxs)

        bbox = geom.Box2I(minimum=geom.Point2I(mins), maximum=geom.Point2I(maxs))

        data = {}
        for deblend_in in deblends:
            data_d = {}
            for band, catexp in deblend_in.data.items():
                cat = catexp.cat[deblend_in.is_child]
                img_model = afwImage.Image(bbox, dtype='F')
                for child in cat:
                    model_child, bbox_extra = get_spanned_image(child.getFootprint())
                    if model_child is not None:
                        img_model.subset(bbox_extra).array += model_child
                data_d[band] = CatExp(
                    band=band,
                    img=catexp.img.subset(bbox),
                    model=img_model,
                    siginv=catexp.siginv.subset(bbox),
                    cat=cat,
                )
            data[deblend_in.name_deblender] = Deblend(
                cat_ref=deblend_in.cat_ref, data=data_d, idx_parent=deblend_in.idx_parent,
                name_deblender=deblend_in.name_deblender,
            )

        self.bbox = bbox
        self.data = data


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
