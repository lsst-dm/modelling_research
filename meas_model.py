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
import lsst.meas.algorithms as measAlg
import lsst.geom as geom
import matplotlib.patches as patches
import matplotlib.patheffects as pathfx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Sequence

try:
    from lsst.meas.extensions.multiprofit.utils import get_spanned_image
except ModuleNotFoundError:
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

try:
    from multiprofit.utils import flux_to_mag, mag_to_flux
except ModuleNotFoundError:
    def flux_to_mag(ndarray):
        return -2.5 * np.log10(ndarray)

    def mag_to_flux(ndarray):
        return 10 ** (-0.4 * ndarray)

try:
    from gauss2d.utils import covar_to_ellipse
except ModuleNotFoundError:
    def covar_to_ellipse(sigma_x_sq, sigma_y_sq, cov_xy, degrees=False):
        """Convert covariance matrix terms to ellipse major axis, axis ratio and
        position angle representation.

        Parameters
        ----------
        sigma_x_sq, sigma_y_sq : `float` or array-like
            x- and y-axis squared standard deviations of a 2-dimensional normal
            distribution (diagonal terms of its covariance matrix).
            Must be scalar or identical length array-likes.
        cov_xy : `float` or array-like
            x-y covariance of a of a 2-dimensional normal distribution
            (off-diagonal term of its covariance matrix).
            Must be scalar or identical length array-likes.
        degrees : `bool`
            Whether to return the position angle in degrees instead of radians.

        Returns
        -------
        r_major, axrat, angle : `float` or array-like
            Converted major-axis radius, axis ratio and position angle
            (counter-clockwise from the +x axis) of the ellipse defined by
            each set of input covariance matrix terms.

        Notes
        -----
        The eigenvalues from the determinant of a covariance matrix are:
        |a-m b|
        |b c-m|
        det = (a-m)(c-m) - b^2 = ac - (a+c)m + m^2 - b^2 = m^2 - (a+c)m + (ac-b^2)
        Solving:
        m = ((a+c) +/- sqrt((a+c)^2 - 4(ac-b^2)))/2
        ...or equivalently:
        m = ((a+c) +/- sqrt((a-c)^2 + 4b^2))/2

        Unfortunately, the latter simplification is not as well-behaved
        in floating point math, leading to square roots of negative numbers when
        one of a or c is very close to zero.

        The values from this function should match those from
        `Ellipse.make_ellipse_major` to within rounding error, except in the
        special case of sigma_x == sigma_y == 0, which returns a NaN axis ratio
        here by default. This function mainly intended to be more convenient
        (and possibly faster) for array-like inputs.
        """
        apc = sigma_x_sq + sigma_y_sq
        x = apc / 2
        pm = np.sqrt(apc ** 2 - 4 * (sigma_x_sq * sigma_y_sq - cov_xy ** 2)) / 2

        r_major = x + pm
        axrat = np.sqrt((x - pm) / r_major)
        r_major = np.sqrt(r_major)
        angle = np.arctan2(2 * cov_xy, sigma_x_sq - sigma_y_sq) / 2
        return r_major, axrat, (np.degrees(angle) if degrees else angle)


# Classes for row-wise measurements
@dc.dataclass(frozen=True)
class Centroid:
    x: float
    y: float
    x_err: float = None
    y_err: float = None

    @property
    def cen(self):
        return self.x, self.y


@dc.dataclass(frozen=True)
class Shape:
    r_maj: float
    r_min: float
    ang: float


@dc.dataclass(frozen=True)
class Ellipse:
    centroid: Centroid
    shape: Shape


@dc.dataclass(frozen=True)
class Measurement:
    mag: float
    ellipse: Ellipse
    mag_err: float


@dc.dataclass(frozen=True)
class Source:
    idx_row: int
    measurements: Sequence[Measurement]


@dc.dataclass(frozen=True)
class CatExp:
    band: str = ''
    cat: afwTable.SourceCatalog = dc.field(default=None, repr=False)
    img: afwImage.Image = dc.field(default=None, repr=False)
    model: afwImage.Image = dc.field(default=None, repr=False)
    photoCalib: afwImage.PhotoCalib = None
    psf: measAlg.CoaddPsf = None
    siginv: afwImage.Image = dc.field(default=None, repr=False)


def get_source_points(band, sources=None):
    cxs, cys, mags = [], [], []
    if sources:
        for source in sources:
            measure = source.measurements[band]
            cxs.append(measure.ellipse.centroid.x)
            cys.append(measure.ellipse.centroid.y)
            mags.append(measure.mag)
    return cxs, cys, mags


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


def add_psf_models(img, psf, calib, meas_psfs, name_model='Base PSF'):
    for meas_psf in meas_psfs:
        model = meas_psf.measurements[name_model]
        flux = calib.magnitudeToInstFlux(model.mag) if calib is not None else mag_to_flux(model.mag)
        img_psf = psf.computeImage(geom.Point2D(x=model.ellipse.centroid.x, y=model.ellipse.centroid.y))
        img_psf.array *= flux
        bbox_intersect = img_psf.getBBox().clippedTo(img.getBBox())
        if bbox_intersect.area > 0:
            img.subset(bbox_intersect).array += img_psf.subset(bbox_intersect).array


@dc.dataclass
class Deblend:
    band_ref: str
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
        label_data=None, label_model=None, offsetxy_texts=None, color_true=None,
        models_psf=None, model_psf_true=False, show=True, idx_children_sub=None, ax_legend=1, **kwargs
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
        if models_psf and len(models_psf) != 1:
            raise ValueError('models_psf must be dict with one entry')
        for b in bands:
            datum = self.data[b]
            weight = bands_weights.get(b, 1.)
            weights[b] = weight
            img, model = (
                (x.subset(bbox) if bbox is not None else x)
                for x in (datum.img, datum.model)
            )
            if models_psf:
                cat_band = datum.cat
                # TODO: determine why this is necessary
                children_band = [int(np.where(cat_band['id'] == child['id'])[0]) for child in self.children]
                meas_psfs = get_sources_meas(
                    cat_band, self.cat_ref, b, children_band, models_psf,
                    sources_true=sources_true if model_psf_true else None,
                    model_true=list(models_psf.keys())[0] if model_psf_true else None,
                    photoCalib=datum.photoCalib, offsets=measmodels['Base PSF'].get('offset', (0, 0)),
                )
                model = afwImage.ImageD(img.getBBox())
                add_psf_models(model, datum.psf, calib=datum.photoCalib, meas_psfs=meas_psfs)
            elif idx_children_sub is not None:
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
        img_model_rgb = make_lupton_rgb(
            *(i.array*w for i, w in zip(models.values(), weights.values())),
            **kwargs
        )
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img_rgb)
        bands = list(bands_weights.keys())
        label_bands = ''.join(bands)
        ax[0].set_title(f'{label_bands} {label_data} data')
        ax[1].imshow(img_model_rgb)
        n_y, n_x, n_c = img_rgb.shape
        if sources_true is not None:
            handle = plot_sources(
                ax, *get_source_points(bands[0], sources_true),
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
                    cx, cy = ellipse.centroid.cen
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
                        ax_i.annotate(f'{mag:.1f}', (cx + offsetxy_text[0], cy + offsetxy_text[1]),
                                      color=color, fontsize=5)
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

            chisq = 0
            n_chi = 0

            for idx, band in enumerate(bands):
                data_band = self.data[band]
                model_b, img_b, siginv_b = (
                    x if bbox is None else x.subset(bbox)
                    for x in (models[band], data_band.img, data_band.siginv)
                )
                residual = model_b.array - data_residual_factor * img_b.array
                chi = residual * siginv_b.array
                chi_finite = chi[np.isfinite(chi)]
                n_chi += chi_finite.size
                chisq += np.sum(chi_finite**2)

                chi_rgb[:, :, idx] = 256 * np.clip(chi / (2 * chi_clip) + 0.5, 0, 1)
                res_rgb[:, :, idx] = 256 * np.clip(residual / (2 * residual_scale) + 0.5, 0, 1)

            ax_sig[0].imshow(res_rgb)
            ax_sig[0].set_title(f'{label_bands} Model - Data Residuals (clipped +/- {residual_scale:.2f})')
            ax_sig[1].imshow(chi_rgb)
            ax_sig[1].set_title(f'{label_bands} Model - Data Chi (clipped +/- {chi_clip:.2f})'
                                f' chisqred={chisq/n_chi:.2f}')

            if sources_sig is not None:
                handle = plot_sources(
                    ax_sig, *get_source_points(bands[0], sources_sig),
                    # Can also try bbox=dict(facecolor='black', pad=1)
                    # I find that obscures the image too much
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
                    photoCalib=catexp.photoCalib,
                    psf=catexp.psf,
                    siginv=catexp.siginv.subset(bbox),
                    cat=cat,
                )
            data[deblend_in.name_deblender] = Deblend(
                band_ref=deblend_in.band_ref, cat_ref=deblend_in.cat_ref, data=data_d,
                idx_parent=deblend_in.idx_parent, name_deblender=deblend_in.name_deblender,
            )

        self.bbox = bbox
        self.data = data


def get_sources_meas(
    cat_meas, cat_ref, band, idx_children, models_meas, sources_true=None, model_true=None, photoCalib=None,
    offsets=None,
):
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
                x_err=model.get_cen(child, 'xErr', comp=1),
                y_err=model.get_cen(child, 'yErr', comp=1),
            )
            try:
                mag = model.get_mag_total(child, band=band)
            except:
                cat_meas = photoCalib.calibrateCatalog(cat_meas)
                child = cat_meas[idx_row]
                mag = model.get_mag_total(child, band=band)
            try:
                mag_err = child[f'{model.get_field_prefix(band=band)}_magErr']
            except:
                mag_err = np.nan
            # This is stupid, I know, but necessary for now... sorry
            if model.n_comps > 0:
                # It can't be a single child but must be a table for some reason
                r_maj, axrat, ang = model.get_ellipse_terms(cat_meas[idx_row:(idx_row+1)], comp=1)
                shape = Shape(r_maj=r_maj[0], r_min=r_maj[0]*axrat[0], ang=ang[0])
            else:
                shape = None
            measures[name_model] = Measurement(mag=mag, mag_err=mag_err, ellipse=Ellipse(centroid=cen, shape=shape))

        sources.append(Source(idx_row=idx_row, measurements=measures))

    if sources_true is not None:
        if model_true is None:
            raise ValueError('Must specify model_true if providing sources_true')
        if offsets is None:
            offsets = (0, 0)
        n_true = len(sources_true)
        mag_true, x_true, y_true = (np.zeros(n_true) for _ in range(3))
        unmatched = np.ones(n_true, dtype=bool)
        for idx, source in enumerate(sources_true):
            meas = source.measurements[band]
            mag_true[idx] = meas.mag
            x_true[idx] = meas.ellipse.centroid.x
            y_true[idx] = meas.ellipse.centroid.y

        n_unmatched = n_true
        for source in sorted(sources, key=lambda source: source.measurements[model_true].mag):
            if n_unmatched > 0:
                is_unmatched = unmatched == True
                meas = source.measurements[model_true]
                cen = meas.ellipse.centroid
                chi_sqs = np.zeros(n_unmatched)
                for truth, value, err in (
                    (mag_true, meas.mag, meas.mag_err),
                    (x_true, cen.x + offsets[0], cen.x_err),
                    (y_true, cen.y + offsets[1], cen.y_err),
                ):
                    chi_sqs += ((truth[is_unmatched] - value)/err)**2
                min = np.argmin(chi_sqs)
                chi_sq = chi_sqs[min]
                if chi_sq > 0 and np.isfinite(chi_sq):
                    mag_true_matched = mag_true[is_unmatched][min]
                    # Set the matched array element
                    unmatched[np.where(mag_true == mag_true_matched)[0][0]] = False
                    source.measurements[model_true] = Measurement(
                        mag=mag_true_matched, ellipse=meas.ellipse, mag_err=meas.mag_err)
                    n_unmatched -= 1

    return sources


# Classes for column-wise measurements
def get_prefix_comp_multiprofit(prefix, comp):
    return f'{prefix}_c{comp}'


def is_field_fit(field):
    return (is_field_modelfit(field) or is_field_multiprofit(field) or is_field_ngmix(field)
            or is_field_scarlet(field))


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
    column_band_prefixed = False
    column_band_separator: str = '_'
    column_flux: str = 'instFlux'
    column_separator: str = '_'
    prefix_centroid_default: str = 'base_SdssCentroid_'

    def get_cen(self, cat, axis, comp=None):
        if self.is_multiprofit:
            return cat[f'{get_prefix_comp_multiprofit(self.name, comp)}{self.column_separator}cen{axis}']
        return cat[f'{self.prefix_centroid_default}{axis}']

    def get_color_total(self, cat, band1, band2):
        """Return a single total color.

        Parameters
        ----------
        cat : `dict` [`str`, array-like]
            A table-like with equal-length array-likes of magnitudes of each
            component.
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
        return covar_to_ellipse(*terms_covar)

    def get_field_prefix(self, band=None, comp=None):
        """ Return the mandatory prefix for all model fields.

        Parameters
        ----------
        band : `str`
            The band of the field, if any.
        comp : `str`
            The component, if any.

        Returns
        -------
        prefix : `str`
            The field prefix.
        """
        prefix = self.name
        if self.column_band_prefixed:
            prefix = f'{band}{self.column_band_separator}{prefix}'
        else:
            if (self.is_psf and self.is_multiprofit) or self.is_ngmix or self.is_modelfit_forced \
                    or self.is_scarlet:
                prefix = f'{prefix}{self.column_separator}{band}'
        if comp is not None:
            return self.get_prefix_comp(prefix, comp)
        return prefix

    def get_moment_xy(self, cat, band='', comp=None):
        if self.is_multiprofit:
            return (
                self.get_rho(cat)
                * self.get_moment(cat, 'x', band=band, comp=comp)
                * self.get_moment(cat, 'y', band=band, comp=comp)
            )
        else:
            return cat[(f'{self.get_field_prefix(band=band, comp=comp)}{self.column_separator}'
                        f'{self.prefix_ellipse}xy')]

    def get_moment(self, cat, axis, band='', comp=None):
        if self.is_multiprofit:
            return cat[(
                f'{self.get_field_prefix(band=band, comp=comp)}{self.column_separator}sigma'
                f'{self.column_separator}{axis}'
            )]
        else:
            return np.sqrt(self.get_moment2(cat, axis))

    def get_moment2(self, cat, axis, band='', comp=None):
        if self.is_multiprofit:
            return self.get_moment(cat, axis, band=band, comp=comp)**2
        else:
            return cat[(
                f'{self.get_field_prefix(band=band, comp=comp)}{self.column_separator}'
                f'{self.prefix_ellipse}{axis}{axis}'
            )]

    def get_prefix_comp(self, prefix, comp):
        if self.is_multiprofit:
            return get_prefix_comp_multiprofit(prefix, comp)
        elif self.is_psf:
            return f'{prefix}{self.column_separator}{comp}'
        return prefix

    def get_rho(self, cat, band='', comp=None):
        if self.is_multiprofit:
            return cat[f'{self.get_field_prefix(band=band, comp=comp)}{self.column_separator}rho']
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
            flux = self.column_flux
        if self.is_multiprofit:
            postfix = f'{self.column_separator}{band}{self.column_separator}{flux}'
            data = [
                cat[f'{self.get_prefix_comp(self.name, comp + 1)}{postfix}']
                for comp in range(self.n_comps)
            ]
            if self.n_comps == 1:
                return data[0]
            else:
                return np.sum(data, axis=0)
        return cat[f'{self.get_field_prefix(band=band)}{self.column_separator}{flux}']

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
            return cat[(f'{get_prefix_comp_multiprofit(self.name, comp=comp)}'
                        f'{self.column_separator}{band}{self.column_separator}mag')]
        else:
            #TODO: Implement for modelfit if necessary
            return None

    def get_mag_total(self, cat, band, zeropoint=None, **kwargs):
        """Get total model magnitude.

        Parameters
        ----------
        cat : `dict` [`str`, array-like]
            A table-like with equal-length array-likes of magnitudes of each component.
        band : `str`
            A filter name.
        zeropoint : `float`
            A magnitude zeropoint. Used only in fallback if mag column does not exist.
        kwargs : `dict`
            Additional keyword arguments to pass to `get_flux_total`.

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
            try:
                mags = cat[f'{self.get_field_prefix(band=band)}{self.column_separator}mag']
            except:
                mags = flux_to_mag(self.get_flux_total(cat, band, **kwargs))
                if zeropoint is not None:
                    mags += zeropoint
        if self.mag_offset is not None:
            mags = np.copy(mags)
            mags += self.mag_offset
        return mags

    def __init__(self, desc, name, n_comps, is_psf=False, mag_offset=None,
                 column_flux=None, column_separator=None, column_band_prefixed=None,
                 column_band_separator=None, prefix_centroid_default=None):
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
        if column_flux is not None:
            self.column_flux = column_flux
        if column_separator is not None:
            self.column_separator = column_separator
        if column_band_prefixed is not None:
            self.column_band_prefixed = column_band_prefixed
        if column_band_separator is not None:
            self.column_band_separator = column_band_separator
        if prefix_centroid_default is not None:
            self.prefix_centroid_default = prefix_centroid_default
