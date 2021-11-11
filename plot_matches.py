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

from modelling_research.plotting import plotjoint_running_percentiles
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _source_is_type(cat, resolved, include_nan=False, threshold=0.5):
    if include_nan:
        return ~_source_is_type(cat, not resolved, threshold=threshold)
    else:
        return (cat['refExtendedness'] >= threshold) if resolved else (
                cat['refExtendedness'] < threshold)


def __update_bin(idx_bin, values_x, values_y, out, thresholds, idx_thresh, done, greater=True):
    y = values_y[idx_bin]
    while (not done) and ((y >= thresholds[idx_thresh]) if greater else (y <= thresholds[idx_thresh])):
        idx_prev = idx_bin - 1
        y_prev = values_y[idx_prev]
        x = values_x[idx_bin]
        if idx_bin > 0:
            shift = (thresholds[idx_thresh] - y_prev) / (y - y_prev)
            x += shift * (x - values_x[idx_prev])
        out[idx_thresh] = x
        idx_thresh += 1
        if idx_thresh >= len(thresholds):
            done = True
    return done, idx_thresh


def _get_compure_percentiles_mags(edge_mags, x_mags, compures, percentiles, mags):
    mags_pc, pcs_mag = (np.repeat(np.nan, len(x)) for x in (percentiles, mags))
    idx_pc, idx_mag = 0, 0
    done_pc, done_mags = False, False
    for idx_bin in range(len(edge_mags)):
        done_pc, idx_pc = __update_bin(idx_bin, x_mags, compures, mags_pc, percentiles, idx_pc, done_pc)
        done_mags, idx_mag = __update_bin(idx_bin, compures, x_mags, pcs_mag, mags, idx_mag, done_mags,
                                          greater=False)
    pcs_mag = np.clip(pcs_mag, 0, 1)
    return mags_pc, pcs_mag


def plot_compure(
        mags, matched, mag_max=None, mag_bin_complete=0.1,
        label_x='mag', label_y='Completeness', prefix_title='', postfix_title='',
        title_middle_n=True, percentiles=None, mags_print=None, label_step=1
):
    """ Plot completeness or purity versus magnitude.

    Parameters
    ----------
    mags : array-like
        Magnitudes of sources.
    matched : array-like
        The type of match: > 0 is right type, <0 is wrong type, 0 is no match.
    mag_max : `float`
        The maximum (faintest) magnitude to plot.
    mag_bin_complete : `float`
        The width of the magnitude bin to plot.
    label_x, label_y : `str`
        X- and y-axis labels.
    prefix_title, postfix_title : `str`
        Pre- and postfixes for the title.
    title_middle_n : `bool`
        Whether to print the number of sources between the pre/post-fixes.
    percentiles : array-like
        Percentile thresholds to print magnitudes for.
    mags_print : array-like
        Magnitudes to print compurity for.
    label_step : `float`
        Distance between labels on the (log) y-axis.

    Returns
    -------
    lim_x : array-like
        The plot x limits.
    """

    label_step = np.clip(label_step, 0.1, np.Inf)
    if percentiles is None:
        percentiles = np.array([0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    if mags_print is None:
        mags_print = 1.
    if mag_max is None or (not np.isfinite(mag_max)):
        mag_max = np.ceil(np.max(mags) / mag_bin_complete) * mag_bin_complete
    n_bins = np.int(np.ceil((mag_max - np.min(mags)) / mag_bin_complete)) + 1

    compure_true, compure_false = (np.zeros(n_bins) for _ in range(2))
    n_matches_true, n_matches_false, n_withins = (-np.ones(n_bins) for _ in range(3))
    errors = np.ones(n_bins)
    mag_bins = np.floor((mag_max - mags) / mag_bin_complete).astype(int)
    edge_mags = mag_max - np.arange(n_bins) * mag_bin_complete
    x_mags = edge_mags - 0.5 * mag_bin_complete

    if np.isscalar(mags_print):
        mags_print = np.arange(edge_mags[0], edge_mags[-1], -np.abs(mags_print))[1:]

    n_total = 0
    for idx in range(n_bins):
        within = mag_bins == idx
        n_match_true = np.sum(matched[within] >= 1)
        n_match_false = np.sum(matched[within] <= -1)
        n_within = np.sum(within)
        n_total += n_within
        if n_within > 0:
            compure_true[idx] = n_match_true / n_within
            compure_false[idx] = n_match_false / n_within
            errors[idx] = 1 / np.sqrt(n_within)
            n_withins[idx] = np.log10(n_within)
            if n_match_true > 0:
                n_matches_true[idx] = np.log10(n_match_true)
            if n_match_false > 0:
                n_matches_false[idx] = np.log10(n_match_false)

    xlim = (mag_max - n_bins * mag_bin_complete, mag_max)
    has_false = any(compure_false > 0)
    compure = compure_true + compure_false
    fig = sns.JointGrid(x=x_mags, y=compure, xlim=xlim, ylim=(0, 1))
    fig.plot_joint(
        plt.errorbar, yerr=errors, color='k', label='All match' if has_false else None
    ).set_axis_labels(label_x, label_y)
    if has_false:
        sns.lineplot(x_mags, compure_true, color='b', label='Right type')
        sns.lineplot(x_mags, compure_false, color='r', label='Wrong type')
    fig.ax_marg_y.set_axis_off()
    ax = fig.ax_marg_x
    # I couldn't figure out a compelling way to do this in seaborn with distplot,
    # even though it worked in plotjoint. Oh well.
    # 0.5 tends to be too crowded
    n_log_max = np.ceil(np.max(n_withins) / label_step) * label_step
    ax.set_ylim(-0.25, n_log_max)
    ax.step(edge_mags, n_withins, color='k', where='post')
    if has_false:
        ax.step(edge_mags, n_matches_true, color='b', where='post')
        ax.step(edge_mags, n_matches_false, color='r', where='post')
    ticks_y = np.arange(0, n_log_max + 1, label_step)
    ax.yaxis.set_ticks(ticks_y)
    ax.yaxis.set_ticklabels((f'$10^{{{x:.1f}}}$' for x in ticks_y), visible=True)
    ax.yaxis.tick_left()
    ax.tick_params(which='minor', direction='out', length=6, width=2, colors='k')
    title = f' N={n_total}' if title_middle_n else ''
    fig.fig.suptitle(f'{prefix_title}{title}{postfix_title}', y=1., verticalalignment='top')
    mags_pc, pcs_mag = _get_compure_percentiles_mags(edge_mags, x_mags, compure, percentiles, mags_print)
    text_pcs = '\n'.join(f'{100 * pc:2.1f}%: {mag_pc:.2f}'
                         for pc, mag_pc in zip(reversed(percentiles), reversed(mags_pc)))
    text_mags = '\n'.join(f'{mag_pc:.2f}: {100 * pc:5.1f}%'
                          for pc, mag_pc in zip(reversed(pcs_mag), reversed(mags_print)))
    fig.fig.text(0.825, 0.95, f'{text_pcs}\n\n{text_mags}', verticalalignment='top')
    plt.show()
    return xlim


def _plot_completeness(mags_true, cat_target, matched_truth, indices, select_truth, resolved,
                       postfix_title='', field_extend=None, **kwargs):
    if field_extend is None:
        field_extend = 'refExtendedness'
    for band, mags_true_b in mags_true.items():
        matched = select_truth & matched_truth
        good_mod = np.array(matched, dtype=int)
        extended = np.array(cat_target[field_extend])[indices[matched]]
        good_mod[matched] -= 2 * _source_is_type({field_extend: extended}, not resolved)
        plot_compure(mags_true_b[select_truth], good_mod[select_truth], postfix_title=postfix_title,
                     label_x=f'${band}_{{true}}$', **kwargs)


def _plot_purity(
    bands, cat_target, matched_truth, indices, select_truth, select_target, resolved, models,
    compare_mags_psf=True, compare_mags_lim_y=None, mag_zeropoint=None,
    **kwargs
):
    mag_max = kwargs.get('mag_max', np.Inf)
    if compare_mags_lim_y is None:
        compare_mags_lim_y = [-0.02, 0.02]
    has_psf = 'PSF' in models.keys()
    if compare_mags_psf and not has_psf:
        raise RuntimeError('Must include PSF model if comparing mags before purity plot')
    n_target = len(cat_target)
    cat_target = cat_target[select_target]
    print(f'Measuring purity for {len(cat_target)}/{n_target} sources')
    right_type = _source_is_type(cat_target, resolved)
    mags_psf = {}
    if has_psf:
        model = models['PSF']
        for band in bands:
            mags_psf[band] = model.get_mag_total(
                cat_target[right_type], band, zeropoint=mag_zeropoint,
            )

    matched = None
    for name, model in models.items():
        is_psf = name == 'PSF'
        do_psf_compare = compare_mags_psf and not is_psf
        for band in bands:
            if matched is None:
                matched = np.zeros(n_target)
                matched[indices[matched_truth & select_truth]] = 1
                matched[indices[matched_truth & ~select_truth]] = -1
                matched = matched[select_target]

            mags = mags_psf[band] if is_psf else model.get_mag_total(
                cat_target[right_type], band, zeropoint=mag_zeropoint,
            )
            within = mags < mag_max
            matched_right = matched[right_type]

            print(f'{np.sum(matched_right != 0)} matched and {np.sum(within)} mag < {mag_max};'
                  f' {np.sum((matched_right != 0) & within)} both'
                  f' and {np.sum(~np.isfinite(mags))} non-finite'
                  f'; {np.sum(right_type)}/len(cat)={len(cat_target)}')
            lim_x = plot_compure(mags[within], matched_right[within],
                                 label_x=f'${band}_{{{name}}}$', label_y='Purity', **kwargs)

            if do_psf_compare:
                mags_psf_band = mags_psf[band]
                within |= (mags_psf_band < mag_max)
                mags, mags_psf_band = mags[within].values, mags_psf_band[within].values
                mag_limit = mag_max + 1
                mags[~(mags < mag_limit)] = mag_limit
                mags_psf_band[~(mags_psf_band < mag_limit)] = mag_limit
                prefix = kwargs.get("prefix_title", "")
                if prefix:
                    prefix = f'{prefix} '
                plotjoint_running_percentiles(
                    x=mags_psf_band, y=mags - mags_psf_band, title=f'{prefix}N={np.sum(within)}',
                    labelx=f'${band}_{{PSF}}$', labely=f'${band}_{{{name}}}$-${band}_{{PSF}}$',
                    scatterleft=True, scatterright=True,
                    limx=lim_x, limy=compare_mags_lim_y,
                )


def plot_matches(
    cat_ref, cat_target, resolved, models, fluxes_true, select_target, centroids_ref=None,
    colors=None, mag_max=None, mag_max_compure=None, match_dist_asec=None, mag_bin_complete=0.1,
    models_diff=None, models_dist=None, models_purity=None,
    limits_x_color=None, limits_y_diff=None, limits_y_dist=None, limits_y_chi=None, limits_y_color_diff=None,
    plot_compure=True, plot_chi=False,
    compare_mags_psf_lim=None, mag_zeropoint_ref=None, title=None, kwargs_get_mag=None,
    **kwargs
):
    """ Make plots of completeness, purity and delta mag/colour for sources in two matched catalogs.

    Parameters
    ----------
    cat_ref: `pandas.DataFrame`
        Catalog of reference sources.
    cat_target: `pandas.DataFrame`
        Catalog of target (measured) sources.
    resolved: `bool`
        Is this plot for resolved objects (galaxies) or unresolved (stars, etc.)?
    models: `dict` [`str`, `modelling_research.meas_model.Model`]
        A dict of measurement models keyed by a short name useable as a label.
    fluxes_true: `dict` [`str`, `np.array`]
        A dict of true fluxes for the input catalog, keyed by band.
    colors : iterable [`tuple` [`str`, `str`]]
        A list of pairs of filters to plot colours for (filter1 - filter2).
    select_target : `np.array`
        Boolean array of target sources eligible for matching.
    centroids_ref : `dict` [`str`]
        Dict with x, y entries specifying column names for the centroids
        of reference sources. 
    mag_max : `float`
        The maximum (faintest) magnitude to filter data points.
    mag_max_compure : `float`
        The maximum (faintest) magnitude to plot compurity for.
    match_dist_asec : `float`
        The maximum distance for matches in arcseconds.
    mag_bin_complete : `float`
        The width of the magnitude bin to plot.
    models_diff : iterable of `str`
        A list of models to plot diffs for. Must all be in `models`.
    models_dist : iterable of `str`
        A list of models to plot distances for. Must all be in `models` and `models_diff`.
    models_purity : iterable of `str`
        A list of models to plot compurity for. Must all be in `models`.
    limits_x_color:
        x-axis limits for color plots.
    limits_y_diff : tuple `float`
        y-axis limits for diff plots.
    limits_y_dist : tuple `float`
        y-axis limits for dist plots.
    limits_y_chi:
        y-axis limits for chi plots.
    limits_y_color_diff : tuple `float`
        y-axis limits for diff plots with color x-axes.
    plot_compure : `bool`
        Whether to make plots of completeness and purity.
    plot_chi : `bool`
        Whether to make plots of chi (delta/error).
    mag_zeropoint_ref : `float`
        The zeropoint magnitude for the reference catalog, if needed.
    title : `str`
        A prefix string for all plot titles.
    compare_mags_psf_lim : `float`
        The y-axis (delta mag) limit for delta mag plots accompanying purity.
    kwargs_get_mag : `dict`
        Additional keyword arguments to pass to `get_mag_total` when retrieving
        target catalog magnitudes.
    **kwargs
        Additional keyword arguments to pass to `plotjoint_running_percentiles`.

    Returns
    -------
    select_truths : `dict`
        Dict of truth selection by tract, filled only if `return_select_truth` is True.
    """
    if mag_max is None:
        mag_max = np.Inf
    if mag_max_compure is None:
        mag_max_compure = mag_max
    if models_diff is None:
        models_diff = {}
    if models_dist is None:
        models_dist = {}
    if title is None:
        title = ""
    if kwargs_get_mag is None:
        kwargs_get_mag = {}

    models_purity = {name: models[name] for name in (models_purity if models_purity is not None else [])}
    obj_type = 'Resolved' if resolved else 'Unresolved'

    #cat_truth, cat_meas = cat_ref, cat_target
    select_truth = cat_ref['is_pointsource'] == (not resolved)
    indices = cat_ref['match_row']
    truth_matched = indices >= 0
    if plot_chi:
        sigmas_mag_band = {}

    if centroids_ref is None:
        if models_dist:
            raise ValueError('centroids_ref must be provided if models_dists is specified')
    else:
        x_true = cat_ref[centroids_ref['x']]
        y_true = cat_ref[centroids_ref['y']]

    bands = list(fluxes_true.keys())
    mags_true = {band: -2.5*np.log10(flux) + mag_zeropoint_ref for band, flux in fluxes_true.items()}

    args_plot = {
        'mag_max': mag_max_compure, 'mag_bin_complete': mag_bin_complete,
        'prefix_title': ' '.join((x for x in (title, obj_type, f'{match_dist_asec:.2f}"') if x)),
    }

    if plot_compure:
        _plot_completeness(mags_true, cat_target, truth_matched, indices,
                           select_truth, resolved, **args_plot)
        _plot_purity(
            bands, cat_target, truth_matched, indices,
            select_truth, select_target, resolved, models_purity,
            compare_mags_lim_y=compare_mags_psf_lim, mag_zeropoint=mag_zeropoint_ref,
            **args_plot
        )

    has_limy = 'limy' in kwargs
    limy = kwargs['limy'] if has_limy else None
    for name in models_diff:
        model = models[name]
        mags_band = {}
        plot_dist = (name in models_dist) and (centroids_ref is not None)

        for band in bands:
            matched = select_truth & truth_matched
            target = cat_target.iloc[indices[matched]]
            mags_ref_band = mags_true[band][matched]
            mags_target_band = model.get_mag_total(target, band, **kwargs_get_mag).values - mags_ref_band
            good = np.isfinite(mags_target_band)
            mags_band[band] = (mags_ref_band, mags_target_band)

            title_band = ' '.join((x for x in (title, f'{obj_type} {band}, {name}') if x))
            print(f'Plotting diffs: {title_band}')
            labelx = f'${band}_{{true}}$'
            plotjoint_running_percentiles(
                mags_ref_band[good], mags_target_band[good], title=f'{title_band}, N={np.sum(good)}',
                labelx=labelx, labely=f'${band}_{{model}}$ - {labelx}',
                limy=limits_y_diff,
                **kwargs
            )
            if plot_dist:
                plt.show()
                dx = model.get_cen(target, axis='x') - x_true[matched].values
                dy = model.get_cen(target, axis='y') - y_true[matched].values
                dists = np.hypot(dx, dy)
                good_cen = good & np.isfinite(dists)
                plotjoint_running_percentiles(
                    mags_ref_band[good_cen], dists[good_cen], title=f'{title_band}, N={np.sum(good_cen)}',
                    labelx=labelx, labely='Distance/pixels',
                    limy=limits_y_dist,
                    **kwargs
                )
            if plot_chi:
                plt.show()
                fluxes = fluxes_true[band][matched]
                sigma = model.get_flux_total(target, band, flux=f'{model.column_flux}Err').values
                sigmas_mag_band[band] = sigma/fluxes
                chi = (fluxes - model.get_flux_total(target, band).values)/sigma
                good_chi = np.isfinite(chi)
                limy = kwargs.get(limy)
                print(f'Plotting chis: {title_band}')
                plotjoint_running_percentiles(
                    mags_ref_band[good_chi], chi[good_chi],
                    title=f'{title_band}, N={np.sum(good_chi)}',
                    labelx=labelx, labely=f'chi=(data - model)/error ($flux_{{band}}$)',
                    limy=limits_y_chi,
                    **kwargs
                )
                if plot_dist:
                    for name_coord, coord in (('x', dx), ('y', dy)):
                        chi = coord/model.get_cen(target, axis=f'{name_coord}Err')
                        good_chi = good & np.isfinite(chi)
                        plotjoint_running_percentiles(
                            mags_ref_band[good_chi], chi[good_chi],
                            title=f'{title_band}, N={np.sum(good_chi)}',
                            labelx=labelx, labely=f'chi=(data - model)/error ({name_coord}/pix)',
                            limy=limits_y_chi,
                            **kwargs
                        )
            plt.show()
        if colors is None:
            colors = list(bands)
            colors = [(colors[idx], colors[idx + 1]) for idx in range(len(colors) - 1)]
        elif not colors:
            continue
        for b1, b2 in colors:
            labelx = f'${b1}_{{true}}$'
            x = mags_band[b1][0]
            y = mags_band[b1][1] - mags_band[b2][1]
            good = np.isfinite(y)
            band = f'{b1}-{b2}'
            title_band = ' '.join((x for x in (title, f'{obj_type} {band}, {name}') if x))
            print(f'Plotting color diffs: {title_band}')
            plotjoint_running_percentiles(
                x[good], y[good], title=f'{title_band}, N={np.sum(good)}',
                labelx=f'${b2}_{{true}}$', labely=f'${band}_{{model-true}}$',
                limy=limits_y_color_diff,
                **kwargs)
            plt.show()
            if plot_chi:
                sigma = 2.5/np.log(10)*np.hypot(sigmas_mag_band[b1], sigmas_mag_band[b2])
                chi = y/sigma
                good = np.isfinite(chi)
                limy = kwargs.get(limy)
                print(f'Plotting color chis: {title_band}')
                plotjoint_running_percentiles(
                    x[good], chi[good], title=f'{title_band}, N={np.sum(good)}',
                    labelx=labelx, labely=f'chi=(data - model)/error ({band})',
                    limy=limits_y_chi,
                    **kwargs
                )
                plt.show()
            if plot_dist:
                y = mags_band[b1][0] - mags_band[b2][0]
                labely = f'${band}_{{true}}$'
                limx = kwargs['limx']
                kwargs['limx'] = limits_x_color
                limy = (-limits_y_dist[1], limits_y_dist[1])
                good_col = np.isfinite(y)
                for axis, values in (('x', dx), ('y', dy)):
                    good = good_col & np.isfinite(values)
                    plotjoint_running_percentiles(
                        y[good], values[good], title=f'{title_band}, N={np.sum(good)}',
                        labelx=labely, labely=f'${axis}_{{model-true}}$/pixels',
                        limy=limy,
                        **kwargs
                    )
                    plt.show()
                good = good_col & np.isfinite(dists)
                plotjoint_running_percentiles(
                    y[good], dists[good], title=f'{title_band}, N={np.sum(good)}',
                    labelx=labely, labely=f'Distance/pixels',
                    limy=limits_y_dist,
                    **kwargs
                )
                plt.show()
                if plot_chi:
                    for axis, values in (('x', dx), ('y', dy)):
                        chi = values/model.get_cen(target, axis=f'{name_coord}Err')
                        good = good_col & np.isfinite(chi)
                        plotjoint_running_percentiles(
                            y[good], chi[good], title=f'{title_band}, N={np.sum(good)}',
                            labelx=labely, labely=f'chi=(data - model)/error ({axis}/pixels)',
                            limy=limits_y_chi,
                            **kwargs
                        )
                        plt.show()
                kwargs['limx'] = limx
    return select_truth
