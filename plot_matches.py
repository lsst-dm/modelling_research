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
    ax.tick_params(which='y', direction='out', length=6, width=2, colors='k')
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


def _plot_completeness(mags_true, cats_meas, good, indices, select_truth, resolved, postfix_title,
                       field=None, **kwargs):
    if field is None:
        field = 'base_ClassificationExtendedness_value'
    for band, mags_true_b in mags_true.items():
        good_mod = np.array(good, dtype=int)
        extend = np.array(cats_meas[band][field])[indices[select_truth][good]]
        good_mod[good] -= 2 * _source_is_type({field: extend}, not resolved)
        plot_compure(mags_true_b[select_truth], good_mod, postfix_title=postfix_title,
                     label_x=f'${band}_{{true}}$', **kwargs)


def _plot_purity(
        models, cats, resolved, indices, select_truth, compare_mags_psf=True, compare_mags_lim_y=None,
        **kwargs
):
    mag_max = kwargs.get('mag_max', np.Inf)
    if compare_mags_lim_y is None:
        compare_mags_lim_y = [-0.02, 0.02]
    has_psf = 'PSF' in models.keys()
    if compare_mags_psf and not has_psf:
        raise RuntimeError('Must include PSF model if comparing mags before purity plot')
    right_types = {
        band: _source_is_type(cat, resolved) & ~cat['merge_footprint_sky']
        for band, cat in cats.items()
    }
    mags_psf = {}
    if has_psf:
        model = models['PSF']
        for band, cat in cats.items():
            mags_psf[band] = model.get_mag_total(cat[right_types[band]], band)

    matched = None
    for name, model in models.items():
        is_psf = name == 'PSF'
        do_psf_compare = compare_mags_psf and not is_psf
        for band, cat in cats.items():
            right_type = right_types[band]
            if matched is None:
                matched = np.zeros(len(cat))
                matched_any = indices >= 0
                matched[indices[matched_any & select_truth]] = 1
                matched[indices[matched_any & ~select_truth]] = -1

            mags = mags_psf[band] if is_psf else model.get_mag_total(cat[right_type], band)
            within = mags < mag_max
            matched_right = matched[right_type]

            print(f'{np.sum(matched_right != 0)} matched and {np.sum(within)} mag < {mag_max};'
                  f' {np.sum((matched_right != 0) & within)} both'
                  f' and {np.sum(~np.isfinite(mags))} non-finite; {np.sum(right_type)}/len(cat)={len(cat)}')
            lim_x = plot_compure(mags[within], matched_right[within],
                                 label_x=f'${band}_{{{name}}}$', label_y='Purity', **kwargs)

            if do_psf_compare:
                mags_psf_band = mags_psf[band]
                within |= (mags_psf_band < mag_max)
                mags, mags_psf_band = mags[within], mags_psf_band[within]
                mag_limit = mag_max + 1
                mags[~(mags < mag_limit)] = mag_limit
                mags_psf_band[~(mags_psf_band < mag_limit)] = mag_limit
                prefix = kwargs.get("prefix_title", "")
                if prefix:
                    prefix = f'{prefix} '
                plotjoint_running_percentiles(
                    mags_psf_band, mags - mags_psf_band, title=f'{prefix}N={np.sum(within)}',
                    labelx=f'${band}_{{PSF}}$', labely=f'${band}_{{{name}}}$-${band}_{{PSF}}$',
                    scatterleft=True, scatterright=True,
                    limx=lim_x, limy=compare_mags_lim_y
                )


def _rematch(indices, indices2, within, good, select_truth):
    # Truths that matched to the right type of true object
    right_type = select_truth[indices2]
    # Truths matched within the threshold distance
    rematched = within & right_type
    # Index in cat_truth of the truths matched
    idx_rematch = np.cumsum(select_truth)[indices2[rematched]] - 1
    print(f'Rematched {np.sum(~good[idx_rematch])}/{len(good)}; {np.sum(good)} originally '
          f'and {np.sum(good[idx_rematch])} rematched already matched')
    # Is the good true match supposed to be rematched because it wasn't marked good yet?
    to_rematch = ~good[idx_rematch]
    # The np.where is because we want the indices of the measurements, which is what indices is
    # TODO: Make sure this is actually correct
    indices[idx_rematch[to_rematch]] = np.where(rematched)[0][to_rematch]
    good[idx_rematch[to_rematch]] = True


def plot_matches(
        cats, resolved, models, bands=None, band_ref=None, band_multi=None, band_ref_multi=None,
        colors=None, mag_max=None, mag_max_compure=None, match_dist_asec=None, mag_bin_complete=0.1,
        rematch=False, models_purity=None, plot_compure=True, plot_diffs=True, compare_mags_psf_lim=None,
        return_select_truth=False,
        **kwargs
):
    """ Make plots of completeness, purity and delta mag/colour for sources in two matched catalogs.

    Parameters
    ----------
    cats: `dict`
        A dictionary with entries as returned by dc2.match_refcat.
    resolved: `bool`
        Is this plot for resolved objects (galaxies) or unresolved (stars, etc.)?
    models: `dict` [`str`, `modelling_research.meas_model.Model`]
        A dict of measurement models keyed by a short name useable as a label.
    bands: iterable of `str`
        An iterable of filter names.
    band_ref : `str`
        The reference filter that was used for matching.
    band_multi : `str`
        The name of the multi-band fit, if any.
    band_ref_multi : `str`
        The reference filter to use on the x-axis to colour plots.
    colors : iterable [`tuple` [`str`, `str`]]
        A list of pairs of filters to plot colours for (filter1 - filter2).
    mag_max : `float`
        The maximum (faintest) magnitude to filter data points.
    mag_max_compure : `float`
        The maximum (faintest) magnitude to plot compurity for.
    match_dist_asec : `float`
        The maximum distance for matches in arcseconds.
    mag_bin_complete : `float`
        The width of the magnitude bin to plot.
    rematch : `bool`
        Whether to attempt to rematch multiple matches to their nearest neighbour, if still unmatched.
    models_purity : iterable of `str`
        A list of models to plot compurity for. Must all be in `models`.
    plot_diffs : `bool`
        Whether to plot delta mag/colour vs mag in addition to compurity.
    compare_mags_psf_lim : `float`
        The y-axis (delta mag) limit for delta mag plots accompanying purity.
    return_select_truth : `bool`
        Whether to return the truth selection array.
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

    models_purity = {name: models[name] for name in (models_purity if models_purity is not None else [])}
    bands_is_none = bands is None
    has_multi = band_multi is not None
    obj_type = 'Resolved' if resolved else 'Unresolved'
    is_afw = match_dist_asec is None
    select_truths = {}

    for tract, cats_type in cats.items():
        cat_truth, cats_meas = cats_type['truth'], cats_type['meas']
        select_truth = (cat_truth['id'] > 0) == resolved

        if has_multi:
            cat_multi = cats_meas[band_multi]
            if is_afw:
                cat_multi = cat_multi[select_truth]
                cat_truth = cat_truth[select_truth]

        if bands_is_none:
            bands = cats_meas.keys()
            bands = (band for band in bands if band != band_multi)
        else:
            cats_meas = {band: cats_meas[band] for band in bands}

        if band_ref is None:
            band_ref = bands[0]

        mags_true = {band: -2.5 * np.log10(cat_truth[f'lsst_{band}_flux']) + 31.4 for band in bands}
        good_mags_true = {band: mags_true[band] < mag_max for band in bands}

        if is_afw:
            cats_meas = {band: cat[select_truth] if is_afw else cat for band, cat in cats_meas.items()}
        else:
            indices, dists = (cats_type[x] for x in ('indices1', 'dists1'))
            # Cheat a little and set negatives to -1
            indices = np.copy(indices)
            indices[dists > match_dist_asec] = -1
            # bincount only works on non-negative integers, but we want to preserve the true indices and
            # don't need the total count of unmatched sources
            n_matches = np.bincount(indices + 1)[1:]
            matches_multi = n_matches > 1
            mags_true_ref = mags_true[band_ref]

            # set multiple matches to integers < -1
            for idx in np.where(matches_multi)[0]:
                matches = np.where(indices == idx)[0]
                brightest = np.argmax(mags_true_ref[matches])
                indices[matches] = -idx - 2
                indices[matches[brightest]] = idx

            good = indices[select_truth] >= 0

            args_plot = {
                'mag_max': mag_max_compure, 'mag_bin_complete': mag_bin_complete,
                'prefix_title': f'DC2 {tract} {obj_type} {match_dist_asec:.2f}asec',
                'postfix_title': ' !rematch',
            }

            print(f"N={np.sum(cats_meas[band_ref]['merge_footprint_sky'][indices])} sky object matches")

            # This took hours and caused much regret
            if rematch:
                if plot_compure:
                    _plot_completeness(mags_true, cats_meas, good, indices, select_truth, resolved,
                                       **args_plot)
                    _plot_purity(
                        models_purity, {band: cats_type['meas'][band] for band in bands}, resolved,
                        indices, select_truth, compare_mags_lim_y=compare_mags_psf_lim, **args_plot
                    )

                args_plot['postfix_title'] = ' rematch'
                indices2, dists2 = (cats_type[x] for x in ('indices2', 'dists2'))
                _rematch(indices, indices2, dists2 < match_dist_asec, good, select_truth)
                print(f"N={np.sum(cats_meas[band_ref]['merge_footprint_sky'][indices])}"
                      f"sky object matches after rematching")

            if plot_compure:
                _plot_completeness(mags_true, cats_meas, good, indices, select_truth, resolved, **args_plot)
                _plot_purity(
                    models_purity, {band: cats_type['meas'][band] for band in bands}, resolved, indices,
                    select_truth, compare_mags_lim_y=compare_mags_psf_lim, **args_plot
                )

            if plot_diffs:
                cat_truth, indices = (x[select_truth][good] for x in (cat_truth, indices))
                cats_meas = {band: cat.copy(deep=True).asAstropy()[indices]
                             for band, cat in cats_meas.items()}
                mags_true = {band: -2.5 * np.log10(cat_truth[f'lsst_{band}_flux']) + 31.4 for band in bands}
                good_mags_true = {band: mags_true[band] < mag_max for band in bands}
                if has_multi:
                    cat_multi = cat_multi.copy(deep=True).asAstropy()[indices]

        for name, model in models.items() if plot_diffs else {}:
            cats_type = [(cats_meas, False)]
            if band_multi is not None and model.multiband:
                cats_type.append((cat_multi, True))
            mags_diff = {}
            for band, good_band in good_mags_true.items():
                mags_diff[band] = {}
                true = mags_true[band]
                for cat, multi in cats_type:
                    y = model.get_mag_total(cat if multi else cat[band], band) - true
                    mags_diff[band][multi] = y
                    x, y = true[good_band], y[good_band]
                    good = np.isfinite(y)
                    postfix = f'({band_multi})' if multi else ''
                    title = f'DC2 {tract} {obj_type} {band}-band, {name}{postfix}, N={np.sum(good)}'
                    print(title)
                    labelx = f'${band}_{{true}}$'
                    plotjoint_running_percentiles(
                        x[good], y[good], title=title,
                        labelx=labelx, labely=f'${band}_{{model}}$ - {labelx}',
                        **kwargs
                    )
                    plt.show()
            if colors is None:
                colors = list(bands)
                colors = [(colors[idx], colors[idx + 1]) for idx in range(len(colors) - 1)]
            elif not colors:
                continue
            for b1, b2 in colors:
                bx = b2 if band_ref_multi is None else band_ref_multi
                good_band = good_mags_true[bx]
                for _, multi in cats_type:
                    x = mags_true[bx][good_band]
                    y = mags_diff[b1][multi][good_band] - mags_diff[b2][multi][good_band]
                    good = np.isfinite(y)
                    band = f'{b1}-{b2}'
                    postfix = f'({band_multi})' if multi else ''
                    title = f'DC2 {tract} {obj_type} {band}, {name}{postfix}, N={np.sum(good)}'
                    print(title)
                    plotjoint_running_percentiles(
                        x[good], y[good], title=title,
                        labelx=f'${bx}_{{true}}$', labely=f'${band}_{{model-true}}$',
                        **kwargs)
                    plt.show()
        if return_select_truth:
            select_truths[tract] = select_truth
    return select_truths


def _source_is_type(cat, resolved, include_nan=False, threshold=0.5):
    if include_nan:
        return ~_source_is_type(cat, not resolved, threshold=threshold)
    else:
        return (cat[
                    'base_ClassificationExtendedness_value'] >= threshold) if resolved else (
                cat['base_ClassificationExtendedness_value'] < threshold)
