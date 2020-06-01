import matplotlib.pyplot as plt
from modelling_research.plotting import plotjoint_running_percentiles
from multiprofit.gaussutils import sigma_to_reff
from multiprofit.utils import flux_to_mag, mag_to_flux
import numpy as np
import seaborn as sns
import traceback

names_ellipse = ('reff', 'axrat', 'ang')
names_optional_comp = ("nser",)


def assign_field(datum, field, name):
    if field is not None:
        datum[name] = field


def assign_fields(datum, fields, names, postfix=''):
    for field, name in zip(fields, names):
        assign_field(datum, field, f'{name}{postfix}')


def is_band_hst(band_short):
    return band_short.startswith('f')


def get_bands(bands_short):
    if is_band_hst(bands_short):
        return {bands_short: bands_short.upper()}
    else:
        return {band: f'HSC-{band.upper()}' for band in bands_short}


def get_field(cat, field, log=False):
    if field in cat.columns:
        return cat[field]
    if log:
        print(f"Didn't find field {field}")
    return None


def reduce_cat(cat, name_cat, scale_dist, is_single, models, field_flux='flux', flags_bad=None,
               flags_good=None, has_mags=True, log=False, add_prereq_time=True, names_optional=None):
    if flags_bad is None:
        flags_bad = {"base_PixelFlags_flag_saturatedCenter", "base_PixelFlags_flag_sensor_edgeCenter",
                     "deblend_tooManyPeaks", "modelfit_CModel_flag"}
    if flags_good is None:
        flags_good = {"detect_isPatchInner"}
    if names_optional is None:
        names_optional = set()
    bands_fit = name_cat.split('_unmatched')[0]
    datum_bands = {}
    sigma2reff = sigma_to_reff(1)
    scale_reff = scale_dist * sigma2reff
    bands = get_bands(bands_fit)

    for name_model, algos in models.items():
        is_cmodel = name_model == "cmodel"
        is_psf = name_model == "psf"
        is_gauss = name_model == "gauss"
        is_gauss_no_psf = name_model == 'gauss_no_psf'
        datum_model = {}
        for algo, meas_model in algos.items():
            n_comps = meas_model.n_comps
            is_base = algo == "base"
            is_mmf = algo == "mmf"
            is_mpf = algo == "mpf"
            is_gauss_no_psf_base = is_gauss_no_psf and is_base
            datum = {}

            # gauss_no_psf only works in single band for now
            if is_single or (is_mpf and not (is_gauss_no_psf and (len(bands) > 1))):
                for band_short, band_full in bands.items():
                    if is_psf:
                        for comp in range(is_mpf, n_comps + is_mpf):
                            reff, axrat, ang = meas_model.get_ellipse_terms(cat, band=band_full, comp=comp)
                            assign_fields(datum, (np.log10(scale_dist * reff), axrat, ang), names_ellipse,
                                          postfix=f'{comp}')
                            for name_item in names_optional_comp:
                                name_src = f'{meas_model.get_field_prefix(band_full, comp=comp)}_{name_item}'
                                name_out = f'{name_item}_{comp}_{band_short}'
                                assign_field(datum, get_field(cat, name_src), name_out)
                        for name_item in names_optional:
                            name_src = get_field(cat, f'{meas_model.get_field_prefix(band_full)}_{name_item}')
                            assign_field(datum, name_src, f'{name_item}_{band_short}')
                    else:
                        datum[f'flux_{band_short}'] = meas_model.get_flux_total(cat, band=band_full,
                                                                                flux=field_flux)
                        if is_cmodel and is_mmf:
                            datum[f"fracDev_{band_short}"] = cat[
                                f'{meas_model.get_field_prefix(band_full)}_fracDev']
                        if is_cmodel and is_mpf:
                            # Could do it for HST but not going to bother
                            if has_mags:
                                flux_exp = mag_to_flux(meas_model.get_mag_comp(cat, band=band_full, comp=1))
                                flux_dev = mag_to_flux(meas_model.get_mag_comp(cat, band=band_full, comp=2))
                                mag_c = flux_to_mag(flux_dev + flux_exp)
                                datum[f'mag_{band_short}'] = mag_c
                                datum[f"fracDev_{band_short}"] = flux_dev / (flux_dev + flux_exp)
                        elif has_mags:
                            datum[f'mag_{band_short}'] = meas_model.get_mag_total(cat, band=band_full)
                if not is_psf:
                    if not (is_cmodel or (is_base and is_gauss)):
                        for comp in range(1, n_comps + 1):
                            reff, axrat, ang = meas_model.get_ellipse_terms(cat, comp=comp)
                            assign_fields(datum, (
                            np.log10((scale_reff if is_gauss_no_psf_base else scale_dist) * reff), axrat,
                            ang), names_ellipse)
                            for name_item in names_optional_comp:
                                name_src = f'{meas_model.get_field_prefix("", comp=comp)}_{name_item}'
                                assign_field(datum, get_field(cat, name_src), f'{name_item}_{comp}')
                        for ax in ('x', 'y'):
                            assign_field(datum, meas_model.get_cen(cat, axis=ax, comp=1), f'cen{ax}')
                    for name_item in names_optional:
                        name_src = f'multiprofit_measmodel_like_{name_model}' if (
                                    is_mmf and name_item == 'loglike') else \
                            f'{meas_model.get_field_prefix("")}_{name_item}'
                        assign_field(datum, get_field(cat, name_src), f'{name_item}')
            if log:
                print(algo, name_model, datum.keys())
            if datum:
                good = None
                for flags, is_bad in ((flags_bad, True), (flags_good, False)):
                    for flag in flags:
                        good_new = ~cat[flag] if is_bad else cat[flag]
                        if good is None:
                            good = good_new
                        else:
                            good &= good_new
                        n_good = np.sum(good)
                        if not n_good > 0:
                            raise RuntimeError(f'Found {n_good}/{len(good)} after flag {flag}')
                datum['good'] = good
                datum_model[algo] = datum
        datum_bands[name_model] = datum_model
    # Some models depend on other models and so their fitting time should include prereqs
    # TODO: This should be recursive
    if add_prereq_time:
        models_prereqs = {
            'mg8serb': ('gauss', 'exp', 'dev'),
            'cmodel': ('gauss', 'exp', 'dev'),
        }
        for model, prereqs in models_prereqs.items():
            if model in datum_bands:
                for algo in datum_bands[model]:
                    # mpf CModel depends on the exp that depends on gauss
                    # mmf CModel does not (there is no mmf gauss) but it does have an initial fit that isn't included yet
                    prereqs_algo = [prereq for prereq in prereqs if
                                    'time' in datum_bands[prereq].get(algo, {})]
                    if prereqs_algo:
                        has_base = 'time' in datum_bands[model][algo]
                        if not has_base:
                            datum_bands[model][algo]['time'] = np.copy(
                                datum_bands[prereqs_algo[0]][algo]['time'])
                        for prereq in prereqs_algo[has_base:]:
                            if prereq in datum_bands:
                                datum_bands[model][algo]['time'] += datum_bands[prereq][algo]['time']
    return datum_bands


# Define functions for plotting parameter values in dicts (not the original tables)
def get_columns_info(column_info, name_plot, labels=None):
    if labels is None:
        labels = {}
    column = column_info.get('column', name_plot)
    postfix = column_info.get('postfix', '')
    x = column_info.get('column_x', column)
    column_x = f"{x}{column_info.get('postfix_x', postfix)}"
    name_column_x = labels.get(x, x)
    datum_idx_x = column_info.get('datum_idx_x', 0)
    datum_idx_y = column_info.get('datum_idx_y', 1)
    y = column_info.get("column_y", x)
    plot_cumulative = column_info.get("plot_cumulative", False)
    column_y = column_x if y is x else f"{y}{column_info.get('postfix_y', postfix)}"
    name_column_y = labels.get(y, name_column_x if y is x else y)
    return column_x, column_y, name_column_x, name_column_y, datum_idx_x, datum_idx_y, plot_cumulative


def plot_column_pair(
        x, y, cond, column_info, name_column_x, name_column_y, label_x, label_y,
        algo_x, algo_y, model_x, model_y, band, argspj=None, units=None, title=None,
        cumulative=False, title_cumulative=None, show=True
):
    if argspj is None:
        argspj = {}
    if units is None:
        units = {}
    is_log = column_info.get('log', False)
    is_log_x = column_info.get('log_x', is_log)
    is_log_y = column_info.get('log_y', is_log)
    is_ratio = column_info.get('ratio', False)
    is_difference = column_info.get('difference', False)
    is_combo = is_ratio or is_difference
    crop_x = column_info.get('crop_x', False)
    crop_y = column_info.get('crop_y', False)
    y_plot = y
    if is_difference:
        y_plot = y_plot - x
    elif is_ratio:
        y_plot = y / x
    if is_log_x:
        x = np.log10(x)
    if is_log_y:
        y_plot = np.log10(y_plot)
    unit_x = units.get(name_column_x, None)
    unit_x_fmt = f' ({unit_x})' if unit_x is not None else ''
    unit_y = units.get(name_column_y, None)
    unit_y_fmt = f" ({unit_y})" if (not is_ratio and name_column_y in units) else ''
    good = cond & np.isfinite(x) & np.isfinite(y)
    if name_column_x == "reff":
        good = good & (x > -1.8)
    lim_x = column_info.get('limx', (0, 3))
    lim_y = column_info.get('limy', (-1, 1))
    if crop_x:
        good = good & (x > lim_x[0]) & (x < lim_x[1])
    if crop_y:
        good = good & (y_plot > lim_y[0]) & (y_plot < lim_y[1])
    prefix = "log10 " if is_log else ""
    postfix_x = f" [{algo_x} {model_x}, {band}-band]{unit_x_fmt}"
    middle_y = f" {'/' if is_ratio else '-'} {algo_x} {model_x}" if is_combo else ""
    postfix_y = f" [{algo_y} {model_y}{middle_y}]{unit_y_fmt}"
    label_x = f"{prefix}{label_x}{postfix_x}"
    x_good, y_good = (ax[good] for ax in [x, y_plot])
    plotjoint_running_percentiles(
        x_good, y_good, **argspj,
        labelx=label_x, labely=f"{prefix}{label_y}{postfix_y}",
        title=title,
        limx=lim_x, limy=lim_y,
    )
    if show:
        plt.show(block=False)
    if cumulative:
        x_plot = [(np.sort(x_good), is_log_x, f'{algo_x} {model_x}, {band}-band')]
        plot_y = unit_x == unit_y
        if plot_y:
            if is_difference or is_ratio:
                y_plot = np.log10(y[good]) if is_log_y else y[good]
            x_plot.append((np.sort(y_plot), is_log_y, f'{algo_y} {model_y}, {band}-band'))
        y_max = 0
        for x_cumul, is_log, label in x_plot:
            y_cumul = np.cumsum(10 ** x_cumul if is_log else x_cumul)
            y_max = np.nanmax([y_max, y_cumul[-1]])
            postfix_label = ''
            # Clip slightly before lim_x[1] so that it plots nicely at the edge if it needs to be clipped
            x_max = lim_x[1] - 1e-3 * (lim_x[1] - lim_x[0])
            if x_cumul[-1] > x_max:
                idx = np.searchsorted(x_cumul, x_max)
                x_cumul[idx] = x_max
                y_cumul[idx] = y_cumul[-1]
                idx = idx + 1
                x_cumul = x_cumul[0:idx]
                y_cumul = y_cumul[0:idx]
                postfix_label = ' (clipped)'
            sns.lineplot(x=x_cumul, y=y_cumul, label=f'{label}{postfix_label}', ci=None)
        if plot_y:
            plt.legend()
        plt.xlim(lim_x)
        plt.ylim([0, y_max])
        plt.xlabel(label_x)
        plt.ylabel(f'Cumulative {label_y} ({unit_x})')
        if title_cumulative is not None:
            plt.title(title_cumulative)
        if show:
            plt.show(block=False)


def plot_models(data, band, algos, columns_plot, columns_plot_size, models=None,
                labels=None, units=None, argspj=None):
    if argspj is None:
        argspj = {}
    if labels is None:
        labels = {}
    if units is None:
        units = {}
    if models is None:
        models = ["exp", "dev", "cmodel"]
    data_band = data[band]
    for model in models:
        is_single_comp = model != "cmodel"
        data_model = data_band[model]
        data_algos = [data_model[algo] for algo in algos]
        data_cond = data_algos[0]
        cond = (data_cond[f'mag_i'] < 29) & (data_cond['good'])
        title = f'N={np.count_nonzero(cond)}'
        for name_plot, column_info in (columns_plot_size if is_single_comp else columns_plot).items():
            print(f"Plotting model {model} plot {name_plot}")
            column_x, column_y, name_column_x, name_column_y, datum_idx_x, datum_idx_y, plot_cumulative = \
                get_columns_info(column_info, name_plot, labels=labels)
            try:
                x = data_algos[datum_idx_x][column_x]
                y = data_algos[datum_idx_y][column_y]
                plot_column_pair(
                    x, y, cond, column_info,
                    column_x, column_y, name_column_x, name_column_y,
                    algos[datum_idx_x], algos[datum_idx_y], model, model, band,
                    units=units, title=title, cumulative=plot_cumulative,
                    title_cumulative=title if plot_cumulative else None, argspj=argspj
                )
            except Exception as e:
                data_model_name = f"data['{band}']['{model}']"
                print(f"Failed to read {data_model_name}['{algos[datum_idx_x]}']['{column_x}'] and/or "
                      f"{data_model_name}['{algos[datum_idx_y]}']['{column_y}'] "
                      f"due to {getattr(e, 'message', repr(e))}")
                traceback.print_exc()


def plot_models_algo(data, band, algo, models, columns_plot, columns_plot_size,
                     labels=None, units=None, argspj=None):
    if argspj is None:
        argspj = {}
    if labels is None:
        labels = {}
    if units is None:
        units = {}
    data_band = data[band]
    data_models = [data_band[model] for model in models]
    is_single_comp = all([model != "cmodel" for model in models])
    data_algos = [data_model[algo] for data_model in data_models]
    cond = (data_algos[0][f'mag_i'] < 29) & (data_algos[0]['good'])
    title = f'N={np.count_nonzero(cond)}'
    for name_plot, column_info in (columns_plot_size if is_single_comp else columns_plot).items():
        print(f"Plotting models {models} plot {name_plot}")
        column_x, column_y, name_column_x, name_column_y, datum_idx_x, datum_idx_y, plot_cumulative = \
            get_columns_info(column_info, name_plot, labels=labels)
        title_cumul = title if plot_cumulative else None
        try:
            if column_x == column_y:
                x = data_algos[0][column_x]
                y = data_algos[1][column_y]
                plot_column_pair(
                    x, y, cond, column_info,
                    column_x, column_y, name_column_x, name_column_y,
                    algo, algo, models[0], models[1], band,
                    units=units, title=title, cumulative=plot_cumulative,
                    title_cumulative=title_cumul, argspj=argspj,
                )
            else:
                for idx in range(2):
                    datum_algo = data_algos[idx]
                    model = models[idx]
                    plot_column_pair(
                        datum_algo[column_x], datum_algo[column_y], cond, column_info,
                        column_x, column_y, name_column_x, name_column_y,
                        algo, algo, model, model, band,
                        units=units, title=title, cumulative=plot_cumulative,
                        title_cumulative=title_cumul, argspj=argspj,
                    )
        except Exception as e:
            data_model_names = [f"data['{band}']['{model}']" for model in models]
            print(f"Failed to read {data_model_names[0]}['{algo}']['{column_x}'] and/or "
                  f"{data_model_names[1]}['{algo}']['{column_y}'] "
                  f"due to {getattr(e, 'message', repr(e))}")
            traceback.print_exc()