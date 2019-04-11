import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _getname(name, prefix, postfix, separator=None):
    if prefix is None and postfix is None:
        return name
    if separator is None:
        separator = ''
    name = separator.join([prefix, name]) if prefix is not None else name
    name = separator.join([name, postfix]) if postfix is not None else name
    return name


def jointplotcolorbar(jointplot, label='', ticks=None, ticklabels=None,
                      labelsize='x-small', ticklabelsize='xx-small'):
    """
    Plots a colorbar in the top-right corner of a joint plot, which is normally empty.
    :param jointplot: A seaborn JointGrid.
    :param label: string, optional; a label (title) to place above the colorbar.
    :param ticks: List of floats, optional; values to place ticks on the colorbar.
    :param ticklabels: List of string, optional; tick labels.
    :param labelsize: Float, optional; font size for label.
    :param ticklabelsize: Float, optional; size of ticks.
    :return: matplotlib colorbar
    """
    plt.setp(jointplot.ax_marg_x.get_yticklabels(), visible=True)
    plt.setp(jointplot.ax_marg_y.get_xticklabels(), visible=True)
    #ax = jointplot.ax_marg_y
    with sns.axes_style("ticks"):
        cax = jointplot.fig.add_axes([.85, .85, .12, .05])
        cbar = plt.colorbar(cax=cax, orientation='horizontal', ticklocation='top')
        if ticks is not None:
            cbar.set_ticks(ticks)
            if ticklabels is not None:
                cbar.set_ticklabels(ticklabels)
        cbar.set_label(label, size=labelsize)
        cbar.ax.tick_params(labelsize=ticklabelsize)
        return cbar
    return None


def plotjoint(tab, columns, labels=None, columncolor=None, colorbaropts=None,
              logx=True, logy=True, logratio=True, limitsx=None, limitsy=None,
              plotmarginals=True, hist_kws={'log': False},
              prefixx=None, prefixy=None, postfixx=None, postfixy=None, separator=None,
              plotratiosjoint=True, bins=20, **kwargs):
    """
    Makes joint plots of pairs of columns in a table using Seaborn and with various options.
    :param tab: A table indexable by all of the columns.
    :param columns: Columns to make joint plots of, which must all be valid indices of tab. Must be strings
        if any other optional parameters like pre/postfixes are also strings.
    :param labels: Dict, optional; key column: value string label for the column. Default is str(column).
    :param columncolor: Column, optional; a valid column index of table to color-code points with.
    :param colorbaropts: Dict, optional; key arg: value to pass to jointplotcolorbar
    :param logx: Bool, optional; take log10 of x? Default False.
    :param logy: Bool, optional; take log10 of y? Default False.
    :param logratio: Bool, optional; take log10 of ratios? Default True.
    :param limitsx: float[2], optional; x-axis limits. Default None.
    :param limitsy: float[2], optional; y-axis limits. Default None.
    :param plotmarginals: Bool, optional; plot marginal histograms? Default True.
    :param hist_kws: Dict, optional; key arg:value to pass to marginal plots.
    :param prefixx: String, optional; prefix for x columns.
    :param prefixy: String, optional; prefix for y columns.
    :param postfixx: String, optional; postfix for x columns.
    :param postfixy: String, optional; postfix for y columns.
    :param separator: String, optional; separator between column name and pre/postfixes.
    :param plotratiosjoint: Bool, optional;
    :param bins:
    :return:
    """
    ratios = {
        var: tab[_getname(var, prefixy, postfixy, separator=separator)]/tab[
            _getname(var, prefixx, postfixx, separator=separator)]
        for var in columns
    }
    if labels is None:
        labels = {column: str(column) for column in columns}
    hascolor = columncolor is not None
    if hascolor:
        if colorbaropts is None:
            colorbaropts = {}
        color = tab[columncolor]
    else:
        color = None
    hasprefixes = prefixx is not None or prefixy is not None
    haspostfixes = postfixx is not None or postfixy is not None
    hasfixes = hasprefixes or haspostfixes
    joints = []
    for i, columnx in enumerate(columns):
        x = np.array(ratios[columnx])
        if logx:
            x = np.log10(x, out=x)
        isfinitex = np.isfinite(x)
        if limitsx is not None:
            x = np.clip(x, limitsx[0], limitsx[1], out=x)
        if plotratiosjoint:
            labelaxis = ('log10' if logratio else '') + '{} ratio'
            labeldenom = '' if not hasfixes else separator.join(
                ([prefixy] if prefixy is not None else []) + ([postfixy] if postfixy is not None else []))
            for columny in columns[(i+1):(len(columns)+1)]:
                labelpostfix = '' if not hasfixes else (
                    ' (' + labeldenom + '/' + separator.join(
                        ([prefixx] if prefixx is not None else []) +
                        ([postfixx] if postfixx is not None else [])) +
                    ')')
                labelaxisx = labelaxis.format(labels[columnx]) + labelpostfix
                labelaxisy = labelaxis.format(labels[columny]) + labelpostfix
                y = np.array(ratios[columny])
                if logy:
                    y = np.log10(y, out=y)
                isfinite = isfinitex & np.isfinite(y)
                if limitsy is not None:
                    y = np.clip(y, limitsy[0], limitsy[1], out=y)
                fig = sns.JointGrid(x=x[isfinite], y=y[isfinite])
                colorxy = color[isfinite] if color is not None else None
                jointplot = fig.plot_joint(plt.scatter, c=colorxy, zorder=2, **kwargs).set_axis_labels(
                    labelaxisx, labelaxisy)
                x0, x1 = jointplot.ax_joint.get_xlim()
                y0, y1 = jointplot.ax_joint.get_ylim()
                lims = [max(x0, y0), min(x1, y1)]
                jointplot.ax_joint.plot(lims, lims, '-k', zorder=1)
                if hascolor:
                    jointplotcolorbar(jointplot, **colorbaropts)
                if plotmarginals:
                    fig.plot_marginals(sns.distplot, kde=False, hist_kws=hist_kws)

        y = np.array(tab[_getname(columnx, prefixx, postfixx, separator=separator)])
        if logy:
            y = np.log10(y, out=y)
        isfinite = isfinitex & np.isfinite(y)
        if limitsy is not None:
            y = np.clip(y, limitsy[0], limitsy[1], out=y)
        fig = sns.JointGrid(x=y[isfinite], y=x[isfinite])
        colorxy = color[isfinite] if color is not None else None
        labelprefix = 'log10' if logx else ''
        jointplot = fig.plot_joint(plt.scatter, c=colorxy, zorder=2, **kwargs).set_axis_labels(
                labelprefix + '({}) ({})'.format(columnx, prefixx),
                labelprefix + '({} ratio) ({}/{})'.format(columnx, prefixy, prefixx))
        x0, x1 = jointplot.ax_joint.get_xlim()
        jointplot.ax_joint.plot([x0, x1], [0, 0], '-k', zorder=1)
        if hascolor:
            jointplotcolorbar(jointplot, **colorbaropts)
        if plotmarginals:
            jointgrid = fig.plot_marginals(sns.distplot, kde=False, hist_kws=hist_kws, bins=bins)
        joints.append(jointplot)
    return joints
