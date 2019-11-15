import matplotlib as mpl
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


def colorbarinset(jointplot, label='', ticks=None, ticklabels=None, labelsize='x-small',
                  ticklabelsize='xx-small', position=None):
    """
    Plots a colorbar in the top-right corner of a joint plot, which is normally empty.
    :param jointplot: A seaborn JointGrid.
    :param label: string, optional; a label (title) to place above the colorbar.
    :param ticks: List of floats, optional; values to place ticks on the colorbar.
    :param ticklabels: List of string, optional; tick labels.
    :param labelsize: Float, optional; font size for label.
    :param ticklabelsize: Float, optional; size of ticks.
    :param position: Float[4], optional; position as a fraction of image size in pyplot units of
        [left, bottom, width, height]; default [.85, .85, .12, .05].
    :return: matplotlib colorbar
    """
    if position is None:
        position = [.85, .85, .12, .05]
    plt.setp(jointplot.ax_marg_x.get_yticklabels(), visible=True)
    plt.setp(jointplot.ax_marg_y.get_xticklabels(), visible=True)
    #ax = jointplot.ax_marg_y
    with sns.axes_style("ticks"):
        cax = jointplot.fig.add_axes(position)
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
              logx=True, logy=True, logratio=True, plotmarginals=True, hist_kws={'log': False},
              prefixx=None, prefixy=None, postfixx=None, postfixy=None, separator=None,
              plotratiosjoint=True, limitsxratio=None, limitsyratio=None, bins=20, **kwargs):
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
    :param plotmarginals: Bool, optional; plot marginal histograms? Default True.
    :param hist_kws: Dict, optional; key arg:value to pass to marginal plots.
    :param prefixx: String, optional; prefix for x columns.
    :param prefixy: String, optional; prefix for y columns.
    :param postfixx: String, optional; postfix for x columns.
    :param postfixy: String, optional; postfix for y columns.
    :param separator: String, optional; separator between column name and pre/postfixes.
    :param plotratiosjoint: Bool, optional; make plots
    :param limitsxratio: float[2], optional; x-axis limits for ratio plots. Default None.
    :param limitsyratio: float[2], optional; y-axis limits for ratio plots. Default None.
    :param bins: Int, optional; number of bins for marginal histograms. Default 20.
    :return: jointplot, jointgrids; lists of jointplot (and jointgrid if plotratiosjoint) handles.
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
    jointplots = []
    jointplotsratio = []
    for i, columnx in enumerate(columns):
        x = np.array(ratios[columnx])
        if logx:
            x = np.log10(x, out=x)
        isfinitex = np.isfinite(x)
        if limitsxratio is not None:
            x = np.clip(x, limitsxratio[0], limitsxratio[1], out=x)
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
                if limitsyratio is not None:
                    y = np.clip(y, limitsyratio[0], limitsyratio[1], out=y)
                fig = sns.JointGrid(x=x[isfinite], y=y[isfinite])
                colorxy = color[isfinite] if color is not None else None
                jointplot = fig.plot_joint(plt.scatter, c=colorxy, zorder=2, **kwargs).set_axis_labels(
                    labelaxisx, labelaxisy)
                x0, x1 = jointplot.ax_joint.get_xlim()
                y0, y1 = jointplot.ax_joint.get_ylim()
                lims = [max(x0, y0), min(x1, y1)]
                jointplot.ax_joint.plot(lims, lims, '-k', zorder=1)
                if hascolor:
                    colorbarinset(jointplot, **colorbaropts)
                if plotmarginals:
                    fig.plot_marginals(sns.distplot, kde=False, hist_kws=hist_kws)
                jointplotsratio.append(jointplot)
        y = np.array(tab[_getname(columnx, prefixx, postfixx, separator=separator)])
        if logy:
            y = np.log10(y, out=y)
        isfinite = isfinitex & np.isfinite(y)
        fig = sns.JointGrid(x=y[isfinite], y=x[isfinite])
        colorxy = color[isfinite] if color is not None else None
        labelprefix = 'log10' if logx else ''
        jointplot = fig.plot_joint(plt.scatter, c=colorxy, zorder=2, **kwargs).set_axis_labels(
                labelprefix + '({}) ({})'.format(columnx, prefixx),
                labelprefix + '({} ratio) ({}/{})'.format(columnx, prefixy, prefixx))
        x0, x1 = jointplot.ax_joint.get_xlim()
        jointplot.ax_joint.plot([x0, x1], [0, 0], '-k', zorder=1)
        jointplots.append(jointplot)
        if hascolor:
            colorbarinset(jointplot, **colorbaropts)
        if plotmarginals:
            fig.plot_marginals(sns.distplot, kde=False, hist_kws=hist_kws, bins=bins)
    return jointplots, jointplotsratio


def plotjoint_running_percentiles(x, y, percentiles=None, percentilecolours=None, limx=None, limy=None,
                                  ndivisions=None, nbinspan=None, labelx=None, labely=None, title=None,
                                  histtickspacingxmaj=None, histtickspacingymaj=None,
                                  scatterleft=False, scatterright=False, drawzeroline=True):
    """

    :param x: Float[]; x data.
    :param y: Float[]; y data.
    :param percentiles: Float[]; percentiles to plot. Default [5, 16, 50, 84, 95]
    :param percentilecolours: Float[]; percentile line colours. Default black to blue.
    :param limx: Float[2]; x-axis limits. Default (min(x), max(x)).
    :param limy: Float[2]; y-axis limits. Default (min(y), max(y)).
    :param ndivisions: Int; number of x-axis divisions. Default ceil(len(x)**(1/3)).
    :param nbinspan: Int; number of x-axis divisions that each bin should span. Default ceil(ndivisions/10).
    :param labelx: String; x-axis label.
    :param labely: String; y-axis label.
    :param title: String; title.
    :param histtickspacingxmaj: Float; spacing for major ticks on the x-axis marginal histogram's y-axis.
    :param histtickspacingymaj: Float; spacing for major ticks on the y-axis marginal histogram's y-axis.
    :param scatterleft: Bool; scatter plot points leftwards of the leftmost bin center?
    :param scatterright: Bool; scatter plot points rightwards of the rightmost bin center?
    :param drawzeroline: Bool; draw line at y=0?
    :return: seaborn.JointGrid handle for the plot.
    """
    numpoints = len(x)
    if len(y) != numpoints:
        raise ValueError('len(x)={} != len(y)={}'.format(numpoints, len(y)))
    if percentiles is None:
        percentiles = [5, 16, 50, 84, 95]
        percentilecolours = [(0.2, 0.5, 0.8), (0.1, 0.25, 0.4)]
        percentilecolours = percentilecolours + [(0, 0, 0)] + list(reversed(percentilecolours))
    if not all([0 <= x <= 100 for x in percentiles]):
        raise ValueError('Percentiles {} not all >=0 and <=100'.format(percentiles))
    if not len(percentiles) == len(percentilecolours):
        raise ValueError('len(percentiles)={} != len(percentilecolours)={}'.format(
            len(percentiles), len(percentilecolours)
        ))
    # TODO: Check all inputs
    if ndivisions is None:
        ndivisions = np.int(np.ceil(numpoints**(1./3.)))
    if nbinspan is None:
        nbinspan = np.int(np.ceil(ndivisions/10.))
    nedgesover = ndivisions*nbinspan + 2
    nbinsover = (ndivisions - 1)*nbinspan
    if limx is None:
        limx = (np.nanmin(x), np.nanmax(x))
    if limy is None:
        limy = (np.nanmin(y), np.nanmax(y))
    isylo = y < limy[0]
    isyhi = y > limy[1]
    y[isylo] = limy[0]
    y[isyhi] = limy[1]
    # Make a joint grid, plot a KDE and leave the marginal plots for later
    p = sns.JointGrid(x=x, y=y, ylim=limy, xlim=limx)
    p.plot_joint(sns.kdeplot, cmap="Reds", shade=True, shade_lowest=False,
                 n_levels=np.int(np.ceil(numpoints**(1/3))))
    # Setup bin edges to have overlapping bins for running percentiles
    binedges = np.sort(x)[np.asarray(np.round(np.linspace(0, len(x)-1, num=nedgesover)), dtype=int)]
    if drawzeroline:
        plt.axhline(y=0, color='k', linewidth=1, label='')
    plt.xlabel(labelx)
    plt.ylabel(labely)
    xbins = np.zeros(nbinsover)
    ybins = [np.zeros(nbinsover) for _ in range(len(percentiles))]
    # Get running percentiles
    for idxbin in range(nbinsover):
        xlower, xupper = binedges[[idxbin, idxbin + nbinspan + 1]]
        condbin = (x >= xlower) & (x <= xupper)
        # Add the bin percentiles
        if np.any(condbin):
            xbins[idxbin] = np.median(x[condbin])
            ybin = np.sort(y[condbin])
            for idxper, percentile in enumerate(percentiles):
                ybins[idxper][idxbin] = np.percentile(ybin, percentile)
        else:
            # Repeat previous value if bin is somehow empty
            xbins[idxbin] = (xlower + xupper)/2.
            for idxper, percentile in enumerate(percentiles):
                ybins[idxper][idxbin] = ybins[idxper][idxbin-1] if idxbin > 0 else np.nan
    for yper, pc, colpc in zip(ybins, percentiles, percentilecolours):
        plt.plot(xbins, yper, linestyle='-', color=colpc, linewidth=1.5, label=str(pc) + 'th %ile')
        for idxbin, idxlim in [(0, 0), (-1, 1)]:
            plt.plot([limx[idxlim], xbins[idxbin]], [yper[idxbin], yper[idxbin]], linestyle='-',
                     color=colpc, linewidth=1.5, label=None)
    plt.legend()
    xlowerp = binedges[0]
    idxxupper = np.int(np.ceil(nbinspan / 2))
    xupperp = binedges[idxxupper]
    # Plot outliers, with points outside of the plot boundaries as triangles
    # Not really necessary but more explicit.
    # TODO: Color code outliers by y-value?
    for idxbin in range(nbinsover):
        condbin = (x >= xlowerp)*(x <= xupperp)
        xcond = x[condbin]
        ycond = y[condbin]
        for upper, condoutlier in enumerate([ycond <= ybins[0][idxbin], ycond >= ybins[-1][idxbin]]):
            noutliers = np.sum(condoutlier)
            if noutliers > 0:
                if upper:
                    condy2 = isyhi[condbin]
                    marker = 'v'
                else:
                    condy2 = isylo[condbin]
                    marker = '^'
                colourpc = percentilecolours[-1 if upper else 0]
                for condplot, markercond, sizecond in [
                    (condoutlier*condy2, marker, 4),
                    (condoutlier*(~condy2), '.', 2)]:
                    if np.sum(condplot) > 0:
                        plt.scatter(xcond[condplot], ycond[condplot], s=sizecond, marker=markercond,
                                    color=colourpc)
                #plt.scatter(xcond[condoutlier], ycond[condoutlier], s=2, marker=markercond, color='k')
        xlowerp = xupperp
        if idxbin == (nbinsover-2):
            idxxupper = -1
        else:
            idxxupper += 1
        xupperp = binedges[idxxupper]
    for idxbin in ([0] if scatterleft else []) + ([nbinsover-1] if scatterright else []):
        cond = (y > ybins[0][idxbin]) & (y < ybins[-1][idxbin])
        cond = cond & ((x < xbins[idxbin]) if (idxbin == 0) else (x > xbins[idxbin]))
        plt.scatter(x[cond], y[cond], s=1, marker='+', color='k')
    p.ax_marg_x.hist(x, bins=ndivisions * 2, weights=np.repeat(1.0 / len(x), len(x)))
    plt.setp(p.ax_marg_x.get_yticklabels(), visible=True)
    if histtickspacingxmaj is not None:
        p.ax_marg_x.yaxis.set_major_locator(mpl.ticker.MultipleLocator(histtickspacingxmaj))
    p.ax_marg_y.hist(y, orientation='horizontal', bins=ndivisions * 4, weights=np.repeat(1.0 / len(y), len(y)))
    p.ax_marg_y.xaxis.set_ticks_position('top')
    plt.setp(p.ax_marg_y.get_xticklabels(), visible=True)
    if histtickspacingymaj is not None:
        p.ax_marg_y.xaxis.set_major_locator(mpl.ticker.MultipleLocator(histtickspacingymaj))
    if title is not None:
        p.fig.suptitle(title)
    return p
