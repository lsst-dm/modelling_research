import matplotlib.colors as mplcol
import pandas as pd
import seaborn as sns

from modelling_research.plotting import plotjoint

plotjoinsersicdefaults = {
    'limitsx': (-1, 1),
    'limitsy': (-1, 1),
    'columncolor': "profit.hst.serb.nser.1.1",
    'colorbaropts': {
        'label': r"ProFit HST $n_{Ser}$",
        'ticks': [0.1, 0.5, 1, 2, 4, 10],
        'ticklabels': [0.1, 0.5, 1, 2, 4, 10],
    },
    'cmap': mplcol.ListedColormap(sns.color_palette("RdYlBu_r", 100)),
    'norm': mplcol.LogNorm(vmin=0.1, vmax=10),
    'marker': '.',
    'edgecolor': 'k',
    's': 36,
    'separator': '.',
}


def plotjointsersic(tab, prefixx, prefixy, varnames, **kwargs):
    opts = plotjoinsersicdefaults.copy()
    opts.update(kwargs)
    plotjoint(tab, varnames, prefixx=prefixx, prefixy=prefixy, **opts)


def readtable(filename):
    tab = pd.read_csv(filename)
    scalesources = {
        'hst': 0.03,
        'hst2hsc': 0.168,
    }
    for column in ['cosmos.hst.' + x for x in ['ser.re.1', 'devexp.re.1', 'devexp.re.2']]:
        tab[column] *= scalesources['hst']
    return tab

