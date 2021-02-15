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

import matplotlib.colors as mplcol
import pandas as pd
import seaborn as sns

from modelling_research.plotting import plotjoint

plotjoinsersicdefaults = {
    'limitsxratio': (-1, 1),
    'limitsyratio': (-1, 1),
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
    return plotjoint(tab, varnames, prefixx=prefixx, prefixy=prefixy, **opts)


def readtable(filename):
    tab = pd.read_csv(filename)
    scalesources = {
        'hst': 0.03,
        'hst2hsc': 0.168,
    }
    for column in ['cosmos.hst.' + x for x in ['ser.re.1', 'devexp.re.1', 'devexp.re.2']]:
        tab[column] *= scalesources['hst']
    return tab

