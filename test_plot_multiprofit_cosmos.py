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

import matplotlib as mpl
import seaborn as sns

from modelling_research.plot_multiprofit_cosmos import readtable
from modelling_research.plot_multiprofit_cosmos import plotjointsersic

if __name__ == '__main__':
    filename = 'data/multiprofit-cosmos-fits.csv'
    tab = readtable(filename)
    varnames = ["flux", "re.1", "nser.1"]
    sns.set_style('darkgrid')
    mpl.rcParams['figure.dpi'] = 160
    mpl.rcParams['image.origin'] = 'lower'
    sns.set(rc={'axes.facecolor': '0.85', 'figure.facecolor': 'w'})
    plotjointsersic(tab, 'profit.hst.serb', 'cosmos.hst.ser', varnames, plotratiosjoint=False, postfixx='1')
    plotjointsersic(tab, 'cosmos.hst.ser', 'profit.hst.serb', varnames, postfixy='1')
