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
