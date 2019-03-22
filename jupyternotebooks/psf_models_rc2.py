
# coding: utf-8

# ### PSF Investigation

# In[1]:


import pandas as pd                                                               
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
from astropy.table import Table
import lsst.afw.table as afwTable
savepre = '/home/dtaranu/raid/lsst/w2019_02_coaddpsf'
savepost = '.parq'
bands = ['HSC-' + x for x in ['G','R','I']]
files = {
    band: glob.glob('/datasets/hsc/repo/rerun/RC/w_2019_02/DM-16110/'
                    'deepCoadd-results/{}/*/*/meas*'.format(band))
    for band in bands
}
filesl5 = {
    band: glob.glob('/datasets/hsc/repo/rerun/private/yusra/RC2/w_2019_06_lanczos5/'
                    'deepCoadd-results/{}/*/*/meas*'.format(band))
    for band in bands
}
get_ipython().magic('matplotlib inline')
sns.set_style("darkgrid")
mpl.rcParams['figure.dpi'] = 240


# In[2]:


def make_summary(file):
    try:
        src = afwTable.BaseCatalog.readFits(file)
    except:
        return None
    print(file)
    mask = np.logical_and.reduce((
        src.get('calib_psf_used')==1,
        src.get('base_PixelFlags_flag_interpolated')==False,
        src.get('base_PixelFlags_flag_saturated')==False,
        src.get('base_PixelFlags_flag_inexact_psf')==False,
        src.get('base_PixelFlags_flag_clipped')==False
    ))
    if np.sum(mask)==0:
        print('No data')
        return None
    sIxxKey = src.schema.find('base_SdssShape_xx').key
    sIyyKey = src.schema.find('base_SdssShape_yy').key
    sIxyKey = src.schema.find('base_SdssShape_xy').key
    mIxxKey = src.schema.find('base_SdssShape_psf_xx').key
    mIyyKey = src.schema.find('base_SdssShape_psf_yy').key
    mIxyKey = src.schema.find('base_SdssShape_psf_xy').key
    fluxPsfKey = src.schema.find('base_PsfFlux_instFlux').key
    fluxPsfErrKey = src.schema.find('base_PsfFlux_instFluxErr').key
    fluxCmodelKey = src.schema.find('modelfit_CModel_instFlux').key
    fluxCmodelErrKey = src.schema.find('modelfit_CModel_instFluxErr').key
    stars = src[mask]
    starIxx = src.get(sIxxKey)[mask]
    starIxy = src.get(sIxyKey)[mask]
    starIyy = src.get(sIyyKey)[mask]
    modelIxx = src.get(mIxxKey)[mask]
    modelIxy = src.get(mIxyKey)[mask]
    modelIyy = src.get(mIyyKey)[mask]
    data = {}
    data['starE1'] = (starIxx-starIyy)/(starIxx+starIyy)
    data['starE2'] = (2*starIxy)/(starIxx+starIyy)
    data['starSize'] = np.sqrt(0.5*(starIxx + starIyy))*2.354820045*0.168
    data['modelE1'] = (modelIxx-modelIyy)/(modelIxx+modelIyy)
    data['modelE2'] = (2*modelIxy)/(modelIxx+modelIyy)
    data['modelSize'] = np.sqrt(0.5*(modelIxx + modelIyy))*2.354820045*0.168
    data['ra'] = [a.getCoord().getRa().asDegrees() for a in stars]
    data['dec'] = [a.getCoord().getDec().asDegrees() for a in stars]
    data['fluxPsf'] = src.get(fluxPsfKey)[mask]
    data['fluxPsfErr'] = src.get(fluxPsfErrKey)[mask]
    data['fluxCmodel'] = src.get(fluxCmodelKey)[mask]
    data['fluxCmodelErr'] = src.get(fluxCmodelErrKey)[mask]
    data['file'] = file
    #data['visit'] = visit
    df = pd.DataFrame(data)
    return df

def make_parquet(filesin, fileout):
    data=[]
    for f in filesin:
        result = make_summary(f)
        if result is None:
            continue
        data.append(result)
    warp = pd.concat(data)
    warp.to_parquet(fileout)
    return warp


# In[3]:


def plot_joint_running_percentiles(x, y, percentiles=None, percentilescolours=None, limx=None, limy=None, nbins=None, binoverlap=None,
                                   labelx=None, labely=None, title=None, histtickspacingxmarg=None, histtickspacingymarg=None):
    numpoints = len(x)
    if len(y) != numpoints:
        raise ValueError('len(x)={} != len(y)={}'.format(numpoints, len(y)))
    if percentiles is None:
        percentiles = [5, 16, 50, 84, 95]
        percentilecolours = [(0.2, 0.5, 0.8), (0.1, 0.25, 0.4)]
        percentilecolours = percentilecolours + [(0, 0, 0)] + list(reversed(percentilecolours))
    # TODO: Check all inputs
    if nbins is None:
        nbins = np.int(np.ceil(numpoints**(1/3)))
    if binoverlap is None:
        binoverlap = np.int(np.cil(nbins/10))
    nedgesover = nbins*binoverlap + 1
    nbinsover = (nbins-1)*binoverlap
    if limx is None:
        limx = (np.min(x), np.max(x))
    if limy is None:
        limy = (np.min(y), np.max(y))
    isylo = y < limy[0]
    isyhi = y > limy[1]
    y[isylo] = limy[0]
    y[isyhi] = limy[1]
    # Make a joint grid, plot a KDE and leave the marginal plots for later
    p = sns.JointGrid(x=x, y=y, ylim=limy, xlim=limx)
    p.plot_joint(sns.kdeplot, cmap="Reds", shade=True, shade_lowest=False, n_levels = np.int(np.ceil(numpoints**(1/3))))
    # Setup bin edges to have overlapping bins for running percentiles
    binedges = np.sort(x)[np.asarray(np.round(np.linspace(0, len(x)-1, num=nedgesover)), dtype=int)]
    plt.plot(binedges[[0, -1]], [0, 0], 'k-', linewidth=1, label='')
    plt.xlabel(labelx)
    plt.ylabel(labely)
    xbins = np.zeros(nbinsover)
    ybins = [np.zeros(nbinsover) for _ in range(len(percentiles))]
    # Get running percentiles
    for idxbin in range(nbinsover):
        xlower, xupper = binedges[[idxbin,idxbin+binoverlap-1]]
        condbin = (x >= xlower)*(x <= xupper)
        xbins[idxbin] = np.median(x[condbin])
        ybin = np.sort(y[condbin])
        for idxper, percentile in enumerate(percentiles):
            ybins[idxper][idxbin] = np.percentile(ybin, percentile)
    for yper, pc, colpc in zip(ybins, percentiles, percentilecolours):
        plt.plot(xbins, yper, linestyle='-', color=colpc, linewidth=1.5, label=str(pc) + 'th %ile')
    plt.legend()
    xlowerp = binedges[0]
    idxxupper = np.int(np.ceil(binoverlap/2))
    xupperp = binedges[idxxupper]
    # Plot outliers, with points outside of the plot boundaries as triangles
    # Not really neccessary but more explicit.
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
                #print(np.sum(condbin), np.sum(condoutlier), np.sum(condy2), np.sum(condoutlier*condy2), np.sum(condoutlier*(1-condy2)))
                for condplot, markercond in [(condoutlier*condy2, marker), (condoutlier*(~condy2), '.')]:
                    if np.sum(condplot) > 0:
                        plt.scatter(xcond[condplot], ycond[condplot], s=2, marker=markercond, color='k')
                #plt.scatter(xcond[condoutlier], ycond[condoutlier], s=2, marker=markercond, color='k')
        xlowerp = xupperp
        if idxbin == (nbinsover-2):
            idxxupper = -1
        else:
            idxxupper += 1
            xupperp = binedges[idxxupper]
    p.ax_marg_x.hist(x, bins=nbins*2, weights=np.repeat(1.0/len(x), len(x)))
    plt.setp(p.ax_marg_x.get_yticklabels(), visible=True)
    if histtickspacingxmarg is not None:
        p.ax_marg_x.yaxis.set_major_locator(mpl.ticker.MultipleLocator(histtickspacingxmarg))
    p.ax_marg_y.hist(y, orientation='horizontal', bins=nbins*4, weights=np.repeat(1.0/len(y), len(y)))
    p.ax_marg_y.xaxis.set_ticks_position('top')
    plt.setp(p.ax_marg_y.get_xticklabels(), visible=True)
    if histtickspacingymarg is not None:
        p.ax_marg_y.xaxis.set_major_locator(mpl.ticker.MultipleLocator(histtickspacingymarg))
    if title is not None:
        p.fig.suptitle(title)
    return p


# In[4]:


data = {}
datal5 = {}
for band in bands:
    for datadict, filestoread, desc in [(data, files, ''), (datal5, filesl5, 'l5')]:
        savefile = ('_' + band).join([savepre, desc + savepost])
        print(savefile)
        if not os.path.exists(savefile):
            datadict[band] = make_parquet(filestoread[band], savefile)
        else:
            datadict[band] = pd.read_parquet(savefile)


# In[5]:


binoverlap = 8
limsfrac = (-0.05, 0.05)
tickshist=np.linspace(0, 1, 101)
bandsplot = bands
#bandsplot = ['HSC-G']
doonlyoneplot = False
labelflux = 'log10(instFluxPsf)'
labelsizeresid = r'($FWHM_{data}$ - $FWHM_{model}$)/$FWHM_{data}$'
labelsize = r'$FWHM_{data}/arcsec$'
limxs = {
    'HSC-G': (0.54, 0.94),
    'HSC-R': (0.41, 0.71),
    'HSC-I': (0.41, 0.71),
}
for dataplot, typeofdata in [(data, 'Lanczos3'), (datal5, 'Lanczos5')]:
    for band in bandsplot:
        datum = dataplot[band]
        limx = limxs[band]
        x = np.clip(datum['starSize'], limx[0], limx[1])
        sizefrac = (datum['starSize']-datum['modelSize'])/datum['starSize']
        title='{} {} PSF model residuals (N={})'.format(band, typeofdata, len(x))
        plot_joint_running_percentiles(x, sizefrac, limx=limx, limy=limsfrac, nbins=80, binoverlap=8, labelx=labelsize, labely=labelsizeresid,
                                       title=title, histtickspacingxmarg=0.02, histtickspacingymarg=0.02)

for dataplot, typeofdata in [(data, 'Lanczos3'), (datal5, 'Lanczos5')]:
    for band in bandsplot:
        datum = dataplot[band]
        sizefrac = (datum['starSize']-datum['modelSize'])/datum['starSize']
        #patches = np.unique(datum['file'])
        patches = []
        condsizefrac = np.isfinite(sizefrac)*np.isfinite(datum['fluxPsf'])*(datum['starSize']<2)
        sizebins = np.sort(datum['starSize'][condsizefrac])[np.asarray(np.round(np.linspace(0, np.sum(condsizefrac)-1, num=10+1)), dtype=int)]
        for idx in range(len(sizebins)*(not doonlyoneplot)+2*doonlyoneplot-1):
            sizemin, sizemax = sizebins[idx:idx+2]
            cond = condsizefrac*(datum['starSize'] > sizemin)*(datum['starSize'] < sizemax)
            numpoints = np.sum(cond)
            title = '{} {}, {:.3f} <= FWHM <= {:.3f}, N={}'.format(band, typeofdata, sizemin, sizemax, numpoints)
            print(title)
            nbins = np.int(np.ceil(numpoints**(1/3)))
            nedgesover = nbins*binoverlap + 1
            nbinsover = (nbins-1)*binoverlap
            if numpoints > 10:    
                x = np.log10(datum['fluxPsf'][cond])
                y = sizefrac[cond]
                plot_joint_running_percentiles(x, y, limy=limsfrac, nbins=nbins, binoverlap=binoverlap, labelx=labelflux, labely=labelsizeresid,
                                               title=title, histtickspacingxmarg=0.05, histtickspacingymarg=0.05)
        # TODO: Make sensible plots per patch
        for patch in patches:
            cond = (datum['file'] == patch)*condsizefrac
            grid = sns.jointplot(np.log10(datum['fluxPsf'][cond]), sizefrac[cond], ylim=[-0.1,0.1],
                                 stat_func=None, size=4, s=1, marginal_kws={'bins':30})
            grid.fig.suptitle(band)
plt.show()


# In[ ]:




