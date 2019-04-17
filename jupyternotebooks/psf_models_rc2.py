#!/usr/bin/env python
# coding: utf-8

# ### PSF Investigation

# In[1]:


from astropy.table import Table
import glob
import lsst.afw.table as afwTable
import matplotlib as mpl
import matplotlib.pyplot as plt
from modelling_research.plotting import plotjoint_running_percentiles
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

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


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
mpl.rcParams['figure.dpi'] = 240
mpl.rcParams['image.origin'] = 'lower'
sns.set(rc={'axes.facecolor': '0.85', 'figure.facecolor': 'w'})


# In[3]:


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


nbinspan = 8
limsfrac = (-0.05, 0.05)
tickshist=np.linspace(0, 1, 101)
bandsplot = bands
doonlyoneplot = False
if doonlyoneplot:
    bandsplot = ['HSC-I']
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
        plotjoint_running_percentiles(x, sizefrac, limx=limx, limy=limsfrac, ndivisions=32, nbinspan=nbinspan, labelx=labelsize, labely=labelsizeresid,
                                       title=title, histtickspacingxmaj=0.02, histtickspacingymaj=0.02, scatterleft=True, scatterright=True)

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
            if numpoints > 10:    
                x = np.log10(datum['fluxPsf'][cond])
                y = sizefrac[cond]
                plotjoint_running_percentiles(x, y, limy=limsfrac, nbinspan=nbinspan, labelx=labelflux, labely=labelsizeresid,
                                               title=title, histtickspacingxmaj=0.05, histtickspacingymaj=0.05, scatterleft=True, scatterright=True)
        # TODO: Make sensible plots per patch
        for patch in patches:
            cond = (datum['file'] == patch)*condsizefrac
            grid = sns.jointplot(np.log10(datum['fluxPsf'][cond]), sizefrac[cond], ylim=[-0.1,0.1],
                                 stat_func=None, size=4, s=1, marginal_kws={'bins':30})
            grid.fig.suptitle(band)
plt.show()


# In[ ]:




