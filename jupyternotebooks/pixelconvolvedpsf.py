
# coding: utf-8

# In[1]:


# Test what happens if one fits a galaxy model with a pixel-convolved PSF at 
# two different spatial (pixel) scales

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyprofit.python.objects as proobj
import pyprofit.python.util as proutil
import galsim as gs

mpl.rcParams['figure.dpi'] = 160
mpl.rcParams['image.origin'] = 'lower'

nx = 200
ny = nx

# 1000 counts/pixel
background = 1e3

# Galaxy parameters
nser = 0.5
re = 2
band = ""
# Slightly ludicrous S/N
flux = 10*background*nx*ny

# PSF parameters
fwhm = 4

galaxy = gs.Sersic(flux=flux, n=nser, half_light_radius=re)
psf = gs.Gaussian(flux=1, fwhm=fwhm)

imgpsf = psf.drawImage(nx=25, ny=25, scale=1, method="no_pixel").array
imgpsfnoisy = (np.random.poisson(flux*imgpsf + background) + 0.0).reshape(
    imgpsf.shape)
imgpsfsigma = np.sqrt(imgpsfnoisy)
imgpsfnoisy -= background
plt.imshow(np.log10(imgpsfnoisy))
plt.show()


# In[2]:


# draw the true galaxy model
modelgal = proutil.getmodel(
        {band: flux}, "sersic:1", (ny, nx),
        sizes=[np.sqrt(re**2 + (fwhm/2)**2)], axrats=[1], slopes=[0.5],
        engineopts={'drawmethod': 'no_pixel', 'gsparams': gs.GSParams()})
proutil.setexposure(modelgal, band, "empty")
eval = modelgal.evaluate(keepimages=True, getlikelihood=False)
imgmodel = np.random.poisson(modelgal.data.exposures[band][0].meta[
    'modelimage']+background)

plt.imshow(np.log10(imgmodel))
plt.show()


# In[3]:


def downsample(x, scale):
    if scale == 1:
        return x
    return x.reshape((x.shape[0]//scale, scale, x.shape[1]//scale,
                      scale)).sum(3).sum(1)

models = {}
for scale in [1, 5]:
    modelpsf = proutil.getmodel(
        {band: flux}, "sersic:1", [x//scale for x in imgpsf.shape],
        sizes=[fwhm/2/scale], axrats=[1], slopes=[0.5],
        engineopts={'drawmethod': 'no_pixel', 'gsparams': gs.GSParams()})
    for param in modelpsf.getparameters(fixed=False):
        if param.name in ["cenx", "ceny", "nser", "axrat", "ang"]:
            param.fixed = True
        elif isinstance(param, proobj.FluxParameter):
            param.fixed = not param.isfluxratio
    proutil.setexposure(modelpsf, band, downsample(imgpsfnoisy, scale), 
                        1.0/np.sqrt(downsample(imgpsfnoisy+background, scale)))
    modelpsf.evaluate(plot=True)
    plt.show()
    fit, modeller = proutil.fitmodel(
        modelpsf, printfinal=True, modellibopts={
        'algo': 'Nelder-Mead', 'options': {'disp': True}})
    print(["=".join([x.name, str(x.getvalue())]) for x in 
           modelpsf.getparameters(fixed=True)])
    
    modelpsf.getparameters(fixed=True, flatten=False)[0][1][0][0].setvalue(
        1, transformed=False)
    modelgal = proutil.getmodel(
        {band: flux}, "sersic:1", [x//scale for x in imgmodel.shape],
        sizes=[re/scale], axrats=[1], slopes=[nser],
        engineopts={'drawmethod': 'no_pixel', 'gsparams': gs.GSParams()})
    for param in modelgal.getparameters(fixed=False):
        if param.name in ["cenx", "ceny", "nser", "axrat", "ang"]:
            param.fixed = True
        elif isinstance(param, proobj.FluxParameter):
            param.fixed = not param.isfluxratio
    proutil.setexposure(modelgal, band, downsample(imgmodel-background, scale),
                        1.0/np.sqrt(downsample(imgmodel, scale)),
                        psf=proobj.PSF(band, model=modelpsf.sources[0], 
                                       usemodel=True, modelpixelated=True))
    fit, modeller = proutil.fitmodel(
        modelgal, printfinal=True, modellibopts={
        'algo': 'Nelder-Mead', 'options': {'disp': True}})
    modelgal.evaluate(plot=True)
    
    print(["=".join([x.name, str(x.getvalue())]) for x in 
       modelgal.getparameters(fixed=True)])
    


# In[ ]:




