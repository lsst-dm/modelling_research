
# coding: utf-8

# In[1]:


# This notebook demonstrates using pyprofit to fit an 8-component Gaussian 
# mixture model to an image of a Sersic model galaxy

# This is one method of determing weights for multi-Gaussian Sersics a la 
# Hogg & Lang 2013 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyprofit as pyp
import galsim as gs

mpl.rcParams['figure.dpi'] = 240
mpl.rcParams['image.origin'] = 'lower'

# Make an image of a galaxy with a given Sersic index
## Plotting only the upper-right quadrant since it's round and symmetric

nx = 200
ny = nx
nser = 6.0
re = 5
band = ""
flux = nx*ny

img = gs.Sersic(flux=flux, n=nser, half_light_radius=re).drawImage(
    nx=nx, ny=ny, scale=1, method="real_space")
imgquad = img.array[(ny//2):ny, (nx//2):nx]
print(np.sum(img.array)/np.array([1, flux]))
print(np.sum(imgquad))
plt.imshow(np.log10(imgquad))
plt.show()


# In[2]:


import pyprofit.python.objects as proobj
import pyprofit.python.util as proutil
ncomp = 8
offsetrefac = 1
refacmax = 0.5
fluxfracmod = 0

# Drawmethod fft will integrate over pixels; no_pixel will evaluate at center

model = proutil.getmodel(
	{band: flux}, "sersic:" + str(ncomp), imgquad.shape,
	sizes=np.power(re, np.linspace(
        offsetrefac-refacmax, offsetrefac+refacmax, ncomp)),
	axrats=np.repeat(1, ncomp),
	slopes=np.repeat(0.5, ncomp),
    engineopts={'drawmethod': 'fft', 'gsparams': gs.GSParams()})
for param in model.getparameters(fixed=False):
    if param.name == "cenx" or param.name == "ceny":
        param.setvalue(0)
        param.fixed = True
    elif param.name in ["nser", "axrat", "ang"]:
        param.fixed = True
    elif isinstance(param, proobj.FluxParameter):
        param.fixed = not param.isfluxratio
        if param.isfluxratio:
            param.setvalue(param.getvalue(transformed=True)+fluxfracmod, 
                           transformed=True)
    elif param.name == "re":
        print(param.getvalue())
# TODO: Think of a better way to incorporate sky-like background noise 
sigma = np.sqrt(imgquad + 1) 
proutil.setexposure(model, band, imgquad, 1.0/sigma)

print(["=".join([x.name, str(x.getvalue())]) for x in 
       model.getparameters(fixed=True)])


# In[3]:


eval = model.evaluate(plot=True)
print(eval[0])
print(eval[1])
plt.show()


# In[5]:


fit, modeller = proutil.fitmodel(model, modellib="pygmo", 
                                 printfinal=True, printsteps=100,
                                 modellibopts={'algo': 'neldermead'})


# In[5]:


# TODO: Why is this plot so tiny??

model.evaluate(plot=True)
plt.show()


# In[9]:


fluxes = []
varfracs = []
for param in model.getparameters(fixed=False):
    if isinstance(param, proobj.FluxParameter):
        if param.isfluxratio:
            fluxes.append(param.getvalue(transformed=False))
    elif param.name == "re":
        varfracs.append((param.getvalue(transformed=False)/re)**2)

print(fluxes)
fluxes.append(1)
flux = 1
for i in range(len(fluxes)):
    fluxes[i] *= flux
    flux -= fluxes[i]

print(fluxes)
print(varfracs)



# In[ ]:




