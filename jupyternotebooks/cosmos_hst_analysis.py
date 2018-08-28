
# coding: utf-8

# In[2]:


import astropy as ap
import galsim as gs
import os
import pickle


# In[ ]:


with open(os.path.expanduser("~/raid/lsst/cosmos/cosmos_25.2_training_fits_pickle.dat"), 'rb') as f:
    data = pickle.load(f)


# In[3]:


path = '/r0/taranu/hsc/cosmos/COSMOS_25.2_training_sample/'
file = 
ccat = gs.COSMOSCatalog("real_galaxy_catalog_25.2.fits", dir=path)
rgcfits = ap.io.fits.open(path, file))[1].data


# In[4]:


#     SERSICFIT[0]: intensity of light profile at the half-light radius.
#     SERSICFIT[1]: half-light radius measured along the major axis, in units of pixels
#                   in the COSMOS lensing data reductions (0.03 arcsec).
#     SERSICFIT[2]: Sersic n.
#     SERSICFIT[3]: q, the ratio of minor axis to major axis length.
#     SERSICFIT[4]: boxiness, currently fixed to 0, meaning isophotes are all
#                   elliptical.
#     SERSICFIT[5]: x0, the central x position in pixels.
#     SERSICFIT[6]: y0, the central y position in pixels.
#     SERSICFIT[7]: phi, the position angle in radians.  If phi=0, the major axis is
#                   lined up with the x axis of the image.

params = ["IDENT", "mag_auto", "flux_radius", "zphot", "use_bulgefit", "viable_sersic"]
paramsser = ["flux","re","n","q","phi","x0","y0"]
colnames = ["ser." + param for param in paramsser] + ["expdev." + param for param in [comp + "." + param for comp in ["exp","dev"] for param in paramsser]]
colnames = ["id","ra","dec"] + ["cosmos." + x for x in colnames]


# In[ ]:


for idx in [0]:
    row = [idx] + rgcfits[idx][1:3]]
    rec = ccat.getParametricRecord(idx)
    row = row + [rec[param] for param in params]

