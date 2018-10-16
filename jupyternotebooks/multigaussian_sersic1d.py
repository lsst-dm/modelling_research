
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprofit.objects as mpfobj
import numpy as np
import seaborn as sns
import scipy as sp

get_ipython().magic('matplotlib inline')
sns.set_style("darkgrid")
mpl.rcParams['figure.dpi'] = 240


# In[2]:


# Surface brightness of a unit flux Sersic profile
def sersic(r, n, re):
    bn = sp.stats.gamma.ppf(0.5, 2*n)
    g_factor = sp.special.gamma(2*n)
    flux = np.power(re, 2)*2*np.pi*n*g_factor*np.exp(bn)/np.power(bn, 2*n)
    ir = np.exp(-bn*(np.power(r/re, 1./n)-1.))
    return ir/flux


# Compute chisq as log difference in 2D binned flux
# i.e. not really chisq since it's a sum of squares but without any error
def chisq_sersic(params, x, y, weightsbins, plotdata=False, plotmodel=False, returnall=False):
    ymodel = np.zeros(len(x))
    weights, res = paramstoweightres(params)
    for weight, re in zip(weights, res):
        ymodel += weight*sersic(x, 0.5, re)
    if plotdata:
        plt.plot(x, np.log10(y))
    if plotmodel:
        plt.plot(x, np.log10(ymodel))
    if weightsbins is not None:
        chisq = np.log10(np.sum(weightsbins*(y-ymodel)**2))
        if returnall:
            return chisq, ymodel
        return chisq


def paramstoweightres(params):
    nsplit = (len(params)+1)//2
    weights = np.zeros(nsplit)
    res = np.zeros(nsplit)
    total = 1.0
    for i in range(nsplit):
        if i < (nsplit-1):
            weight = sp.special.expit(params[i])
            weights[i] = total*weight
            total *= (1.0-weight)
        res[i] = 10**params[i+nsplit-1]
    weights[nsplit-1] = 1-np.sum(weights)
    return weights, res


def weightrestoparams(weights, res):
    paramweights = []
    paramres = []
    total = 1.0
    for weight, re in zip(weights, res):
        paramweights.append(sp.special.logit(weight/total))
        total -= weight
        paramres.append(np.log10(re))
    return paramweights[:-1] + paramres


def fitweights(nvals, weightssigmas={}, method='BFGS', plot=True):
    for nvalsi, weightsvars in nvals:
        params = weightrestoparams(weightsvars[0], weightsvars[1]) if weightsvars is not None else None
        for n in nvalsi:
            idxs = cuts['min' if n <= 2 else ('max' if n >= 4 else 'mid')]
            y = sersic(rmid[idxs], n, 1)
            paramsbytype = {}
            if params is not None:
                paramsbytype['prev.'] = params
            if n in weightssigmas:
                weightsvars = weightssigmas[n]
                paramsbytype['existing'] = weightrestoparams(
                    weightsvars[0][::-1], np.sqrt(weightsvars[1][::-1])*gaussian_sigma_to_re)
            plotteddata = not plot
            chisqmin = np.Inf
            for name, paramsi in paramsbytype.items():
                chisq = chisq_sersic(paramsi, rmid[idxs], y, areasq[idxs], plotdata=not plotteddata,
                                     plotmodel=plot)
                print(name, ' chisq =', chisq)
                if chisq < chisqmin:
                    params = paramsi
                plotteddata = True
            fit = sp.optimize.minimize(chisq_sersic, params, args=(rmid[idxs], y, areasq[idxs]),
                                       tol=1e-5, options={'disp': True, }, method=method)
            params = fit['x']
            print(chisq_sersic(params, rmid[idxs], y, areasq[idxs], plotmodel=plot))
            plt.show()
            weights, res = paramstoweightres(params)
            idxsort = np.argsort(res)[::-1]
            print('{}: ('.format(n))
            for arr, prefix, postfix in [(weights, 'normalize(', ')'), (res, '', '')]:
                print('    ' + prefix + 'np.array([', ', '.join(
                    ['{:.10e}'.format(x) for x in np.array(arr)[idxsort]]) + postfix, ']),')
            print('),')


def normalize(array):
    array /= np.sum(array)
    return array


# In[3]:


order = 8

nbinsperre = 1000
remin = 8
remid = 10
remax = 12
redge = np.linspace(0, remax, nbinsperre*remax+1)
rmid = (redge[1:] + redge[:-1])/2.
rsq = redge**2
areasq = np.pi*(rsq[1:] - rsq[:-1])

cuts = {
    'min': range(remin*nbinsperre),
    'mid': range(remid*nbinsperre),
    'max': range(remax*nbinsperre),
}

for n in [0.5, 1, 2, 4]:
    plt.plot(rmid[cuts['min']], np.log10(sersic(rmid[cuts['min']], n, 1)))
plt.tight_layout()
for axis in ['x','y']:
    plt.autoscale(enable=True, axis=axis, tight=True)
plt.show()

weightssigmas = mpfobj.MultiGaussianApproximationProfile.weights['sersic'][order]


# In[4]:


#nvals = np.arange(0.55, 0.631, step=0.05)
nvals = [
    (
        np.array([0.5005]),
        (
            normalize(np.array([0.999, 1e-3])),
            np.array([1.00052323e+00, 6.73227271e-01]),
        ),
    ),
    (
        np.array([0.5028]),
        (
            normalize(np.array([0.982, 1.7e-2, 2.8e-4])),
            np.array([1.00445665e+00, 7.93714230e-01, 3.67450599e-01]),
        ),
    ),
    (
        np.array([0.511]),
        (
            normalize(np.array([0.93, 7e-2, 2e-3, 8e-5])),
            np.array([1.015, 0.855, 4.87450599e-01, 0.215]),
        ),
    ),
    (
        np.array([0.531]),
        (
            normalize(np.array([0.7, 2.6e-1, 2e-2, 1e-3, 5e-5])),
            np.array([1.06, 0.855, 4.87450599e-01, 0.215, 0.1]),
        ),
    ),
    (
        np.array([0.565]),
        (
            normalize(np.array([0.55, 0.4, 5e-2, 5e-3, 4e-4, 2e-5])),
            np.array([1.12, 0.92, 0.6, 0.4, 0.2, 0.1]),
        ),
    ),
    (
        np.array([0.6]),
        (
            normalize(np.array([0.4, 0.5, 1e-1, 1e-2, 2e-3, 2e-4, 1e-5])),
            np.array([1.2, 0.97, 0.7, 0.4, 0.2, 0.1, 0.05]),
        ),
    ),
    (
        np.array([0.65]),
        (
            normalize(np.array([0.27, 0.5, 0.15, 3.5e-2, 6e-3, 8e-4, 1e-4, 1e-5])),
            np.array([1.3, 1.04, 0.7, 0.5, 0.3, 0.17, 0.09, 0.04]),
        ),
    ),
    (
        np.array([
            0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
            1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
            1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90,
            1.95, 2.00, 2.05, 2.10, 2.15, 2.20, 2.25, 2.30, 2.35,
            2.40, 2.45, 2.50, 2.55, 2.60, 2.65, 2.70, 2.75, 2.80,
            2.85, 2.90, 2.95, 3.00, 3.05, 3.10, 3.15, 3.20, 3.25,
            3.30, 3.35, 3.40, 3.45, 3.50, 3.55, 3.60, 3.65, 3.70,
            3.75, 3.80, 3.85, 3.90, 3.95, 4.00, 4.05, 4.10, 4.15,
            4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50, 4.55, 4.60,
            4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00, 5.05,
            5.10, 5.15, 5.20, 5.25, 5.30, 5.35, 5.40, 5.45, 5.50,
            5.55, 5.60, 5.65, 5.70, 5.75, 5.80, 5.85, 5.90, 5.95,
            6.00, 6.05, 6.10, 6.15, 6.20, 6.25, 6.3
        ]),
        (
            normalize(np.array([0.27, 0.5, 0.15, 3.5e-2, 6e-3, 8e-4, 1e-4, 1e-5])),
            np.array([1.3, 1.04, 0.7, 0.5, 0.3, 0.17, 0.09, 0.04]),
        ),
    )
]

fitweights(nvals)


# In[5]:


nbins2 = 20000
remax2 = 16
redge2 = np.linspace(0, remax2, nbins2+1)
rmid2 = (redge2[1:] + redge2[:-1])/2.
weightsvars2 = (
    np.array([2.35664934e-01, 2.47200716e-01, 2.15434097e-01, 1.52002399e-01,
              8.82420146e-02, 4.19064584e-02, 1.55837863e-02, 3.96559442e-03]),
    np.array([7.48576191e+00, 2.00069998e+00, 7.36312757e-01, 2.88738813e-01,
              1.13519914e-01, 4.27457135e-02, 1.44861835e-02, 3.82934208e-03]),
)
params2 = weightrestoparams(weightsvars2[0], weightsvars2[1])
y2 = sersic(rmid2, 4, 1)
rsq2 = redge2**2
areasq2 = np.pi*(rsq2[1:] - rsq2[:-1])
rv, ymodel2 = chisq_sersic(params2, rmid2, y2, areasq2, plotdata=True, plotmodel=True, returnall=True)
print(np.sum((y2*areasq2)[0:10000]), np.sum((ymodel2*areasq2)[0:10000]))
print(np.sum(y2*areasq2), np.sum(ymodel2*areasq2))


# In[ ]:




