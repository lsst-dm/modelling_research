
# coding: utf-8

# In[1]:


import copy
import importlib
import numpy as np
import os
import scipy.stats as spstats
import sys

# Import necessary scripts in a hacky way - should just make a proper __init__.py and package
# Not sure what to do about the python path though
sys.path.insert(0, os.path.join("/home","dtaranu","src","mine","taranu_lsst"))
import make_cutout as cutouts
import test_lsst_cmodel as tlc


# In[4]:


# Setup for plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage

plt.style.use('seaborn-notebook')
mpl.rcParams['figure.dpi'] = 150


# In[2]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# In[2]:


# Set up butlers and some conveniences
config = tlc.setupCModelConfig()
[butlers, butlerdefault] = tlc.getCModelButlers(config["params"], config["butlerdata"], config["stack"], config["path"])
meastabledefault = butlerdefault["measures"]
calexp = butlerdefault["calexp"]
cmodelpre = 'modelfit_CModel'
objectisprimary = measdefault['detect_isPrimary']


# In[124]:


# Test re-running cModel
objdefault = butlers["gradthresh"]["1e-2"]["measures"][cmodelpre + "_exp_objective"][objectisprimary]
objcompare = butlers["gradthresh"]["1e-3"]["measures"][cmodelpre + "_exp_objective"][objectisprimary]
objdiff = objdefault - objcompare

print(spstats.describe(objdiff[np.isfinite(objdiff)]))

cmodelrerun = False
if cmodelrerun:
    calexpcopy = copy.deepcopy(butlerdefault['calexp'])
    [measTask, noiseReplacer, newSrcCatalog] = tlc.getCModelRerunTask([idtarget], meastabledefault, calexpcopy)
    measTask.runPlugins(noiseReplacer, newSrcCatalog, calexpcopy)


# In[116]:


cmodelpre = 'modelfit_CModel'
objectiveisinf = np.where(np.isinf(meastabledefault[cmodelpre + "_objective"]))[0]
expfailed = ~np.isfinite(meastabledefault[cmodelpre + "_exp_objective"]) & ~meastabledefault[cmodelpre + "_initial_flag"]
expfailednonnumericprimary = expfailed & ~meastabledefault[cmodelpre + "_dev_flag_numericError"] & measdefault['detect_isPrimary']
for fluxtype in ["slot_" + i + "Flux_flux" for i in ["Ap", "Psf"]]:
    print(meastabledefault[fluxtype][objectiveisinf])
    print(meastabledefault[fluxtype][expfailednonnumericprimary])


# In[120]:


# Display a cutout of the galaxy (not sure why it only shows the one instead of both - TBD)
#displaycutout = afwDisplay.getDisplay(frame=1, backend='ds9')

cenx = meastabledefault['slot_Centroid_x']
ceny = meastabledefault['slot_Centroid_y']
ra = meastabledefault['coord_ra']
dec = meastabledefault['coord_dec']

rowtarget = np.int(np.where(expfailednonnumericprimary)[0][20])
idtarget = meastabledefault["id"][rowtarget]

print('row={:d} id={:d}'.format(rowtarget, idtarget))
print("ra,dec: ", ra[rowtarget], ",", dec[rowtarget])
print("pixel x,y: ", cenx[rowtarget], ",", ceny[rowtarget])

# Get the bounding box of the exposure
bbox = calexp.getBBox()
print("bbox=", bbox)

# Get the footprint of this object and indices in exposure/frame units
footprint = meastabledefault[rowtarget].getFootprint()
indy, indx = footprint.spans.indices()
indy = np.array(indy)
indx = np.array(indx)
# Why is there still no minmax in numpy?!
miny, maxy = [np.min(indy), np.max(indy)]
minx, maxx = [np.min(indx), np.max(indx)]
indy -= bbox.getMinY()
indx -= bbox.getMinX()

sizecutout = 2*np.int64(np.ceil(np.max([
    cenx[rowtarget]-minx, maxx-cenx[rowtarget]
    ,ceny[rowtarget]-miny, maxy-ceny[rowtarget]
])))
print("sizecutout=", sizecutout, "xrange=[", minx, maxx, "]; yrange=[", miny, maxy, "]")

mask = calexp.getMaskedImage().getMask()
print("mask shape=", mask.array.shape)
var = calexp.getVariance()
print("variance shape=", var.array.shape)

# Could do this for calexp after running noiseReplacer too
cutout = cutouts.make_cutout_lsst(coords = [cenx[rowtarget], ceny[rowtarget]], coord_units="pixels", exp=calexp, size=sizecutout)
# Cutout the mask and the footprint, bookkeeping annoying pixel offsets
maskcutout = mask.array[cutout[4][3]:cutout[4][2], cutout[4][1]:cutout[4][0]]
varcutout = var.array[cutout[4][3]:cutout[4][2], cutout[4][1]:cutout[4][0]]
footcutout = maskcutout*0
footcutout[indy-cutout[4][3], indx-cutout[4][1]] = 1
withinfoot = footcutout.flatten()==1
maskfoot = maskcutout.flatten()[withinfoot]
print("Cutout within footprint: sum=", np.sum(cutout[0]), spstats.describe(cutout[0].flatten()[withinfoot]))
print("Mask within footprint", spstats.describe(maskfoot))
print("{:d}/{:d} unmasked".format(np.sum(maskfoot == 0),len(maskfoot)))
#plt.imshow(np.arcsinh(cutout[0]*5), cmap="gray", origin="bottomleft")
plt.imshow(np.arcsinh(cutout[0]*5*footcutout), cmap="gray", origin="bottomleft")
#plt.imshow(np.arcsinh(varcutout*footcutout), cmap="gray", origin="bottomleft")
#plt.imshow(np.arcsinh(cutout[0]*5*footcutout*(maskcutout==0)), cmap="gray", origin="bottomleft")


# In[6]:


# Print stats for PSF models used in cModel - these should not depend on config

measdefault = butlerdefault['measures']
tabdefault = measdefault.getTable()
namepsfmodel = "modelfit_DoubleShapeletPsfApprox"
flagpsfmodelprefix = namepsfmodel + "_flag"
columnspsf = tabdefault.getSchema().extract(namepsfmodel + "*")
primaryflag = measdefault['detect_isPrimary']
childflag = measdefault['deblend_nChild'] == 0

print('nisprimary=' + str(np.sum(primaryflag)) + "/" + str(len(primaryflag)))
print('nischild=' + str(np.sum(childflag)) + "/" + str(len(childflag)))

objectflags = {
    'any' : np.full(len(childflag), True, dtype=bool)
    ,'ischild' : childflag
    ,'isprimary' : primaryflag
}

psfstatsprint = True

# Print some PSF statistics - are the higher order moments actually used? Maybe, but they're mostly close to zero...
if psfstatsprint:
    for column in columnspsf:
        if column.startswith(flagpsfmodelprefix):
            print(column, np.sum(measdefault[column]))
        else:
            print(column, spstats.describe(measdefault[column]))


# In[46]:


# Print some rather ugly summaries of table data
# TODO: Make a nicely formatted table out of this (try pandas first, then astropy)
# TODO: Visualize said table

cmodelparams = config["params"]
objectivedefaultvalue = -5000
countformat = '{:6d}'

params = ['flag' + var for var in ([''] + ['_{0}'.format(name)     for name in ['apCorr','maxIter','numericError','trSmall']])] +     ['flux' + var for var in ['', 'Sigma', '_inner']] +     ['nIter', 'objective', 'time']

# Consider searching from schema if new flags added
baseflags = ["_flag" + post for post in [""] +
                ["_region" + rpost for rpost in ["_maxArea", "_maxBadPixelFraction"]] +
                ["_noShape", "_noShapeletPsf", "_badCentroid"]
            ]

printparamsummary = True

fittypes = ['initial', 'exp', 'dev']

for key, butlerdict in butlers.items():
    
    keydefault = cmodelparams[key]['default']
    meastabledefault = butlerdict[keydefault]['measures']
    filtdefault = butlerdict[keydefault]['measures'][cmodelpre + '_initial_nIter'] > 0
    objectivedefaultfinal = meastabledefault[cmodelpre + '_objective'][filtdefault]
    objectivedefaultfinal[~np.isfinite(objectivedefaultfinal)] = objectivedefaultvalue
    
    print(butlerdict.keys())
    
    for configparam in cmodelparams[key]['names']:       
        isdefault = key == keydefault
        table = butlerdict[configparam]['measures']
        
        for flagname, flag in objectflags.items():
            print('Processing ' + key + '=' + configparam + ", flag=" + flagname)
            for profile in fittypes:
                # Summary
                flagisset = table[cmodelpre + '_' + profile + '_flag'][flag]
                flux = table[cmodelpre + '_' + profile + '_flux'][flag]
                objective = table[cmodelpre + '_' + profile + '_objective'][flag]
                fluxisneg = flux < 0
                fluxisnan = np.isnan(flux)
                objectivenotfinite = ~np.isfinite(objective)
                print(
                    '{:s} objects: {:d}, flux<0: {:d},'
                    'nan-flux {:d}, non-finite objective: {:d}, '
                    'intersect: {:d}\n'.format(
                        profile, len(fluxisneg), np.sum(fluxisneg),
                        np.sum(fluxisnan),
                        np.sum(objectivenotfinite),
                        np.sum(fluxisnan*objectivenotfinite)
                    )
                )
                print(spstats.describe(flux[~fluxisnan]), '\n')

                # Flux estimate summaries
                fluxes = dict.fromkeys([
                    'slot_ApFlux'
                    #,'slot_ModelFlux,
                    ,'slot_PsfFlux'
                ])

                for fluxname, fluxmeasure in fluxes.items():
                    for fluxfilter, fluxcondition in {
                        False : None
                        ,True: objectivenotfinite
                    }.items():
                        if not fluxfilter:
                            x = table[fluxname + '_flux'][flag]
                            fluxes[fluxname] = x
                        else:
                            x = fluxes[fluxname][fluxcondition]
                        xisnan = np.isnan(x)
                        print(fluxname, end="")
                        if fluxfilter:
                            print('[objectivenotfinite]', end="")
                        print((' nisnan=' + countformat).format(np.sum(xisnan)))
                        print(spstats.describe(x[~xisnan]), '\n')               

                # Parameter summaries
                printparams = ['_' + profile + '_' + p for p in params]
                if profile == "initial":
                    printparams += baseflags
                for param in printparams:
                    if printparamsummary:
                        flagfield = cmodelpre + param
                        x = table[cmodelpre + param][flag]
                        lenx = len(x)
                        isnan = np.isnan(x)
                        nansum = np.sum(isnan)
                        isfinite = np.isfinite(x)
                        notfinitesum = np.sum(~isfinite[~isnan])
                        x = x[isfinite]
                        isflag = table.schema.find(flagfield).key.getTypeString() == "Flag"
                        output = ('{:32s}: nansum=' + countformat + '; nanfrac={:.3f}; notfinitesum=' +                                  countformat + '; notfinitefrac={:.3f}')
                        output = output.format(param, nansum, nansum/lenx, notfinitesum, notfinitesum/lenx)
                        if isflag:
                            output += ('; count='+countformat).format(np.sum(x == 1))
                        if param in baseflags:
                            flagintersect = np.intersect1d(np.where(x), np.where(flagisset))
                            output += ('; _initial_flag_intersect=' + countformat).format(len(flagintersect))
                        print(output)
                        if not isflag:
                            print(spstats.describe(x))
                print('\n')


# In[47]:


# How many nan fluxes are there with default settings and what fraction are near the edge?

cenx = meastabledefault['slot_Centroid_x']
ceny = meastabledefault['slot_Centroid_y']
ra = meastabledefault['coord_ra']
dec = meastabledefault['coord_dec']

print(spstats.describe(cenx))
print(spstats.describe(ceny))

minx = min(cenx)
maxx = max(cenx)
miny = min(ceny)
maxy = max(ceny)

nearedge = ((cenx-minx) < 50) | ((maxx - cenx) < 50)     | ((ceny-miny) < 50) | ((maxy - ceny) < 50)

for profile in fittypes:
    for flagname, flag in objectflags.items():
        print(profile + ', flag=' + flagname)
        flux = meastabledefault[cmodelpre + '_' + profile + '_flux'][flag]
        objective = meastabledefault[cmodelpre + '_' + profile + '_objective'][flag]
        fluxisnan = np.isnan(flux)
        print('n_nearedge={nedge:d}, n_isnan={nisnan:d}, intersect={nintersect:d}'.format(
            nedge = np.sum(nearedge[flag])
            , nisnan = np.sum(fluxisnan)
            , nintersect = np.sum(nearedge[flag]*fluxisnan)
        ))

badprimaries = np.where(np.isnan(meastabledefault[cmodelpre + '_flux']) & objectflags['isprimary'])[0]
print(meastabledefault['id'][badprimaries])


# In[109]:


# Display the image in ds9 - log scale with limits 0:30 works ok

display = afwDisplay.getDisplay(backend='ds9')
calexp = butlerdefault['calexp']
print(calexp)
calexpmakeplot = True
if calexpmakeplot:
    display.mtv(calexp)
    display.setMaskTransparency(60)


# In[9]:


data = np.zeros((10,10), dtype=np.float32)
img = afwImage.ImageF(data)
display = afwDisplay.getDisplay(backend='ds9')
display.mtv(img)


# In[48]:


# Mark failed initial fits in orange
primaryonlyplot = True
if primaryonlyplot:
    failed = badprimaries
else:
    failed = np.where(fluxisnan)[0]

with display.Buffering():
    for idx in failed:
        display.dot("o", cenx[idx], ceny[idx],
            size=6, ctype='orange')


# In[15]:


# Plot extreme changes in initial LP between two configurations as crosses
param = "modelfit_CModel_initial_objective"
lps = {
    'default': {'values': meastabledefault[param], 'symbol': "x"}
    , 'compare': {'values': butlers['nComps']['2']['measures'][param], 'symbol': "+"}
}

for key, value in lps.items():
    print(key, spstats.describe(value['values'][~fluxisnan]))

# LP(compare) - LP(default)
lpdiff = lps['default']['values'] - lps['compare']['values']
lpthresh = 100
print((~fluxisnan & (lpdiff < -lpthresh))[85])

displaylps = True
with display.Buffering():
    betterfits = {
        'default': np.where(~fluxisnan & (lpdiff < -lpthresh))[0] 
        , 'compare': np.where(~fluxisnan & (lpdiff > lpthresh))[0]
    }
    #print(np.intersect1d(betterfits['default'], betterfits['compare']))
    for key, value in betterfits.items():
        print(key, len(value))
        if displaylps:
            for idx in value:
                display.dot(lps[key]['symbol'], cenx[idx], ceny[idx],
                    size=6, ctype='yellow')


# In[59]:


# Make an exhausting but not exhaustive set of plots comparing results between configurations

key = 'sr1term'
butlerdict = butlers[key]
sns.set_style("darkgrid", {'axes.labelcolor': '.15'})

for paramstr in cmodelparams[key]['names']:
    paramdefault = cmodelparams[key]['default']
    isdefault = paramstr == paramdefault
    meastable = butlerdict[paramstr]['measures']
    
    calexp = butlerdict[paramstr]['calexp']
    calib = calexp.getCalib()
    filt = meastable['modelfit_CModel_initial_nIter'] > 0
    filtsum = np.sum(filt)
    print('{:s}={:s} vs default={:s}; '
        'n_(nIter>0)={:d}, intersect w/default={:d}'.format(
            key, paramstr, paramdefault,
            filtsum,
            np.sum(filt & filtdefault)
        )
    )
    title = key + '=' + paramstr
    
    # Total time vs niters
    (sns.jointplot(
        x = np.log10(meastable['modelfit_CModel_initial_nIter'][filt]),
        y = np.log10(meastable['modelfit_CModel_initial_time'][filt]),
        color="k", joint_kws={'marker':'.', 's':2}, marginal_kws={'hist_kws' : {'log':True}})
    ).set_axis_labels("log(nIter)", "log(time)")
    plt.title(title)
    # .plot_joint(sns.kdeplot, zorder=0, n_levels=10)
    
    # Average time per iter
    (sns.jointplot(
        x = np.log10(meastable['modelfit_CModel_initial_nIter'][filt]),
        y = np.log10(meastable['modelfit_CModel_initial_time'][filt]/
                   meastable['modelfit_CModel_initial_nIter'][filt]),
        color="k", joint_kws={'marker':'.', 's':2}, marginal_kws={'hist_kws' : {'log':True}})
    ).set_axis_labels("log(nIter)", "log(time/nIter)")
    plt.title(title)
    
    # Check running times
    timetotal = 0
    for fittype in fittypes:
        timefit = meastable['modelfit_CModel_' + fittype + '_time']
        timefitsum = np.sum(timefit)
        print('t_' + fittype + '_total=' + str(timefitsum))
        timetotal += timefitsum
    print('t_total=' + str(timetotal))
    
    # Compare objective (log probability) values with default setting
    if not isdefault:
        for objname, objcode in {"init": "_initial", "exp" : "_exp", "dev": "_dev", "final" : ""}.items():
            objective = "modelfit_CModel" + objcode + "_objective"
            y = meastable[objective][filt]
            
            if objname != "init":
                y[~np.isfinite(y)] = objectivedefaultvalue
            
            if objname == "final":
                # TOOD: replace with histogram
                #print(pd.value_counts(np.round(y,3)).sort_index())
                print('objective(' + objname + ')', spstats.describe(y))
                y2 = objectivedefaultfinal
            else:
                y2 = meastabledefault[objective][filt]
                if objname != 'init':
                    y2[~np.isfinite(y2)] = objectivedefaultvalue
            # the objective are -ln(L) so we want ln(objective) - ln(objectivedefault)
            y = y2 - y
                
            print('deltaobj(' + objname + ')', spstats.describe(y))
            loghist = objname == "init"
            
            x = np.log10(meastable['modelfit_CModel_initial_time'][filt])
            titlecompare = title + " vs. " + paramdefault + "(" + objname + ")"
            # Plot delta objective vs time
            (sns.jointplot(
                x = x
                , y = y, color="k"
                , joint_kws={'marker':'.', 's':8}
                , marginal_kws={'hist_kws' : {'log':loghist}}
            )).set_axis_labels("log(time)", "deltaLP(" + objname + ")")
            plt.title(titlecompare)
            y[y > 10] = 10
            y[y < -10] = -10
            
            # Same delta objective vs time with smaller limits
            (sns.jointplot(
                x = x
                , y = y, color="k"
                , joint_kws={'marker':'.', 's':8}
                , marginal_kws={'hist_kws' : {'log':loghist}}
            )).set_axis_labels("log(time)", "deltaLP(" + objname + ")")
            plt.title(titlecompare)
        
        # Compare final cModel mags
        x = meastable['modelfit_CModel_flux'][filt]
        plt.title(titlecompare)
        filtx = (x > 0) * np.isfinite(x)             *(meastabledefault['modelfit_CModel_flux'][filt] > 0)             *np.isfinite(meastabledefault['modelfit_CModel_flux'][filt])
        x = calib.getMagnitude(x[filtx])

        deltamag = x - calib.getMagnitude(meastabledefault            ['modelfit_CModel_flux'][filt][filtx])

        # Delta mag vs mag
        for magcrop in [False, True]:
            if magcrop:
                deltamag[deltamag > 0.1] = 0.1
                deltamag[deltamag < -0.1] = -0.1
            (sns.jointplot(
                x = x
                ,y = deltamag
                ,color="k"
                ,joint_kws={'marker':'.', 's':8}
                ,marginal_kws={'hist_kws' : {'log':loghist}}
            )).set_axis_labels("mag_cModel", "delta mag_cModel")
            plt.title(titlecompare)
            
    # Time spent in each fit stage
    timefittotal = np.zeros(filtsum)
    timefitdefaulttotal = np.zeros(filtsum)
    timemin = 1e-4

    for fittype in fittypes:
        timefitname = 'modelfit_CModel_' + fittype + '_time'
        # Should write a function for this and many things above
        timefit = meastable[timefitname][filt]
        timefittotal += timefit
        timefittoosmall = timefit < timemin
        timefitdefault = meastabledefault[timefitname][filt]
        timefitdefaulttotal += timefitdefault
        timefitdefaulttoosmall = timefitdefault < timemin
        print(timefitname,
            'n_(timefit<{:.2f})={:d}, '
            'n_(timefitdefault<{:.2f})={:d}, '\
            .format(
                timemin
                ,np.sum(timefittoosmall)
                ,timemin
                ,np.sum(timefitdefaulttoosmall)
            )
        )
        print(spstats.describe(timefit))
        timefittoosmall = timefittoosmall | timefitdefaulttoosmall
        timefit[timefittoosmall] = timemin
        timefitdefault[timefittoosmall] = timemin

        x = meastable['slot_PsfFlux_flux'][filt]
        filtx = x>0
        x[~filtx] = np.min(x[filtx])
        x = calib.getMagnitude(x)
        # Time vs flux
        (sns.jointplot(
            x = x
            ,y = np.log10(timefit)
            ,color="k"
            ,joint_kws={'marker':'.', 's':8}
            ,marginal_kws={'hist_kws' : {'log':loghist}}
        )).set_axis_labels('mag_psf', 'log(t_' + fittype + ')')
        plt.title(title)

        if not isdefault:
            
            y = np.log10(timefit/timefitdefault)
            
            # Delta time vs time
            (sns.jointplot(
                x = np.log10(timefit)
                , y = y
                , color="k"
                ,joint_kws={'marker':'.', 's':8}
                , marginal_kws={'hist_kws' : {'log':loghist}}
            )).set_axis_labels('log(t_' + fittype + ')', 'delta log(t)')
            plt.title(title + " vs. " + paramdefault + "(" + fittype + ")")
            
            # Delta time vs psfmag
            (sns.jointplot(
                x = x
                , y = y
                , color="k"
                , joint_kws={'marker':'.', 's':8}
                , marginal_kws={'hist_kws' : {'log':loghist}}
            )).set_axis_labels('mag_psf', 'delta log(t)')
            plt.title(title + " vs. " + paramdefault + "(" + fittype + ")")

    y = np.log10(timefittotal)
    # x = mag_psf here
    (sns.jointplot(
        x = x
        , y = y
        , color="k"
        , joint_kws={'marker':'.', 's':8}
        , marginal_kws={'hist_kws' : {'log':loghist}}
    )).set_axis_labels('mag_psf','log(t_total)')
    plt.title(title + "(total)")
    
    if not isdefault:
        logt = np.log10(timefitdefaulttotal)
        y -= logt
        (sns.jointplot(
            x = x
            , y = y
            , color="k"
            , joint_kws={'marker':'.', 's':8}
            , marginal_kws={'hist_kws' : {'log':loghist}}
        )).set_axis_labels('mag_psf','delta log(t_total)')
        plt.title(title + " vs. " + paramdefault + "(total)")
        
        (sns.jointplot(
            x = logt
            , y = y
            , color="k"
            , joint_kws={'marker':'.', 's':8}
            , marginal_kws={'hist_kws' : {'log':loghist}}
        )).set_axis_labels('log(t_total)', 'delta log(t_total)')
        plt.title(title + " vs. " + paramdefault + "(total)")
    
plt.show()

