import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy
import scipy.interpolate as interpolate
import scipy.stats
import scipy.spatial

from .zcolours import *
from .zvoronoi import *
from .zsubdiv import *

def set_mpl_defaults():
    # jupyter and matplotlib defaults
    plt.style.use('ggplot')

    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['axes.facecolor']='whitesmoke'
    mpl.rcParams['axes.edgecolor']='k'
    mpl.rcParams['axes.linewidth']=1
    mpl.rc('font',**{'family':'sans-serif','sans-serif':['Avenir']})
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.color'] = 'k'
    #mpl.rcParams['xtick.major.width'] = 2
    #mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['xtick.color']='k'
    mpl.rcParams['ytick.color']='k'
    mpl.rcParams['axes.labelcolor']='k'

    mpl.rcParams['font.size']=12
    mpl.rcParams['xtick.direction']='in'
    mpl.rcParams['ytick.direction']='in'
    mpl.rcParams['xtick.major.size'] = 5.5
    mpl.rcParams['ytick.major.size'] = 5.5
    mpl.rcParams['xtick.minor.size'] = 3.5

def make_plot(figsize=(10,6),xlabel='',ylabel='',xscale='linear',yscale='linear'):
    fig=plt.figure(figsize=figsize)
    ax=plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    return fig,ax

def make_grid(height,width,figsize=None,thisAx=None,wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    if thisAx==None:
        if figsize==None:
            figsize=(4*width,4*height)
        fig=plt.figure(figsize=figsize)
    grid=mpl.gridspec.GridSpec(width,height,figure=fig,wspace=wspace,hspace=hspace,width_ratios=width_ratios,height_ratios=height_ratios)
    return fig,grid

def subgrid(grid,iVals,jVals):
    return plt.subplot(grid[jVals,iVals])

def centeraxis(xlabel='',ylabel='',ax=None):
    if ax==None:
        ax=plt.gca()
    ax.patch.set_alpha(0)
    ax.grid(False)
    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.annotate(xlabel, xy=(1, 0), xycoords=('axes fraction', 'data'),
            ha='left', va='bottom')
    ax.annotate(ylabel, xy=(0, 1), xycoords=('data', 'axes fraction'),
            ha='left', va='bottom')
    return ax

# kind of a replica of sigRound below - but one i've been using more often
def sigstr(x, xerr=None, sig=2):
    if x==0:
        return str(0)
    if xerr is not None:
        power=sig-int(np.floor(np.log10(abs(xerr))))-1
        return str(round(x, power)),str(round(xerr, power))
    else:
        power=sig-int(np.floor(np.log10(abs(x))))-1
        return str(round(x,power))

# rounds a number to a given number of sigFigs
def sigRound(number,sigFigs=2,extra=False): # returns a number to a given significant digits (if extra is true also returns base of first significant figure)
    roundingFactor=sigFigs - int(np.floor(np.log10(np.abs(number)))) - 1
    rounded=np.round(number, roundingFactor)
    # np.round retains a decimal point even if the number is an integer (i.e. we might expect 460 but instead get 460.0)
    if roundingFactor<=0:
        rounded=rounded.astype(int)
    if extra==False:
        return rounded
    if extra==True:
        return rounded,roundingFactor

# turns an ax object into a colorbar
def make_cbar(ax,cmap,cvals=None,vmin=0,vmax=1,scale='linear',label="",\
    orientation='vertical',position='right',labelpad=10):
    if cvals==None:
        cvals=np.linspace(0,1,256)
    if scale=='log':
        zvals=10**(np.linspace(np.log10(vmin),np.log10(vmax),257))
    else:
        zvals=np.linspace(vmin,vmax,257)

    if orientation=='vertical':
        ax.pcolormesh(np.array([0,1]),zvals,\
            np.swapaxes(np.array([cmap(cvals)]),0,1))
        ax.set_xticks([])
        if position=='right':
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_ylabel(label,rotation=270,labelpad=2*labelpad)
        else:
            ax.set_ylabel(label,labelpad=labelpad)
        ax.tick_params(axis="y",direction="out",which='both')
        ax.set_yscale(scale)

    if orientation=='horizontal':
        ax.pcolormesh(zvals,np.array([0,1]),\
            np.array([cmap(cvals)]))
        ax.set_yticks([])
        if position=='top':
            #ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
        ax.set_xlabel(label,labelpad=labelpad)
        ax.tick_params(axis="x",direction="out",which='both')
        ax.set_xscale(scale)

def zMap(xs,ys,vals=[],showContours=False,showColorbar=True,nContours=10,thisPlot=None,xRange=None,yRange=None,aspect='',colourFunc=np.median,hexKwargs={},contourKwargs={}):
    if thisPlot==None:
        thisPlot=plt.gca()

    # finding xRange and yRange (if unspecified gives whole range, if scalar finds smallest interval containing range)
    if xRange is None:
        xRange = [xs.min(), xs.max()]
    elif len(xRange)==1:
        q = np.array([0.5 - 0.5*xRange, 0.5 + 0.5*xRange])
        xRange = np.percentile(xs, list(100.0 * q))
    if yRange is None:
        yRange = [ys.min(), ys.max()]
    elif len(yRange)==1:
        q = np.array([0.5 - 0.5*yRange, 0.5 + 0.5*yRange])
        yRange = np.percentile(ys, list(100.0 * q))
    if 'extent' not in hexKwargs:
        hexKwargs['extent']=np.hstack((xRange,yRange)).flatten()
    #print(hexKwargs['extent'])

    nPoints=np.flatnonzero((xs>xRange[0]) & (xs<xRange[1]) & (ys>yRange[0]) & (ys<yRange[1])).size
    nBins=np.ceil(min(np.log2(np.sqrt(nPoints))+1,2*np.power(nPoints,1/6))).astype(int)
    if 'gridsize' not in hexKwargs:
        hexKwargs['gridsize']=(nBins,nBins)
    else:
        nBins=min(hexKwargs['gridsize'])
    if 'cmap' not in hexKwargs:
        hexKwargs['cmap']='bone_r'
    if 'linewidths' not in hexKwargs:
        hexKwargs['linewidths']=0.2
    if len(vals)==0:
        im=thisPlot.hexbin(xs,ys,**hexKwargs)
    else:
        im=thisPlot.hexbin(xs,ys,C=vals,reduce_C_function=colourFunc,**hexKwargs)
    if showColorbar:
        plt.colorbar(im)

    if showContours!=False: # plots smoothed contours enclosing some fraction of the data
        counts,xbins,ybins=np.histogram2d(xs,ys,bins=nBins,range=[xRange,yRange])
        counts=counts/counts.sum()
        #scipy.ndimage.zoom(counts.T, int(np.sqrt(nBins)))
        smoothCounts = scipy.ndimage.gaussian_filter(counts, sigma=1.0, order=0)


        n = nBins
        t = np.linspace(0, smoothCounts.max(), n)
        integral = ((smoothCounts.T >= t[:, None, None]) * smoothCounts.T).sum(axis=(1,2))

        f = interpolate.interp1d(integral, t, fill_value="extrapolate")
        if showContours=='log':
            percentiles=10**np.linspace(np.log10(np.min(integral[integral>0])),np.log10(np.max(integral)),nContours+1)
        else:
            percentiles=((1+np.arange(nContours-1))/nContours)[::-1]
        #print('percentiles: ',percentiles)
        t_contours = f(percentiles)
        if ~np.isfinite(np.sum(t_contours)):
            percentiles=percentiles[np.isfinite(t_contours)]
            t_contours=t_contours[np.isfinite(t_contours)]
        t_contours=np.sort(t_contours)
        if t_contours.size==nContours:
            thisPlot.contour(smoothCounts.T, t_contours, extent=np.hstack((xRange,yRange)).flatten(),colors='k',linewidths=1)#2*(1-percentiles))
        else:
            thisPlot.contour(smoothCounts.T, t_contours[:-2], extent=np.hstack((xRange,yRange)).flatten(),colors='k',linewidths=1)#2*(1-percentiles[:-2]))
            thisPlot.contour(smoothCounts.T, t_contours[-2:], extent=np.hstack((xRange,yRange)).flatten(),colors='k',linewidths=1)#2*(1-percentiles[-2:]),ls=':')


    if aspect=='equal':
        thisPlot.set_aspect('equal')
    elif aspect=='square':
        thisPlot.set_aspect(1.0/thisPlot.get_data_ratio())

    thisPlot.set_xlim(xRange)
    thisPlot.set_ylim(yRange)
    return thisPlot

def zcorner(params,span=None):
    nParams=len(params)
    fig,grid=makeGrid(nParams,nParams)
    i,j=0,0
    stateList=[]
    rangeList=[]
    for i in range(nParams):
        stateList.append(params[i].map(params[i].states.flatten()))
        #stateList.append(params[i].states.flatten())
        rangeList.append(cInterval(stateList[i],interval=span))
    for i in np.arange(nParams):
        print(i)
        sis=stateList[i]
        thisPlot=subGrid(grid,i,i)
        thisPlot.hist(sis,histtype='step',edgecolor='k',bins=32,density=True,range=rangeList[i])
        if i==nParams-1:
            thisPlot.set_xlabel(params[i].name,fontsize='small')
        else:
            thisPlot.set_xticks([])
        thisPlot.set_yticks([])

        for j in np.arange(i+1,nParams):
            sjs=stateList[j]
            thisPlot=subGrid(grid,i,j)
            zMap(sis,sjs,showContours=True,xRange=rangeList[i],yRange=rangeList[j],hexKwargs={},showColorbar=False)
            if i==0:
                thisPlot.set_ylabel(params[j].name,fontsize='small')
            else:
                thisPlot.set_yticks([])
            if j==nParams-1:
                thisPlot.set_xlabel(params[i].name,fontsize='small')
            else:
                thisPlot.set_xticks([])
    return fig,grid

def zerrline(xs,ys,yerrs,ax=None,plotkwargs={},fillkwargs={}):
    if ax==None:
        ax=plt.gca()
    ax.plot(xs,ys,**plotkwargs)
    xFills=np.hstack([xs,np.flip(xs)])
    if type(yerrs)==list:
        yFills=np.hstack([ys+yerrs[0],np.flip(ys-yerrs[1])])
    else:
        yFills=np.hstack([ys+yerrs,np.flip(ys-yerrs)])
    ax.fill_between(xFills,yFills,**fillkwargs)
    return ax

def cplot(xs,ys,cs=0,cmap='Spectral',ax=None,lw=1,ls='-',alpha=1,zorder=2,vmin=None,vmax=None,cscale='linear'):
    if ax==None:
        ax=plt.gca()
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if hasattr(cs, "__len__")==False:
        cs=cs*np.ones_like(xs) # if a single value passed use cmap(cs) for all colours
        clim=(0,1)
    else:
        if vmin==None:  vmin=np.min(cs)
        if vmax==None:  vmax=np.max(cs)

        if cscale=='log':
            cs=np.log10(cs)
            vmin=np.log10(vmin)
            vmax=np.log10(vmax)

        clim=(vmin,vmax)

    lc = mpl.collections.LineCollection(segments)
    lc.set(array=cs,alpha=alpha,cmap=cmap,zorder=zorder,lw=lw,ls=ls,clim=clim)
    ax.add_collection(lc)
    ax.autoscale() # currently broken for log scale axis (matplotlib's fault!)
    return ax

def intervalplot(xs,ys,interval,ax=None,plotmedian=False,c='darkblue',alpha=0.4,xbins=None,xmin=None,xmax=None,nbins=10,xscale='linear'):
    if ax==None:
        ax=plt.gca()
    xbins,ymeds,yints=binned_median(xs,ys,xbins=xbins,xmin=xmin,xmax=xmax,nbins=nbins,xscale=xscale,interval=interval)
    for i in range(xbins.size -1):
        ax.fill_between([xbins[i],xbins[i+1]],yints[i][0],yints[i][1],color=c,alpha=alpha)
    if plotmedian==True:
        if xscale=='log':
            ax.plot(10**(0.5*(np.log10(xbins[1:])+np.log10(xbins[:-1]))),ymeds,c=c)
        else:
            ax.plot(0.5*(xbins[1:]+xbins[:-1]),ymeds,c=c)
    if xscale=='log':
        ax.set_xscale('log')
    return ax

def cInterval(xs,interval=0.95):
    #print(interval)
    #print('xs: ',xs)
    xs=np.sort(xs)
    nData=xs.size
    width=int(nData*interval)
    #print('width: ',width)
    #print('int((1-interval)*nData): ',int((1-interval)*nData))
    ks=np.arange(int((1-interval)*nData))
    #print('ks: ',ks)
    deltas=xs[ks+width]-xs[ks]
    #print('deltas: ',deltas)
    K=np.argmin(deltas)
    return xs[K],xs[K+width]

def binned_median(xs,ys,xbins=None,xmin=None,xmax=None,nbins=10,xscale='linear',interval=None):
    if xscale=='log':
        xs=np.log10(xs)
    if xbins==None:
        if xmin==None:
            xmin=np.min(xs)
        if xmax==None:
            xmax=np.max(xs)
        xbins=np.linspace(xmin,xmax,nbins+1)
    #xmids=0.5*(xbins[1:]+xbins[:-1])
    ymeds=np.zeros(xbins.size-1)
    if interval!=None:
        yints=np.zeros((xbins.size-1,2))
    for i in range(nbins):
        inbin=np.flatnonzero((xs>xbins[i]) & (xs<xbins[i+1]))
        if inbin.size==0:
            continue
        ymeds[i]=np.median(ys[inbin])
        if (interval!=None) & (inbin.size>2):
            yints[i,:]=cInterval(ys[inbin],interval=interval)
    if xscale=='log':
        xbins=10**xbins
    if interval!=None:
        return xbins,ymeds,yints
    else:
        return xbins,ymeds
