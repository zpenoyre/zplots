import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy
import scipy.interpolate as interpolate
import scipy.stats
import scipy.spatial

import zpy.zstats as zstats

def setMplDefaults():
    # jupyter and matplotlib defaults
    plt.style.use('ggplot')
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['axes.facecolor']='white'
    mpl.rcParams['axes.edgecolor']='white'
    mpl.rcParams['axes.linewidth']=3

    mpl.rcParams['text.color'] = 'dimgrey'
    #mpl.rcParams['xtick.major.width'] = 2
    #mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['xtick.color']='k'
    mpl.rcParams['ytick.color']='k'
    mpl.rcParams['axes.labelcolor']='k'

    mpl.rcParams['font.size']=14
    mpl.rcParams['xtick.direction']='in'
    mpl.rcParams['ytick.direction']='in'
    mpl.rcParams['xtick.major.size'] = 5.5
    mpl.rcParams['ytick.major.size'] = 5.5
    mpl.rcParams['xtick.minor.size'] = 3.5
    mpl.rcParams['ytick.minor.size'] = 3.5

    mpl.rcParams["text.usetex"] = True

def makePlot(figsize=(10,6)):
    fig=plt.figure(figsize=figsize)
    ax=plt.gca()
    return fig,ax

def makeGrid(height,width,figsize=None,thisAx=None):
    if thisAx==None:
        if figsize==None:
            figsize=(2*width,2*height)
        fig=plt.figure(figsize=figsize)
    grid=mpl.gridspec.GridSpec(width,height,wspace=0,hspace=0)
    return fig,grid

def subGrid(grid,iVals,jVals):
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
def make_cbar(ax,cmap,cvals=None,vmin=0,vmax=1,log=False,label="",\
    orientation='vertical',position='right',labelpad=10):
    if cvals==None:
        cvals=np.linspace(0,1,256)
    if log==True:
        zvals=10**(np.linspace(vmin,vmax,257))
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
        if log==True:
            ax.set_yscale('log')

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
        if log==True:
            ax.set_xscale('log')

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

def zerrline(xs,ys,yerrs,ax=None,plotKwargs={},fillKwargs={}):
    if ax==None:
        ax=plt.gca()
    ax.plot(xs,ys,**plotKwargs)
    xFills=np.hstack([xs,np.flip(xs)])
    if type(yerrs)==list:
        yFills=np.hstack([ys+yerrs[0],np.flip(ys-yerrs[1])])
    else:
        yFills=np.hstack([ys+yerrs,np.flip(ys-yerrs)])
    ax.fill_between(xFills,yFills,alpha=0.5,**fillKwargs)
    return ax

def cplot(xs,ys,cs,cmap='Spectral',ax=None,lw=1,ls='-',alpha=1,zorder=2):
    if ax==None:
        ax=plt.gca()
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = mpl.collections.LineCollection(segments)
    lc.set(array=cs,alpha=alpha,cmap=cmap,zorder=zorder,lw=lw,ls=ls)
    ax.add_collection(lc)
    plt.autoscale()
    return ax

def makeCmap(hexColour,cVal=[0,1]):
    if type(hexColour) is str:
        #just one colour provided so makes map from white to colour
        hexColour=['#FFFFFF',hexColour]
        cVal=[0,1]
    rList,gList,bList=[],[],[]
    for i in range(len(hexColour)):
        r,g,b=mpl.colors.hex2color(hexColour[i])
        rList.append((cVal[i],r,r))
        bList.append((cVal[i],b,b))
        gList.append((cVal[i],g,g))
    cdict = {
        'red':   tuple(rList),
        'green': tuple(gList),
        'blue':  tuple(bList)
        }
    cmap = mpl.colors.LinearSegmentedColormap('',cdict)
    return cmap

def colourGrid(colours,x,y): #returns a colour from a position (x,y : 0-1) on a grid of 4 (colours : hex strings)
    # changed order so theat colours go clockwise
    rnw,gnw,bnw=mpl.colors.hex2color(colours[3])
    rne,gne,bne=mpl.colors.hex2color(colours[0])
    rsw,gsw,bsw=mpl.colors.hex2color(colours[2])
    rse,gse,bse=mpl.colors.hex2color(colours[1])
    r=rne*x*y + rnw*(1-x)*y + rsw*x*(1-y) + rse*(1-x)*(1-y)
    g=gne*x*y + gnw*(1-x)*y + gsw*x*(1-y) + gse*(1-x)*(1-y)
    b=bne*x*y + bnw*(1-x)*y + bsw*x*(1-y) + bse*(1-x)*(1-y)
    if r>1:
        r=1
    elif r<0:
        r=0
    if g>1:
        g=1
    elif g<0:
        g=0
    if b>1:
        b=1
    elif b<0:
        b=0
    return (r,g,b,1.)

def cGrid(x,y,cs=['#ae2012','#ffb703','#023047','#0a9396']): #returns a colour from a position (x,y : 0-1) on a grid of 4 (colours : hex strings)
    rnw,gnw,bnw=mpl.colors.hex2color(cs[0])
    rne,gne,bne=mpl.colors.hex2color(cs[1])
    rsw,gsw,bsw=mpl.colors.hex2color(cs[2])
    rse,gse,bse=mpl.colors.hex2color(cs[3])
    r=rne*x*y + rnw*(1-x)*y + rsw*x*(1-y) + rse*(1-x)*(1-y)
    g=gne*x*y + gnw*(1-x)*y + gsw*x*(1-y) + gse*(1-x)*(1-y)
    b=bne*x*y + bnw*(1-x)*y + bsw*x*(1-y) + bse*(1-x)*(1-y)
    if r>1:
        r=1
    elif r<0:
        r=0
    if g>1:
        g=1
    elif g<0:
        g=0
    if b>1:
        b=1
    elif b<0:
        b=0
    return (r,g,b,1.)

def starTempRgb(temp):
    """
    Converts from K to RGB, algorithm courtesy of
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    adapted to work with either single temps or arrays of temps
    """
    singlevalue=0
    if np.array(temp).size==1:
        singlevalue=1
        temp=np.array([temp,3000])

    temp=temp
    temp[temp<1000]=1000
    temp[temp>40000]=40000

    tmp_internal = temp / 100.0

    # red
    red=np.zeros(temp.size)
    red[tmp_internal<=66]=255
    red[tmp_internal>66]=329.698727446 * np.power(tmp_internal[tmp_internal>66] - 60, -0.1332047592)
    red[red<0]=0
    red[red>255]=255

    # green
    green=np.zeros(temp.size)
    green[tmp_internal<=66]=99.4708025861 * np.log(tmp_internal[tmp_internal<=66]) - 161.1195681661
    green[tmp_internal>66]=288.1221695283 * np.power(tmp_internal[tmp_internal>66] - 60, -0.0755148492)
    green[green<0]=0
    green[green>255]=255

    # blue
    blue=np.zeros(temp.size)
    blue[tmp_internal>=66]=255
    blue[tmp_internal<=19]=0
    blue[(tmp_internal<66) & (tmp_internal>19)]=\
        138.5177312231 * np.log(tmp_internal[(tmp_internal<66) & (tmp_internal>19)] - 10) - 305.0447927307
    blue[blue<0]=0
    blue[blue>255]=255

    output=np.vstack([red/255, green/255, blue/255, np.ones(temp.size)]).T
    if singlevalue==1:
        output=output[0,:]
    return output

def voronoibin(x,y,C=None,ncells=None,grid=False,random=False, bins=None,
            xscale='linear', yscale='linear',mincnt=None,extent=None,scatter=True,
            mask=False,maskalpha=1,
            ax=None,cmap=mpl.cm.Spectral_r,vmin=None,vmax=None,reduce_C_function=np.mean,
            linewidth=1,linecol='k',alpha=1):
    """
    A 2D plot of data binned into Voronoi cells (calculated on a subset of
    data points specified by ncells). C defines the data to be averaged.
    Made with specific reference to matplotlib's hexbin() function.
    Parameters
    ----------
    x, y : array-like
        The data positions. *x* and *y* must be of the same length.
    C : array-like, optional
        These values are accumulated in the bins. Must be of the same length as *x*
        and *y*.
    ncells : int or (int, int), default: sqrt(x.size)
        The number of random points chosen from the data to construct Voroni cells.
    grid : bool or int, default: False
        Adds a grid of points for Voronoi cell calculations spanning plot extent
        (ensures a relatively even coverage of the plot area).
        If integer value is given then the grid is of size
        int*int. Otherwise if simply true it is of size approximately ncells (must be
        a square number)
    random : bool or int, default: False
        Similar to grid but instead of regularly spaced points generates uniformly
        distributed random points. If an integer value is provided that number of
        points will be generated, otherwise if simply true ncells points are used.
    bins : 'log' or None, default: None
        Discretization of the hexagon values.
        - If *None*, no binning is applied; the color of each hexagon
          directly corresponds to its count value.
        - If 'log', use a logarithmic scale for the colormap.
          Internally, :math:`log_{10}(i+1)` is used to determine the
          hexagon color. This is equivalent to ``norm=LogNorm()``.
    xscale : {'linear', 'log'}, default: 'linear'
        Use a linear or log10 scale on the horizontal axis.
    yscale : {'linear', 'log'}, default: 'linear'
        Use a linear or log10 scale on the vertical axis.
    mincnt : int > 0, default: *None*
        If not *None*, only display cells with more than *mincnt*
        number of points in the cell.
    extent : 4-tuple of float, default: *None*
        The limits of the bins (xmin, xmax, ymin, ymax).
        The default assigns the limits based on
        *gridsize*, *x*, *y*, *xscale* and *yscale*.
        If *xscale* or *yscale* is set to 'log', the limits are
        expected to be the exponent for a power of 10. E.g. for
        x-limits of 1 and 50 in 'linear' scale and y-limits
        of 10 and 1000 in 'log' scale, enter (1, 50, 1, 3).
    scatter : bool, default : True
        Plots scatter points of data in cells with less than mincnt members.
        If C is specified will use same cmap, otherwise black.
    mask : bool or int, default: True
        Overplots a hexbin mask, such that empty regions of the plotting space
        (which may still fall within a Voronoi cell) are shown as blank. If
        an integer value is given it will be used as the hexbin gridsize, otherwise
        if simply true a default of 64 is used.
    Returns
    -------
    I'm honestly not sure
    Other Parameters
    ----------------
    ax : axis object, default: None
        If supplied will plot on given axis object, otherwise will generate and return a new one.
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The Colormap instance or registered colormap name used to map
        the bin values to colors.
    TODO norm : `~matplotlib.colors.Normalize`, optional
        The Normalize instance scales the bin values to the canonical
        colormap range [0, 1] for mapping to colors. By default, the data
        range is mapped to the colorbar range using linear scaling.
    vmin, vmax : float, default: None
        The colorbar range. If *None*, suitable min/max values are
        automatically chosen by the `.Normalize` instance (defaults to
        the respective min/max values of the bins in case of the default
        linear scaling).
        It is an error to use *vmin*/*vmax* when *norm* is given.
    TODO alpha : float between 0 and 1, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    TODO linewidths : float, default: *None*
        If *None*, defaults to 1.0.
    TODO edgecolors : {'face', 'none', *None*} or color, default: 'face'
        The color of the hexagon edges. Possible values are:
        - 'face': Draw the edges in the same color as the fill color.
        - 'none': No edges are drawn. This can sometimes lead to unsightly
          unpainted pixels between the hexagons.
        - *None*: Draw outlines in the default color.
        - An explicit color.
    reduce_C_function : callable, default: `numpy.mean`
        The function to aggregate *C* within the bins. It is ignored if
        *C* is not given. This must have the signature::
            def reduce_C_function(C: array) -> float
        Commonly used functions are:
        - `numpy.mean`: average of the points
        - `numpy.sum`: integral of the point values
        - `numpy.amax`: value taken from the largest point
    See Also
    --------
    matplotlib.axes.hexbin
    """
    # copied (and slightly adjusted) from mpl.axes.hexbin()

    xorig=np.array(x)
    yorig=np.array(y)

    if xscale == 'log':
        if np.any(x <= 0.0):
            raise ValueError("x contains non-positive values, so can not"
                             " be log-scaled")
        x = np.log10(x)
    if yscale == 'log':
        if np.any(y <= 0.0):
            raise ValueError("y contains non-positive values, so can not"
                             " be log-scaled")
        y = np.log10(y)

    if extent is not None:
        xmin, xmax, ymin, ymax = extent
    else:
        xmin, xmax = (np.min(x), np.max(x)) if len(x) else (0, 1)
        ymin, ymax = (np.min(y), np.max(y)) if len(y) else (0, 1)
    xspan=xmax-xmin
    yspan=ymax-ymin

    if mincnt is None:
        mincnt = 1

    if ncells is None:
        ncells=2*int(np.sqrt(x.size))
        print(ncells,'cells used')
    if ncells>x.size:
        raise ValueError("number of cells must be less or equal to the number of data points")

    x=np.array((x-xmin)/xspan)
    y=np.array((y-ymin)/yspan)

    if C is not None:
        inds=np.flatnonzero(~np.isnan(x) & ~np.isnan(y) & ~np.isnan(C))
    else:
        inds=np.flatnonzero(~np.isnan(x) & ~np.isnan(y))

    sel=np.random.choice(inds,ncells,replace=False)
    points=np.vstack([x[sel],y[sel]]).T
    # gridded array of points (either same approx number as ncells or specified with integer)
    if grid != False:
        gridsize=np.floor(np.sqrt(ncells)+0.5)
        if type(grid)==int:
            gridsize=grid
        xygrid=np.mgrid[xmin:xmax:(xmax-xmin)/gridsize,ymin:ymax:(ymax-ymin)/gridsize]
        xypoints=np.vstack([np.hstack(xygrid[0]),np.hstack(xygrid[1])]).T
        points=np.vstack([points,xypoints])
    if random != False:
        randsize=ncells
        if type(random)==int:
            randsize=random
        xrand=xmin+(xmax-xmin)*np.random.rand(randsize)
        yrand=ymin+(ymax-ymin)*np.random.rand(randsize)
        points=np.vstack([points,np.vstack([xrand,yrand]).T])

    allpoints=np.vstack([x[inds],y[inds]]).T

    vor =scipy.spatial.Voronoi(points)
    mat = scipy.spatial.distance_matrix(vor.points,allpoints)
    whichcell=np.argmin(mat,axis=0)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    ncelltotal=len(vor.points)
    cvals=np.zeros(ncelltotal)

    if ax==None: ax=plt.gca()
    flagged=np.zeros(inds.size)
    # probably v. innefficient - loops through each Voronoi cell to find color and draw
    for i in range(ncelltotal):
        incell=np.flatnonzero(whichcell==i)
        if incell.size<mincnt:
            flagged[incell]=1
            continue
        if bins=='log':
            if vmin==None: vmin=np.min(C[C>0])
            if vmax==None: vmax=np.max(C)
            c=reduce_C_function(C[inds[incell]])
            if c<0:
                continue
            cvals[i]=(np.log10(c)-np.log10(vmin))/(np.log10(vmax)-np.log10(vmin))
        else:
            if vmin==None: vmin=np.min(C)
            if vmax==None: vmax=np.max(C)
            c=reduce_C_function(C[inds[incell]])
            cvals[i]=(c-vmin)/(vmax-vmin)

        if type(cmap)==str:
            cmap=mpl.cm.get_cmap(name=cmap)
        col=cmap(cvals[i])

        polygon = np.array(vertices[regions[i]])
        #print(polygon)
        polygon[:,0]=(xspan*polygon[:,0]) + xmin
        polygon[:,1]=(yspan*polygon[:,1]) + ymin
        if xscale=='log':
            polygon[:,0]=10**polygon[:,0]
        if yscale=='log':
            polygon[:,1]=10**polygon[:,1]
        #print(polygon)


        ax.fill(*zip(*polygon),color=col,zorder=0.1,
        linewidth=linewidth,edgecolor=linecol,alpha=alpha)

    # plots scatter points for data in cells with less than mincnt members
    if scatter==True:
        c='k'
        if C is not None:
            if bins=='log':
                cvals=(np.log10(C[inds[flagged==1]])-np.log10(vmin))/(np.log10(vmax)-np.log10(vmin))
            else:
                cvals=(C[inds[flagged==1]]-vmin)/(vmax-vmin)
            c=cmap(cvals)
        ax.scatter(xorig[inds[flagged==1]],yorig[inds[flagged==1]],c=c,s=10,zorder=10,edgecolor='k')

    # plots a hexbin mask over empty regions of plotting space
    if mask!=False:
        gridsize=50
        if type(mask)==int:
            gridsize=mask

        bw = ((0,1,1),(1,1,1))
        cdict = {'red':  bw,'green': bw,'blue': bw,'alpha': ((0.0, 1.0, maskalpha),(1.0, 0.0, 0.0))}
        dropout = mpl.colors.LinearSegmentedColormap('Dropout', cdict)
        plt.register_cmap(cmap = dropout)
        hex_extent=[xmin-0.1*xspan,xmax+0.1*xspan,ymin-0.1*yspan,ymax+0.1*yspan]
        ax.hexbin(xorig,yorig,bins='log',cmap=dropout,gridsize=gridsize,
                  extent=hex_extent,vmax=2,xscale=xscale,yscale=yscale,linewidth=0)

    if xscale=='log':
        ax.set_xlim(10**(xmin-0.1*xspan),10**(xmax+0.1*xspan))
        ax.set_xscale('log')
    else:
        ax.set_xlim(xmin-0.1*xspan,xmax+0.1*xspan)
    if yscale=='log':
        ax.set_ylim(10**(ymin-0.1*yspan),10**(ymax+0.1*yspan))
        ax.set_yscale('log')
    else:
        ax.set_ylim(ymin-0.1*yspan,ymax+0.1*yspan)
    # experimental way to return mappable for colorbars
    #collection = mpl.collections.PolyCollection([])
    #collection.set_cmap(cmap)
    #collection.set_norm(norm)
    return ax,vor,polygon

# utility for voronoi plots - taken directly from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def voronoigrid(xs,ys,zs,xrange=None,yrange=None,npoints=1000,ngrid=100,repeat=10,stat=np.median):
    if xrange is None:
        xrange=[np.min(xs),np.max(xs)]
    if yrange is None:
        yrange=[np.min(ys),np.max(ys)]

    xscaled=np.array(xs-xrange[0])/(xrange[1]-xrange[0])
    yscaled=np.array(ys-yrange[0])/(yrange[1]-yrange[0])

    #zmeans=np.zeros_like(zs)
    cellvals=np.zeros(npoints)

    test_points = np.vstack([xscaled,yscaled]).T
    grid_points_x,grid_points_y = np.meshgrid(np.linspace(0,1,ngrid), np.linspace(0,1,ngrid))

    grid_points=np.vstack([grid_points_x.ravel(), grid_points_y.ravel()]).T
    grid_vals = np.zeros((ngrid**2,repeat))
    grid_meds = np.zeros(ngrid**2)

    for i in range(repeat):
        subsel=np.random.choice(np.arange(xs.size),size=npoints,replace=False)

        voronoi_points = np.vstack([xscaled[subsel],yscaled[subsel]]).T

        voronoi_kdtree = scipy.spatial.KDTree(voronoi_points)

        test_point_dist, test_point_regions = voronoi_kdtree.query(test_points, k=1)
        grid_point_dist, grid_point_regions = voronoi_kdtree.query(grid_points, k=1)

        for j in range(npoints):
            incell=np.flatnonzero(test_point_regions==j)
            #if incell.size==0:
            #    print(voronoi_points[j,:])
            gridincell=np.flatnonzero(grid_point_regions==j)
            #print(zvals[incell])
            cellvals[j]=stat(zs[incell])
            grid_vals[gridincell,i]=cellvals[j]
    grid_meds=np.median(grid_vals,axis=1)

    gridxs=grid_points[:,0]*(xrange[1]-xrange[0])+xrange[0]
    gridys=grid_points[:,1]*(yrange[1]-yrange[0])+yrange[0]

    return gridxs,gridys,grid_meds

def subdiv(func,nvals,nlevels,nx0,ny0,xmin,xmax,ymin,ymax,
           power=1,tol=0.25,xwrap=-1,ywrap=-1,divneigh=-1,printncall=-1):
    """
    subdivides an area (xmin->xmax,ymin->ymax) from a grid (nx0,ny0) into rectangles
    up to 2**nlevels smaller based on deviations from linearity of a func(x,y) which
    returns nval values
    ----------
    Returns
    -------
    xmids,ymids,subs,vals - each a list of n arrays of size nx0*(2**n), ny0*(2**n)
    containg the midpoint of each cell, whether it should be further subdivided
    (subs[n][i,j]=1), if this is the top level subdivision (subs[n][i,j]=0), or
    if it should not be subdividied at this level (subs[n][i,j]=-1) and the
    values of func for any cell for which it was calculated

    ----------------
    """
    # subs tells if each cell on each level should be subdivied (1), exist (0), or not exist (-1)
    global xmids,ymids,subs,vals,ncall,neigh,local
    subs=[]
    # vals contains nvals entries for every cell used on every level (-1 if unused)
    #global vals
    vals=[]
    #global ncall
    ncall=0

    xmids=[]
    ymids=[]

    neigh=np.array([[1,0],[0,1],[-1,0],[0,-1]])
    local=np.array([[0,0],[0,1],[1,0],[1,1]])

    # calculates coords, and initializes arrays for all subs and vals for every level
    for n in range(nlevels):
        print('__n: ',n)
        nx=nx0*(2**n)
        xedges=np.linspace(xmin,xmax,nx+1)
        xnmids=0.5*(xedges[1:]+xedges[:-1])

        ny=ny0*(2**n)
        yedges=np.linspace(ymin,ymax,ny+1)
        ynmids=0.5*(yedges[1:]+yedges[:-1])

        xmids.append(xnmids)
        ymids.append(ynmids)

        subns=-np.ones((nx,ny))
        valns=-np.ones((nx,ny,nvals))

        subs.append(subns)
        vals.append(valns)

        if n==0:
            args=np.argwhere(subs[0]==-1) # slightly ugly way to select all cells on 0 level
            nsubs=len(args)
            for k in range(nsubs):
                if printncall==2:
                    print(ncall,' function calls so far')
                i=args[k,0]
                j=args[k,1]
                checksub(func,n,i,j,power=power,tol=tol)
        else:
            args=np.argwhere(subs[n-1]==1)
            nsubs=len(args)
            for k in range(nsubs):
                i=args[k,0]
                j=args[k,1]

                # optionally force neighbours of subdividied cells to divide to
                # divide to one level below
                if divneigh==1:
                    nxn=xmids[n-1].size
                    nyn=ymids[n-1].size
                    for d in range(4):
                        di=i+neigh[d,0]
                        dj=j+neigh[d,1]
                        # checking if cell in grid
                        if ~(((di>=0) & (di<=nxn-1)) | (xwrap==1)):
                            continue
                        if ~(((dj>=0)& (dj<=nyn-1)) | (ywrap==1)):
                            continue
                        di=di % nxn
                        dj=dj % nyn
                        if subs[n-1][di,dj]<1: # neighbour undivided
                            #print('cell ',i,', ',j,' (level ',n,')')
                            #print('has neighbour ',di,', ',dj,' undivided')
                            #print('subs before')
                            #print(subs)
                            fullforcesub(func,n-1,di,dj,divneigh=divneigh,xwrap=xwrap,ywrap=ywrap)
                            #print('subs after')
                            #print(subs)

                if printncall==2:
                    print(ncall,' function calls so far')
                for l in range(4):
                    li=2*i+local[l,0]
                    lj=2*j+local[l,1]
                    checksub(func,n,li,lj,power=power,tol=tol,divneigh=divneigh)
        if printncall==1:
            print(ncall,' function calls so far')
        n+=1

    return xmids,ymids,subs,vals


def oldchecklinear(vals,power=1,tol=0.25):
    sub=0
    mid=0.5*(vals[0]**power + vals[2]**power)
    diff=np.abs(vals[2]**power - vals[0]**power)
    if np.abs(vals[1]**power - mid)>tol*diff:
        sub+=1
    return sub

def checklinear(vals,tol=0.1,power=1):
    sub=0
    sort=np.sort(vals)
    diffs=sort[1:]-sort[:-1]
    if max(diffs)>tol+2*min(diffs):
        sub+=1
    return sub

def calcvals(func,i,j,n):
    global ncall
    vals[n][i,j,:]=func(xmids[n][i],ymids[n][j])
    ncall+=1

def forcesub(func,n,i,j): # force cell nmax,i,j to subdivide
    nlevels=len(subs)
    if (n==nlevels-1) | (subs[n][i,j]==1): # don't want to subdivide beyond top level or cells already subdivd
        return
    for l in range(4):
        li=2*i + local[l,0]
        lj=2*j + local[l,1]
        if subs[n+1][li,lj]==1:
            continue # already subdivided
        subs[n+1][li,lj]=0
        if vals[n+1][li,lj,0]==-1:
            calcvals(func,li,lj,n+1)

def fullforcesub(func,n,i,j,divneigh=-1,xwrap=-1,ywrap=-1): # cells to sudivide all the way up to n,i,j
    nnow=0
    while nnow<n:
        ndiff=(n-nnow)
        inow=int(i/(2**(ndiff)))
        jnow=int(j/(2**(ndiff)))
        if subs[nnow][inow,jnow]!=1:
            forcesub(func,nnow,inow,jnow)
            subs[nnow][inow,jnow]=1
            if divneigh==1:
                nxn=xmids[nnow].size
                nyn=ymids[nnow].size
                for d in range(4):
                    di=inow+neigh[d,0]
                    dj=jnow+neigh[d,1]
                    # checking if cell in grid
                    if ~(((di>=0) & (di<=nxn-1)) | (xwrap==1)):
                        continue
                    if ~(((dj>=0)& (dj<=nyn-1)) | (ywrap==1)):
                        continue
                    di=di % nxn
                    dj=dj % nyn
                    if subs[nnow][di,dj]==-1: # neighbour undivided
                        fullforcesub(func,nnow,di,dj,divneigh=divneigh,xwrap=xwrap,ywrap=ywrap)

        nnow+=1

def checksub(func,n,i,j,power=1,tol=0.25,xwrap=-1,ywrap=-1,divneigh=-1):
    nlevels=len(subs)
    nx=xmids[n].size
    ny=ymids[n].size
    nvals=vals[0][0,0,:].size
    sub=0
    if ((i>0) & (i<nx-1)) | (xwrap==1):
        dis=np.arange(i-1,i+2)%nx
        for di in dis:
            if vals[n][di,j,0]==-1:
                calcvals(func,di,j,n)
        for v in range(nvals):
            sub+=checklinear(vals[n][dis,j,v],power=power,tol=tol)
    if ((j>0) & (j<ny-1)) | (ywrap==1):
        djs=np.arange(j-1,j+2)%ny
        for dj in djs:
            if vals[n][i,dj,0]==-1:
                calcvals(func,i,dj,n)
        for v in range(nvals):
            sub+=checklinear(vals[n][i,djs,v],power=power,tol=tol)
    if sub>1:
        sub=1
    subs[n][i,j]=sub

def subscol(ax,xmids,ymids,subs,vals,whichval=0,c='k',lw=2,cmap='Spectral',vmin=-1,vmax=1,alpha=0,
            transfunc='None'):
    nlevels=len(subs)
    for n in range(nlevels):
        nxmids=xmids[n]
        nymids=ymids[n]
        nsubs=subs[n]
        nvals=vals[n]
        w=nxmids[1]-nxmids[0]
        h=nymids[1]-nymids[0]
        for i in range(nxmids.size):
            for j in range(nymids.size):
                if (nsubs[i,j]==0) | ((nsubs[i,j]==1) & (n==nlevels-1)):
                    val=nvals[i,j,whichval]
                    if transfunc!='None':
                        val=transfunc(val)
                    col=mpl.colormaps[cmap]((val-vmin)/(vmax-vmin))
                    edgecol=(0,0,0,alpha*(nlevels-n)/nlevels)
                    rect = mpl.patches.Rectangle((nxmids[i]-0.5*w, nymids[j]-0.5*h), w, h, linewidth=1,
                                                 facecolor=col, edgecolor=edgecol,zorder=nlevels-n)
                    ax.add_patch(rect)

def cInterval(xs,interval=0.95):
    xs=np.sort(xs)
    nData=xs.size
    width=int(nData*interval)
    ks=np.arange(0,int((1-interval)*nData))
    deltas=xs[ks+width]-xs[ks]
    K=np.argmin(deltas)
    return xs[K],xs[K+width]
