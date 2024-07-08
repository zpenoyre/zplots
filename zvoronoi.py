# plots based around voronoi grids - including binning data in voronoi cells
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.spatial
import scipy

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
    ax.set_axisbelow(False)
    ax.grid(False)
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
