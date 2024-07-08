# plots based around subdiving 2D (square) grid based on local feautures of interest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# colours subgrids based on vals
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
