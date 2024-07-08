# utilities for generating and manipulating colours and cmaps
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

pride_progress_cs=['#FFFFFF','#FFAFC8','#74D7EE','#613915','#000000','#E40303','#FF8C00','#FFED00','#008026','#24408E','#732982']
pride_progress_altcs=['#000000','#613915','#E40303','#FF8C00','#FFED00','#008026','#24408E','#732982','#74D7EE','#FFAFC8','#FFFFFF']

def makeCmap(hexColour,cVal=[0,1]):
    if type(hexColour) is str:
        #just one colour provided so makes map from white to colour
        hexColour=['#FFFFFF',hexColour]
        cVal=[0,1]
    if len(cVal)!=len(hexColour):
        cVal=np.linspace(0,1,len(hexColour))
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

def colour_grid(colours,x,y): #returns a colour from a position (x,y : 0-1) on a grid of 4 (colours : hex strings)
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
