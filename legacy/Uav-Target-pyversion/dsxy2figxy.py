import numpy as np
import matplotlib.pyplot as plt
    
def dsxy2figxy(varargin = None): 
    if len(varargin[0]) == 1 and ishandle(varargin[0]) and str(get(varargin[0],'type')) == str('axes'):
        hAx = varargin[0]
        varargin = varargin(np.arange(2,end()+1))
    else:
        hAx = gca
    
    if len(varargin) == 1:
        pos = varargin[0]
    else:
        x,y = deal(varargin[:])
    
    axun = get(hAx,'Units')
    set(hAx,'Units','normalized')
    axpos = get(hAx,'Position')
    axlim = plt.axis(hAx)
    axwidth = np.diff(axlim(np.arange(1,2+1)))
    axheight = np.diff(axlim(np.arange(3,4+1)))
    if ('x' is not None):
        varargout[0] = (x - axlim(1)) * axpos(3) / axwidth + axpos(1)
        varargout[2] = (y - axlim(3)) * axpos(4) / axheight + axpos(2)
    else:
        pos[1] = (pos(1) - axlim(1)) / axwidth * axpos(3) + axpos(1)
        pos[2] = (pos(2) - axlim(3)) / axheight * axpos(4) + axpos(2)
        pos[3] = pos(3) * axpos(3) / axwidth
        pos[4] = pos(4) * axpos(4) / axheight
        varargout[0] = pos
    
    set(hAx,'Units',axun)
    return varargout