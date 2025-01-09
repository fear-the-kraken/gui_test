#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for data handling and plotting tasks

@author: amandaschott
"""
import sys
import scipy
import numpy as np
import matplotlib.colors as mcolors
import colorsys
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, QtGui
import pdb

##################################################
########         GENERAL FUNCTIONS        ########
##################################################


def Downsample(arr, n):
    """ Downsample 2D input $arr by factor of $n """
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def Normalize(collection):
    """ Normalize data between 0 and 1 """
    Max=np.nanmax(collection)
    Min=min(collection)
    return np.array([(i-Min)/(Max-Min) for i in collection])

def Closest(num, collection):
    """ Return the $collection value closest to $num """
    return min(collection,key=lambda x:abs(x-num))

def IdxClosest(num, collection):
    """ Return index of the $collection value closest to $num """
    return list(collection).index(Closest(num,collection))

def Center(collection):
    """ Return central value in $collection """
    return collection[int(round(len(collection)/2))]

def IdxCenter(collection):
    """ Return central index in $collection """
    return int(round(len(collection)/2))

def Edges(collection):
    """ Return (first,last) values in collection """
    return (collection[0], collection[-1])

def MinMax(collection):
    """ Return (min,max) values in collection, excluding NaNs """
    return (np.nanmin(collection), np.nanmax(collection))

def Limit(collection, mode=2, pad=0.01, sign=1):
    """ Return lower and/or upper data limits of collection (+/- padding) """
    collection = np.array(collection)
    if collection.size > 0:
        min_max = MinMax(collection)
        vpad = np.ptp(collection[~np.isnan(collection)]) * pad * sign
        vmin, vmax = np.add(min_max, (-vpad, vpad))
    else:
        vmin, vmax = None,None
    if   mode==0 : return vmin
    elif mode==1 : return vmax
    elif mode==2 : return (vmin, vmax)


def SymLimit(collection, pad=0.0):
    """ Return (negative, positive) maximum absolute value in collection """
    abs_max = np.nanmax(np.abs(MinMax(collection)))
    return (-abs_max, abs_max)


def InRange(num, nmin, nmax):
    """ Return whether the value $num falls between (min, max) bounds """
    return (num >= nmin) and (num <= nmax)

def AllInRange(collection, nmin, nmax):
    """ Return whether all values in $collection fall between (min, max) bounds """
    return (min(collection) >= nmin) and (max(collection) <= nmax)
    
    
def CenterWin(collection, n, total=True):
    """ Return window of $n values surrounding central point in collection """
    ctr = IdxCenter(collection)
    N = int(round(n)) if total==True else int(round(n*2))
    nwin = int(n/2)
    ii = np.arange(ctr-nwin, ctr+nwin)
    if N % 2 > 0:
        ii = np.append(ii, ctr+nwin)
    return ii, collection[ii]

def get_sequences(idx, ibreak=1) :  
    """
    get_sequences(idx, ibreak=1)
    idx     -    np.vector of indices
    @RETURN:
    seq     -    list of np.vectors
    """
    diff = idx[1:] - idx[0:-1]
    breaks = np.nonzero(diff>ibreak)[0]
    breaks = np.append(breaks, len(idx)-1)
    
    seq = []    
    iold = 0
    for i in breaks:
        r = list(range(iold, i+1))
        seq.append(idx[r])
        iold = i+1
    return seq


##################################################
########         SIGNAL PROCESSING        ########
##################################################


def butter_bandpass(lowcut, highcut, lfp_fs, order=3):
    """ Return filter coefficients for given freq. cutoffs/sampling rate """
    nyq = 0.5 * lfp_fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, lfp_fs, order=3, axis=-1):
    """ Return bandpass-filtered data arrays """
    b, a = butter_bandpass(lowcut, highcut, lfp_fs, order=order)
    y = scipy.signal.filtfilt(b, a, data, axis=axis)
    return y


##################################################
########       MATPLOTLIB FUNCTIONS       ########
##################################################


def Cmap(data, cmap=plt.cm.coolwarm, norm_data=None, alpha=1.0, use_alpha=False):
    """ Return RGBA array (N x 4) of colors mapped from $data values """
    if norm_data is None:
        norm_data = data
    try:
        normal = plt.cm.colors.Normalize(np.nanmin(norm_data), np.nanmax(norm_data))
        arr = cmap(normal(data))
        arr[:, 3] = alpha
        if use_alpha: return arr
        return arr[:, 0:3]
    except:
        if use_alpha: return np.ones((len(data), 4))
        return np.ones((len(data), 4))

    
def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def color2rgba(c):
    """ Convert input color to 4-element RGBA array, scaled 0-1 """
    import re
                
    def rgb_str(c):
        fmt = '%s' + re.escape("(") + '(.*)' + re.escape(")")  # "rgb(...)" format
        fx  = lambda r: [*map(float, r[0].replace(' ','').split(','))]
        fx2 = lambda k: fx(re.findall(fmt % k, c))
        try:  # convert string (e.g. "255, 0, 150") to numerical values
            try     : res = fx2('rgb')
            except  : res = fx2('rgba')
            return np.array(res)
        except:
            return None
    # convert input to RGBA tuple
    if isinstance(c, str):
        try    : C = np.array(mcolors.to_rgba(c))  # color name/hex string --> rgba
        except : C = np.array(rgb_str(c))          # QtStyleSheets string  --> rgb (or None)
    elif isinstance(c, QtCore.Qt.GlobalColor):  # global Qt color --> rgba
        C = np.array(QtGui.QColor(c).getRgb())
    # RGB(A) values --> check for non-numerical elements, get rid of extra dimensions
    elif type(c) in [list,tuple,np.ndarray]:
        try    : C = np.array(c, dtype='float').reshape(-1)
        except : C = None
    else: C = None
    
    try:  # validate RGB tuple, convert to array for scaling
        assert (len(C) in [3,4]) and (AllInRange(C, 0, 255))
        RGBA = np.array(C, dtype='float32')
    except: 
        return None
    if any(RGBA > 1)  : RGBA = np.array(RGBA / 255.)  # scale 0-1
    if len(RGBA) == 3 : RGBA = np.append(RGBA, 1.0)   # add missing alpha value
    return RGBA

def rgb1(c, alpha=False, dtype=tuple):
    """ Return RGB(A) tuple (0-1) """
    rgba_1 = color2rgba(c)
    if alpha: return dtype(rgba_1)
    return dtype(rgba_1[0:3])

def rgb255(c, alpha=False, dtype=tuple):
    """ Return RGB(A) tuple (0-255) """
    rgba_1 = rgb1(c, alpha, dtype=np.array)
    rgb_255 = np.array(rgba_1 * 255).astype('int32')
    return dtype(rgb_255)

def qstyle_rgb(c, alpha=False):
    """ Return RGB(A) string for QStyleSheets """
    rgb_255 = (rgb255(c, alpha, dtype=tuple))
    if alpha: return 'rgba' + str(rgb_255)
    return 'rgb' + str(rgb_255)

def get_hex(c):
    return mcolors.rgb2hex(rgb1(c))#, keep_alpha=True)

def alpha_like(c, alpha=0.5, bg='white', dtype=tuple):
    if alpha > 1: alpha /= 255.
    rgba = rgb1(c, alpha=True, dtype=list)
    # match input RGBA if given, otherwise use $alpha param
    fx = lambda a: a if a<1 else alpha
    rgb,a = [fx(rgba.pop(-1)), np.array(rgba)][::-1]
    # interpolate between color and background to approximate alpha
    bg = rgb1(bg, dtype=np.array)
    new_rgb1 = (1.-a) * bg + a*rgb
    return dtype(new_rgb1)

def Cmap_alpha(cmap, alpha):
    return np.c_[cmap[:, 0:3], np.ones(len(cmap))*alpha]
    
def Cmap_alpha_like(cmap, alpha):
    converted_c = [*map(lambda c: alpha_like(c, alpha, dtype=np.array), cmap)]
    return np.array(converted_c)


def hue(c, percent, mode=1, cscale=255, alpha=1, res='tuple'):
    """ Adjust input color tint (mode=1), shade (0), or saturation (0.5) """
    rgb = rgb1(c, alpha=False, dtype=np.array)
    #rgb = get_rgb(c, cscale=1)[0:3]
    if mode == 1     : tg = np.array([1., 1., 1.])  # lighten color
    elif mode == 0   : tg = np.array([0., 0., 0.])  # darken color
    elif mode == 0.5 : tg = rgb.mean()              # de-intensify color
    distance = tg - rgb
    step = distance * percent
    adj_c = np.array(list(rgb + step) + [alpha])
        
    if cscale == 255:
        adj_c = np.round(adj_c * 255).astype('int')
    if res == 'tuple' : return tuple(adj_c.tolist())
    elif res == 'hex' : return mcolors.to_hex(adj_c, keep_alpha=True)
    else              : return adj_c
    
    
def rand_hex(n=1, bright=True):
    """ Return $n random colors in hex code format """
    hue = np.random.uniform(size=n)
    if bright:
        l_lo, l_hi, s_lo, s_hi = [0.4, 0.6, 0.7, 1.0]
    else:
        l_lo, l_hi, s_lo, s_hi = [0.0, 1.0, 0.0, 1.0]
    lightness = np.random.uniform(low=l_lo, high=l_hi, size=n)
    saturation = np.random.uniform(low=s_lo, high=s_hi, size=n)
    
    hex_list = []
    for h,l,s in zip(hue,lightness,saturation):
        hexc = mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))
        hex_list.append(hexc)
    if n==1:
        hex_list = hex_list[0]
    return hex_list
    
    
def match_ylimits(*axs, set_ylim=True):
    """ Standardize y-axis limits across plots """
    ylim = MinMax(np.concatenate([ax.get_ylim() for ax in axs]))
    if set_ylim:
        for ax in axs:
            ax.set_ylim(ylim)
    return ylim


def add_legend_items(leg, new_items):
    """ Add new Matplotlib item(s) to existing legend """
    try    : new_items.__iter__()
    except : new_items = [new_items]
    handles = leg.legend_handles + list(new_items)
    labels = [handle.get_label() for handle in handles]
    title = leg.get_title().get_text()
    leg._init_legend_box(handles, labels)
    leg.set_title(title)
    return leg


##################################################
########           PYQT WIDGETS           ########
##################################################


def qapp():
    """ Check for an existing QApplication instance """
    app = QtWidgets.QApplication.instance()
    if not app: 
        print('qapp creating a new QApplication instance')
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setQuitOnLastWindowClosed(True)
    return app

def layout_items(qlayout):
    """ Return all widgets in a given QBoxLayout """
    items = [qlayout.itemAt(i).widget() for i in range(qlayout.count())]
    return items

def stealthy(widget, val):
    """ Lazy method for updating widgets without triggering signals  """
    widget.blockSignals(True)
    try: widget.setValue(val)                       # spinbox value
    except:
        try: widget.setRange(*val)                  # spinbox range
        except:
            try: widget.setCurrentText(val)         # dropdown item
            except:
                try: widget.setPlainText(val)       # text edit content
                except:
                    try: widget.setText(val)        # line edit content
                    except:
                        try: widget.setChecked(val) # button check
                        except:
                            pdb.set_trace()
                            print(f'Could not set value for widget {widget}')
    widget.blockSignals(False)

class DividerLine(QtWidgets.QFrame):
    """ Basic horizontal (or vertical) separator line """
    def __init__(self, orientation='h', lw=3, mlw=3, parent=None):
        super().__init__(parent)
        if orientation == 'h':
            self.setFrameShape(QtWidgets.QFrame.HLine)
        else:
            self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setLineWidth(lw)
        self.setMidLineWidth(mlw)
        
        
def InterWidgets(parent, orientation='v'):
    """ Return layout with an intermediate widget between $parent and children """
    # parent -> interlayout > interwidget > layout
    interlayout = QtWidgets.QVBoxLayout(parent)
    interlayout.setContentsMargins(0,0,0,0)
    interwidget = QtWidgets.QWidget()
    interwidget.setContentsMargins(0,0,0,0)
    if orientation   == 'v' : layout = QtWidgets.QVBoxLayout(interwidget)
    elif orientation == 'h' : layout = QtWidgets.QHBoxLayout(interwidget)
    else                    : layout = QtWidgets.QGridLayout(interwidget)
    interlayout.addWidget(interwidget)
    return interlayout, interwidget, layout
        

def ScreenRect_V1(perc_width=1, perc_height=1, keep_aspect=True):
    """ Return QRect box centered and scaled relative to screen geometry """
    screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
    if keep_aspect:
        perc_width, perc_height = [min([perc_width, perc_height])] * 2
    app_width = int(screen_rect.width() * perc_width)
    app_height = int(screen_rect.height() * perc_height)
    app_x = int((screen_rect.width() - app_width) / 2)
    app_y = int((screen_rect.height() - app_height) / 2)
    qrect = QtCore.QRect(app_x, app_y, app_width, app_height)
    return qrect

def ScreenRect(perc_width=1, perc_height=1, keep_aspect=True):
    qapp()
    screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
    if keep_aspect:
        perc_width, perc_height = [min([perc_width, perc_height])] * 2
    app_width = int(screen_rect.width() * perc_width)
    app_height = int(screen_rect.height() * perc_height)
    app_x = int((screen_rect.width() - app_width) / 2)
    app_y = int((screen_rect.height() - app_height) / 2)
    qrect = QtCore.QRect(app_x, app_y, app_width, app_height)
    return qrect


def get_ddir():
    qapp()
    res = QtWidgets.QFileDialog().getExistingDirectory(None, '', '', 
                                                       QtWidgets.QFileDialog.DontUseNativeDialog)
    return res
    