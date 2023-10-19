#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:56:14 2020

Author: Zhichao Li at UCSD ERL
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mp
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt

# remove type3 fonts in figure
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#===========================================================
# change legend attributes
# ax11.legend(loc='center left', bbox_to_anchor=(1, 0.7), prop={'size':14})
# ax12.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':14})


#===========================================================

def my_default_plt_setting():
    """ Set customized default setting.
    """
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['legend.fontsize'] = 'large'
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.titlepad'] = 10
    # mpl.rcParams['axes.labelpad'] = 3
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams["figure.constrained_layout.w_pad"] = 0.2
    plt.rcParams["figure.constrained_layout.h_pad"] = 0.1
    plt.rcParams["figure.constrained_layout.wspace"] = 0.05
    plt.rcParams["figure.constrained_layout.hspace"] = 0.04
    return 0


def save_fig_to_folder(fig, folder, fname, dpi=300, ftype_ext='.png'):
    """ 
    Save figure to specified location (create folder if it does not exist)
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    figname_full = os.path.join(folder, fname + ftype_ext)
    fig.savefig(figname_full, dpi=dpi, bbox_inches='tight')
    return 0


def debug_print(n, msg):
    """
    Debug printing for showing different level of debug information.
    """
    if n >= 0:
        tab_list = ['  '] * n
        dmsg = ''.join(tab_list) + 'DEBUG ' + msg
        print(dmsg)
    else:
        pass


def set_canvas_box(ax, bd_pts):
    """ 
    Set canvas of ax using map boundary pts.
    """
    xl, xh, yl, yh = bd_pts
    ax.set_xlim([xl, xh])
    ax.set_ylim([yl, yh])
    # ax.grid()
    ax.set_aspect('equal')
    return ax


def create_circle_patch(circle_array, color='tab:grey', alpha=0.8):
    """
    Given a list of same objects: (circles, ploygons, ...), create a
    patch collection object with certain color and trasparent alpha.
    Input: circle_array (num_pt * 3)
           each item [x, y, radius]
    """
    clist = []
    # prevent array size degeneration always turn it to 2D
    circle_array = np.reshape(circle_array, (-1, circle_array.shape[-1]))
    for item in circle_array:
        x, y, r = item
        circle = mp.Circle((x, y), r)
        clist.append(circle)
    patch_circles = PatchCollection(clist, color=color, alpha=alpha)
    return patch_circles


def create_arrowhead_patch():

    verts = [(0,0), (-1,-1), (2,0), (-1,1), (0,0)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, 
                Path.LINETO, Path.CLOSEPOLY]
    return Path(verts, codes)