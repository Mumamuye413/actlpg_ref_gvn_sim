#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:50:23 2020

Author: Zhichao Li at UCSD ERL
"""
import numpy as np
import numpy.linalg as nLA
import matplotlib as mpl
from matplotlib import pyplot as plt
from utils.tools_plot_common import (add_arrow, 
                               create_arrow_patch,
                               create_arrowhead_patch,
                               create_circle_patch, set_peripheral)

#=======================================================================
# Constants
#=======================================================================

#%% font size constants
TITLE_FS = 18
XLABEL_FS = 14

#%% line-width
LW = 2


#%% line types

LT3_TYPE1 = ['b-', 'r-', 'g-']
LT3_TYPE2 = ['b-.', 'r-.', 'g-.']

LT4_TYPE1 = ['b', 'b-.', 'r', 'r-.']

#%% lables
LB3_TYPE1 = [r'$x_1$', r'$x_2$', r'$x_3$']
LB3_TYPE2 = [r'$y_1$', r'$y_2$', r'$y_3$']

LB4_TYPE1 =  [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$']
LB4_TYPE2 =  [r'$x$', r'$y$', r'$\dot{x}$', r'$\dot{y}$']

#%% colors

C4_TYPE1 = ['b', 'b', 'r', 'r']
#=======================================================================
# Plot Trajectories
#=======================================================================

def add_traj(fig, ax,
            tvec, xvec_hist, *,
            x_str='Time (seconds)',
            line_types=LT3_TYPE1,
            labels=LT3_TYPE1,
            lw=2):
    """ Add trajectory of dynamical system.
        xvec_hist (num_pt * num_states)
    """
    for ii in range(xvec_hist.shape[1]):
        ax.plot(tvec, xvec_hist[:, ii],
                line_types[ii], label=labels[ii], lw=lw)
    return fig, ax


def plot_traj_3rd_2D(tvec, xvec_hist, yvec_hist, *,
                     x_str='Time (seconds)',
                     line_types1=LT3_TYPE1,
                     line_types2=LT3_TYPE2,
                     labels1=LB3_TYPE1,
                     labels2=LB3_TYPE2):
    """ Plot trajectory of third-order system.
        xvec_hist (num_pt * num_states)
    """
    # apply plot_traj_3rd_1D twice
    fig, ax = plt.subplots()
    fig, ax = add_traj(fig, ax, tvec, xvec_hist, x_str=x_str,
                     line_types=line_types1,labels=labels1)

    fig, ax = add_traj(fig, ax, tvec, yvec_hist, x_str=x_str,
                     line_types=line_types2,labels=labels2)

    ax.legend(loc='best', prop={'size':10})
    ax.set_xlabel(x_str, fontsize=XLABEL_FS)
    # ax.grid()
    plt.show()

    return fig, ax


def plot_traj_dIntSys_2D(fig, ax,
                         tvec, xvec_hist, *,
                         x_str='Time (seconds)',
                         t_str='System State Trajectory',
                         line_types=LT4_TYPE1,
                         labels=LB4_TYPE1):

    """ plot states of 2D double integrator system
    """

    for ii in range(xvec_hist.shape[1]):
        ax.plot(tvec, xvec_hist[:, ii], line_types[ii],
                label= LB4_TYPE1[ii], color= C4_TYPE1[ii], lw=2)

    ax.set_title(t_str, fontsize=15)
    ax.set_xlabel(x_str, fontsize=15)
    ax.legend()
    # ax.grid()
    plt.show()
    return fig, ax


def plot_polar_state(fig, ax,
                    tvec, xvec_hist, 
                    labels, controller, *,
                    eps = 0,
                    x_str='Time (seconds)',
                    line_types=['b', 'g', 'r'],
                    angle_unit='rad',
                    e_scale_factor=1):
    """ 
    Plot unicycle states in polar coordinates
    """
    print('angle_unit = %s' % angle_unit)
    xvec_hist_copy = xvec_hist.copy()

    if angle_unit == 'deg':
        xvec_hist_copy[:, 1:] = np.rad2deg(xvec_hist_copy[:, 1:])
        xvec_hist_copy[:, 0] = e_scale_factor * xvec_hist_copy[:, 0]
        eps[0] = e_scale_factor*eps[0]
        ylim=[-90, 90]
        ax.set_ylim(ylim)

    fig, ax = add_traj(fig, ax, tvec, xvec_hist_copy, x_str=x_str,
                        line_types=line_types,labels=labels)
    ax.axhspan(0,eps[0], facecolor="#7BC8F6",alpha=0.8, label=r"$\epsilon_e$")
    ax.axhspan(-eps[1],eps[1], facecolor="#15B01A",alpha=0.3, label=r"$\epsilon_{\theta}$")
    ax.set_xlabel(x_str, fontsize=15)

    t_str='Error Dynamic State Trajectory\nwith '+controller+' Controller'
    ax.set_title(t_str, fontsize=15)

    ax.legend()
    ax.grid(visible=True)
    return fig, ax


def plot_sampled_traj_dIntSys(fig, ax, zmat, *, ostar=[], goal_pt=[], sample_n=8):
    """ plot states in cartesian coordinates for dIntSys
    zmat:
        location matrix (num_pt, 2)
    ostar: single point obstacle (2,)
    goal_pt: single point obstacle (2,)
    sample_n: sample point along trajectory
    """

    # add traj
    ax.plot(zmat[:, 0], zmat[:, 1], 'k', lw=2)
    ax = add_arrow(ax, zmat[0])
    # add obstacle
    if len(ostar)  > 0 :
        ax.plot(ostar[0], ostar[1], 'rp', ms=40)
        if nLA.norm(ostar, np.inf) < 2:
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
    # add goal pt
    if len(goal_pt) > 0:
        ax.plot(goal_pt[0], goal_pt[1], 'r*', ms=20)

    ax.set_title('Sampled Robot Position Trajectory', fontsize=15)
    ax.set_xlabel(r"$X_W$", fontsize=15)
    ax.set_ylabel(r"$Y_W$", fontsize=15)

    sample_idx = np.linspace(0, zmat.shape[0]-1, sample_n).astype(int)
    # just display first 5
    rbt_loc_arr = zmat[sample_idx, :2]
    N_sample = len(rbt_loc_arr)
    rbt_radius_arr = np.ones((N_sample,1)) * 0.05
    rbt_circle_arr = np.hstack((rbt_loc_arr, rbt_radius_arr))
    rbt_patch_circles = create_circle_patch(rbt_circle_arr,color='g')
    ax.add_collection(rbt_patch_circles)
    # ax.grid()
    ax.set_aspect('equal')
    plt.show()
    return fig, ax


def plot_sampled_traj_unicycle(figure, zmat, ctrl_type, *,
                               ostar=[], goal_pt=[], sample_n=8):
    """ plot states in cartesian coordinates for unicycle
    zmat:
        state matrix (num_pt, 3) in SE2 (z1, z2, z_phi)
    ostar: single point obstacle (2,)
    goal_pt: single point obstacle (3,)
    sample_n: sample point along trajectory
    """

    fig, ax = figure
    ax.plot(zmat[:, 0], zmat[:, 1], 'k', lw=2)

    # add obstacle
    if len(ostar)  > 0 :
        ax.plot(ostar[0], ostar[1], 'rp', ms=20)
        if nLA.norm(ostar, np.inf) < 2:
            ax.set_xlim([-2.1, 2.1])
            ax.set_ylim([-2.1, 2.1])
    # add goal pt
    if len(goal_pt) > 0:
        gmk = create_arrowhead_patch()
        gmk = gmk.transformed(mpl.transforms.Affine2D().rotate_deg(np.rad2deg(goal_pt[2])))
        ax.plot(goal_pt[0], goal_pt[1], 'g', marker=gmk, ms=12)

    title_str = 'Sampled Robot Position Trajectory \nwith '+ ctrl_type +' controller'
    ax.set_title(title_str, fontsize=15)
    ax.set_xlabel(r"$X_W$", fontsize=15)
    ax.set_ylabel(r"$Y_W$", fontsize=15)

    sample_idx = np.linspace(0, zmat.shape[0]-1, sample_n).astype(int)
    rbt_loc_arr = zmat[sample_idx, :2]
    N_sample = len(rbt_loc_arr)
    rbt_radius_arr = np.ones((N_sample,1)) * 0.04
    rbt_circle_arr = np.hstack((rbt_loc_arr, rbt_radius_arr))
    rbt_patch_circles = create_circle_patch(rbt_circle_arr,color='g')
    ax.add_collection(rbt_patch_circles)
    

    # add orientation arrows  (arrow_item stx, sty, dx, dy)
    arrows_array = np.zeros((N_sample, 4))
    arrows_array[:, 0:2] = zmat[sample_idx, :2]
    arrows_array[:, 2] = np.cos(zmat[sample_idx, 2])
    arrows_array[:, 3] = np.sin(zmat[sample_idx, 2])
    arrows_patches = create_arrow_patch(arrows_array, sf=0.1, width=0.05)
    ax.add_collection(arrows_patches)
    ax.set_aspect('equal')
    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-2.1, 2.1])
    ax.grid(visible=True)
    # plt.show()
    return [fig, ax]

def plot_init_traj_unicycle(figure, zmat, ctrl_type, *,
                               ostar=[], goal_pt=[], sample_n=8):
    """ plot states in cartesian coordinates for unicycle
    zmat:
        state matrix (num_pt, 3) in SE2 (z1, z2, z_phi)
    ostar: single point obstacle (2,)
    goal_pt: single point obstacle (3,)
    sample_n: sample point along trajectory
    """

    fig, ax = figure
    ax.plot(zmat[:, 0], zmat[:, 1], 'k', lw=2)

    # add obstacle
    if len(ostar)  > 0 :
        ax.plot(ostar[0], ostar[1], 'rp', ms=20)
        if nLA.norm(ostar, np.inf) < 2:
            ax.set_xlim([-2.1, 2.1])
            ax.set_ylim([-2.1, 2.1])
    # add goal pt
    if len(goal_pt) > 0:
        gmk = create_arrowhead_patch()
        gmk = gmk.transformed(mpl.transforms.Affine2D().rotate_deg(np.rad2deg(goal_pt[2])))
        ax.plot(goal_pt[0], goal_pt[1], 'g', marker=gmk, ms=25, alpha=0.5)

    # add pose heading along trajectory
    for i in range(400):
        if i%20==0 and i<(len(zmat)/3):
            crt_pt = zmat[i,:]
            tmk = create_arrowhead_patch()
            tmk = tmk.transformed(mpl.transforms.Affine2D().rotate_deg(np.rad2deg(crt_pt[2])))
            ax.plot(crt_pt[0], crt_pt[1], color='brown', marker=tmk, ms=10)
    # add initial pt
    z0 = zmat[0,:]
    smk = create_arrowhead_patch()
    smk = smk.transformed(mpl.transforms.Affine2D().rotate_deg(np.rad2deg(z0[2])))
    ax.plot(z0[0], z0[1], 'r', marker=smk, ms=15)

    # title_str = 'Sampled Robot Position Trajectory \nwith '+ ctrl_type +' controller'
    # ax.set_title(title_str, fontsize=15)
    ax.set_xlabel(r"$X_W$", fontsize=15)
    ax.set_ylabel(r"$Y_W$", fontsize=15)
    
    ax.set_aspect('equal')
    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-2.1, 2.1])
    ax.grid(visible=True)
    # plt.show()
    return [fig, ax]
#=======================================================================
# Plot Control Signals
#=======================================================================

def plot_ctrl_dIntSys_2D(fig, ax,
                         tvec, uvec_hist, *,
                         x_str='Time (seconds)',
                             t_str='System Control Trajectory',
                         line_types=None,
                         labels=None,
                         lw=2):
    """ plot control signals of 2D double integrator system
    """
    for ii in range(uvec_hist.shape[1]):
        ax.plot(tvec, uvec_hist[:, ii],
                line_types[ii], label=labels[ii], lw=lw)

    ax = set_peripheral(ax, t_str=t_str, x_str=x_str)
    return fig, ax

def plot_ctrl_2channel_bds(fig, ax1, ax2, tvec, umat, ctrl_bds, t_str):
    """ plot 2 channel control signals with box bounds
    """
    
    # ones_mat = np.ones(umat.shape[0])

    ax1.plot(tvec, umat[:, 0], 'b', label= r'$v$', lw=2)
    ax2.plot(tvec, umat[:, 1], 'b', label= r'${\omega}$', lw=2)

    if len(ctrl_bds) > 0:
        u1_min, u1_max, u2_min, u2_max = ctrl_bds
        ax1.axhspan(u1_min, u1_max, facecolor='0.5', alpha=0.5, label=r'$v_{range}', lw=3)
        ax2.axhspan(u2_min, u2_max, facecolor='0.5', alpha=0.5, label=r'${\omega}_{range}', lw=3)

    ax1.set_title(t_str, fontsize=15)
    ax2.set_xlabel("Time (Seconds)", fontsize=15)
    ax1.legend(prop={'size':12}, loc='upper right')
    ax2.legend(prop={'size':12}, loc='upper right')
    ax1.grid(visible=True)
    ax2.grid(visible=True)

    return fig, ax1, ax2

#=======================================================================
# Plot CBF Status
#=======================================================================

def plot_CBF_status(fig, ax1, ax2, tvec, hb_vec, hb_dot_vec, c4):
    """ plot status of CBF barrier functions h(t) and hdot(t)
    """
    # fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    label_str1 = str(-c4) + r'$b$'

    ax1.plot(tvec, hb_vec, label= r'$b$', lw=2)
    ax1.plot(tvec, 0 * hb_vec, 'k--', label= '0',   lw=3)

    ax2.plot(tvec[1:], hb_dot_vec,  label= r'$\dot{b}$',   lw=2)
    ax2.plot(tvec, -c4 * hb_vec, 'k--', label= label_str1,   lw=3)

    ax1.set_title('Control Barrier Function Status', fontsize=18)
    ax1.legend(loc='upper right', prop={'size':12}) # change legend box loc
    ax2.legend(loc='upper right', prop={'size':12})
    ax2.set_xlabel("Time (Seconds)", fontsize=15)
    # ax1.grid()
    # ax2.grid()
    plt.show()
    return fig, ax1, ax2


#=======================================================================
# Plot CLF Status
#=======================================================================

def plot_CLF_status(fig, ax1, ax2, tvec, Vvec, Vdot_vec, c3):
    """ plot status of Control Lyapunov Functions V(t) and V_dot(t)
    """
    ax1.plot(tvec, Vvec, label= r'$V$', lw=2)
    ax2.plot(tvec[1:], Vdot_vec, 'b',  label= r'$\dot{V}$', lw=2)
    ax2.plot(tvec, -c3 * Vvec, 'k--',  label= str(-c3) + r'$V$',   lw=3)

    ax1.set_title('Control Lyapunov Function Status', fontsize=18)
    ax1.legend(loc='best', prop={'size':12})
    ax2.legend(loc='best', prop={'size':12})
    ax2.set_xlabel("Time (Seconds)", fontsize=15)
    # ax1.grid()
    # ax2.grid()
    plt.show()
    return fig, ax1, ax2
