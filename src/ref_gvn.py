#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Governor System Class.

Modified from:
    https://github.com/zhl355/ICRA2020_RG_SDDM
    Author: zhichao li @ UCSD-ERL
    Date: 06/26/2020
    BSD 3-Clause License
"""
# python built in package
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
import matplotlib.patches as patches
from utils.tools_geometry import ball_path_intersection, dist_point2obs
from utils.tools_plot_common import create_arrowhead_patch



class GovError(Exception):
    """ User Defined Exceptions for RG.
    """

    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "GovError exception: {0}".format(self.msg)
        else:
            return "GovError exception"

class RbtGovSys:
    """
    A class for robot-governor system (RGS) update in simulation.
    """

    def __init__(self, z_start, z_goal, param_dic, env_map, path, ax, dt=0.1):
        """
        INPUT
        x0              robot initial state [zx, zy, wx, wy]
        param_dic       control gains and other parameters
        env_map         lists of obstacle features
        """

        self.z = z_start[0:2]
        self.g = z_start
        self.v = np.zeros(2)
        self.zg = z_goal[0:2]

        self.map = env_map
        self.path = path
        self.kg = param_dic['kg']
        self.Lf = param_dic['Lf']
        self.eps = param_dic['eps']
        self.eta = param_dic['eta']

        self.dt = dt
        self.ax = ax

        self.radius_LS = 0.1
        self.d_gz = 0.1
        self.g_bar = z_start[0:2]
        self.g_hat = z_start[0:2]
        self.target_ind = 0

        self.g_log = np.array([self.g])
        self.ug_log = np.empty((1,2))
        self.deltaE_log = np.empty(1)

    def get_min_dist2obs(self, r_robot=0.33):
        """
        compute minimium distance from governor/robot position to all static obstacles
        """
        map_bounds = np.array(self.map.obs_boundary)
        map_obs = np.array(self.map.obs_rectangle)
        if len(map_obs)>0:
            obs_all = np.vstack((map_bounds, map_obs))
        else:
            obs_all = map_bounds
        z_min_dist, g_min_dist = np.inf, np.inf
        zp, gp = self.z[0:2], self.g[0:2]

        for ob in obs_all:
            z_dist_current = dist_point2obs(zp, ob)
            z_min_dist = min(z_min_dist, z_dist_current)
            g_dist_current = dist_point2obs(gp, ob)
            g_min_dist = min(g_min_dist, g_dist_current)
        
        self.d_zO = z_min_dist
        self.d_gO = g_min_dist
        self.d_gz = np.linalg.norm(gp-zp) + r_robot

        return z_min_dist, g_min_dist

    def get_LS_ball(self):
        """
        compute Local Safe zone ball raduis in the sense of static obstacles 
        """
        eta = self.eta
        zp, gp = self.z[0:2], self.g[0:2]
        v = self.v
        delta_t = np.linalg.norm(zp-gp)**2+0.5*np.linalg.norm(v)**2
        _, d_gO = self.get_min_dist2obs()
        
        deltaE = d_gO**2 - delta_t
        # radius_LS = d_gO - delta_t + eta
        radius_LS = max(0,np.sqrt(deltaE))

        self.deltaE_log = np.hstack((self.deltaE_log, deltaE))
        self.radius_LS = radius_LS
        return radius_LS
    
    def search_target(self):
        """
        search closest way point within looking ahead distance along path
        """
        within_Lf = True
        gp = self.g[0:2]
        current_ind = np.argmin(np.linalg.norm(self.path-gp, axis=1))
        next_ind = current_ind+1
        current_target = self.path[current_ind]

        while within_Lf:

            if (current_ind + 1) >= len(self.path):
                break  # not exceed goal

            next_target = self.path[next_ind]
            dist2next = np.linalg.norm(next_target-current_target)

            if dist2next >= self.Lf:
                within_Lf = False
                break
            else:
                current_ind += 1
        return current_ind
    
    def get_gvn_goal(self):
        """
        compute governor local projected goal g_bar
        """
        radius_LS = self.get_LS_ball()
        path = self.path
        gp, zgp = self.g[0:2], self.zg[0:2]
        g_error = np.linalg.norm(gp-zgp)
        self.line_seg_ends = []

        if radius_LS < self.eps:
            g_bar = np.copy(gp)
        else:
            goal_exist, g_bar, waypoint_ind = ball_path_intersection(gp, radius_LS, path)
            if goal_exist:
                self.target_ind = waypoint_ind
            else:
                # check if governor is approaching global goal
                if g_error<=radius_LS:
                    g_bar = np.copy(zgp)
                else:
                    """ 
                    if no intersection, find a line segment between governor position and target point
                    on the path as a subpath, get intersection of ball and the subpath line segment
                    """
                    target_ind = self.search_target()
                    target_point = path[target_ind]
                    path_seg = [gp, target_point]
                    _, g_bar, _ = ball_path_intersection(gp, radius_LS, path_seg)
                    self.target_ind = target_ind

                    # plot line segment
                    self.line_seg_ends = [[gp[0], target_point[0]], [gp[1], target_point[1]]]
                    
        self.g_bar = g_bar
        return g_bar
    
    def get_governor_control(self, g_bar=[]):
        """
        compute control input ug = -kg(g-g_bar)
        """
        if len(g_bar)==0:
            g_bar = self.get_gvn_goal()
        gp = self.g[0:2]
        ug = self.kg*(g_bar - gp)
        self.ug = ug
        self.ug_log = np.vstack((self.ug_log, ug))
        return ug

    def update_governor(self, x, ug):
        """
        update governor state g and robot position z
        INPUT
        x       4d robot state [zx, zy, wx, wy]
        """
        gp = self.g[0:2]
        gp_new = gp + ug*self.dt
        g_hat = gp + ug/self.kg

        self.g[0:2] = gp_new
        g_new = self.g
        self.g_log = np.vstack((self.g_log, np.array([g_new])))

        self.g_hat = g_hat
        self.z = x[0:2]
        self.v = x[2:]
        return g_new

    def plotting_governor_tracking_init(self, ctrl_type=None):
        """
        initial governor tracking trojectory plot
        """
        self.ctrl_type = ctrl_type
        # plot local safe zone
        self.circle_ls = patches.Circle(
                        (self.g[0], self.g[1]), self.radius_LS,
                        edgecolor='goldenrod',
                        facecolor='khaki',
                        alpha=0.4,
                        fill=True)
        self.patch_ls = self.ax.add_patch(self.circle_ls)
        # plot robot prediction set
        self.circle_rs = patches.Circle(
                        (self.g[0], self.g[1]), self.d_gz,
                        edgecolor='crimson',
                        facecolor='lightpink',
                        alpha=0.4,
                        fill=True)
        self.patch_rs = self.ax.add_patch(self.circle_rs)
        # plot governor position
        if self.ctrl_type=='Polar':
            self.awhead = create_arrowhead_patch()
            # show governor location and heading
            awhead_rot = self.awhead.transformed(mpl.transforms.Affine2D().rotate(self.g[2]))
            self.loc_gvn, = self.ax.plot(self.g[0], self.g[1], marker=awhead_rot, ms=15, 
                                color='cornflowerblue', label='governor position')
        else:
            self.loc_gvn, = self.ax.plot(self.g[0], self.g[1], 'o', 
                                     color='cornflowerblue', markersize=10, label='governor position')
        # plot governor trajectory
        self.trj_gvn, = self.ax.plot(self.g_log[:,0], self.g_log[:,1], 
                                     lw=1, color='orange', label='governor trajectory')
        # plot governor local projected goal
        self.loc_gbar, = self.ax.plot(self.g_bar[0], self.g_bar[1], 
                                      'o', color='orangered', alpha=0.6, markersize=10, label='local projected goal')
        # plot line segment back to path
        self.line_seg, = self.ax.plot(self.g_bar[0], self.g_bar[1], lw=2, color='yellowgreen')

        # plot optimized projected goal
        self.loc_ghat, = self.ax.plot(self.g_hat[0], self.g_hat[1], 
                                      'X', color='lime', markersize=8, label='optimized projected goal')
        pass

    def plotting_governor_tracking_update(self):
        """
        update governor tracking trojectory plot
        """
        # update local safe zone
        if self.radius_LS>0:
            self.circle_ls.center = (self.g[0], self.g[1])
            self.circle_ls.radius = self.radius_LS
            self.patch_ls.set_p = self.circle_ls
        # update robot prediction set
        if self.d_gz>0:
            self.circle_rs.center = (self.g[0], self.g[1])
            self.circle_rs.radius = self.d_gz
            self.patch_rs.set_p = self.circle_rs
        # update governor heading
        if self.ctrl_type=='Polar':
            awhead_rot = self.awhead.transformed(mpl.transforms.Affine2D().rotate(self.g[2]))
            self.loc_gvn.set_marker(awhead_rot)
        # update governor position
        self.loc_gvn.set_data(self.g[0], self.g[1])
        # update governor trajectory
        self.trj_gvn.set_data(self.g_log[:,0], self.g_log[:,1])
        # update governor local projected goal
        self.loc_gbar.set_data(self.g_bar[0], self.g_bar[1])
        # update optimized governor local projected goal
        self.loc_ghat.set_data(self.g_hat[0], self.g_hat[1])
        # update back-to-path line segment if any
        if len(self.line_seg_ends)>0:
            self.line_seg.set_data(self.line_seg_ends[0], self.line_seg_ends[1])
        pass