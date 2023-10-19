#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stablilizing Unicycle (position & heading convergence) using CLF-CBF QP approach.

Ref:
    Closed-Loop Steering of Unicycle-like Vehicles via Lyapunov Techniques
    (M Aicardi - 1995)

x_dot = u cos (psi)
y_dot = u sin (psi)
psi_dot = omega

Cartesian position x, y, and its orientation angle psi

Modified to be bi-directional.
"""


import numpy as np
import numpy.linalg as npla
from scipy.integrate import solve_ivp

from utils.tools_geometry import wrap_angle, wf2bf2D
from utils.tools_dynsys import unicycle_polar


class PolarController:
    """ 
    Create class to generate control signal for unicycle system
    with polar controller
    """
    def __init__(self,
                 zg,
                 z,
                 ctrl_gains,
                 clf_paras,
                 ctrl_bds,
                 dt=0.05,
                 eps_dist=0.05, 
                 eps_angle=np.pi/20.0,
                 bi_direction=False):

        """ 
        Init PolarController

        System Dimensions:  
            states - local tracking error [e, e_phi, d_phi]
            control input - local velocities [v, w]
            output - global trajectory [z1, z2, _]

        Inputs:
            zg: goal pose [g1, g2, g_theta] in world frame
            z: current robot pose in world frame [z1, z2, z_theta]
            ctrl_gains: parameters of controller
            clf_paras: parameters of pre-designed Lyapunov function
            ctrl_bds: [v_min, v_max, w_min, w_max]
            dt: discretization time
            eps_dist: tolerence for position tracking distance error
            eps_angle: tolerence for position tracking heading error
            bi_direction: using bi-directional control or not
        """

        print('--------Init Class PolarController Start------------')
        self._dt = dt
        self._eps_dist = eps_dist
        self._eps_angle = eps_angle
        self.bi_direction = bi_direction
        
        self.gvec = zg
        self.zvec = z
        xvec0 = self.trans_gc2lp(zg)
        self.xvec = xvec0
        print('zvec0 [z1, z2, z_theta] =  %s' % self.zvec)
        print('gvec0 [g1, g2, g_theta] =  %s' % self.gvec)
        print('xvec0 [e, e_phi, d_phi] =  %s' % self.xvec)

        self._ctrl_gains = ctrl_gains
        self._clf_paras = clf_paras
        self._ctrl_bds = ctrl_bds

        # addtional properties
        self.tidx = 0
        self._nx = len(xvec0)
        self._ng = len(zg)
        self._nu = 2
        self._ny = len(z)

        # --------- log container-----------------
        # control signal
        self.log_uvec = np.empty((0,2))
        self.log_t = [0]

        # states
        self.log_gvec = np.array([zg])
        self.log_zvec = np.array([z])
        self.log_xvec = np.array([xvec0])
        
        # lyap functions log
        self.log_lyap = [self.cpt_lyap()]

    def check_backgoal(self, g_position, z_pose):
        """
        Check if goal position is in the bi_direction half plane w.r.t. robot heading
        """
        g_bf = wf2bf2D(z_pose, g_position)

        if g_bf[0]<0:
            v_dirct = -1
        else:
            v_dirct = 1

        self.g_bf = g_bf
        return v_dirct

    def trans_gc2lp(self, gvec):
        """
        Transform global cartesian coordinates zvec = [z1, z2, z_theta] in
        frame-W to local polar coordiantes xvec = [e, e_phi, d_phi/z_phi] in frame-G.
        """

        zvec = self.zvec
        g1, g2, g_theta = gvec
        z1, z2, z_theta = zvec

        e = npla.norm(gvec[0:2] - zvec[0:2])
        d_theta = np.arctan2(g2 - z2, g1 - z1)

        if self.bi_direction:
            self.kv_dirct = self.check_backgoal(gvec[0:2], zvec)
            g1_bf, g2_bf = self.g_bf
            e_phi = np.arctan(g2_bf/g1_bf)
            if self.kv_dirct<0:
                g_theta_bw = wrap_angle(g_theta + np.pi)
                d_phi = wrap_angle(d_theta - g_theta_bw)
            else:
                d_phi = wrap_angle(d_theta - g_theta)
        else:
            e_phi = wrap_angle(d_theta - z_theta)
            d_phi = wrap_angle(d_theta - g_theta)
        
        xvec = np.array([e, e_phi, d_phi])
        self.xvec = xvec
        self.gvec = gvec
        return xvec

    def trans_lp2gc(self):
        """
        Transform local polar coordiantes xvec = [e, e_phi, d_phi] in frame-G 
        to global cartesian coordinates zvec = [z1, z2, z_theta] in frame-W.
        """
        gvec = self.gvec
        g1, g2, g_theta = gvec
        e, e_phi, d_phi = self.xvec

        d_theta = wrap_angle(g_theta + d_phi)
        z_theta = wrap_angle(d_theta - e_phi)
        if self.bi_direction:
            if self.kv_dirct<0:
                g_theta_bw = wrap_angle(g_theta + np.pi)
                d_theta = wrap_angle(g_theta_bw + d_phi)
                z_theta_bw = wrap_angle(d_theta - e_phi)
                z_theta = wrap_angle(z_theta_bw + np.pi)

        z1 = g1 - e * np.cos(d_theta)
        z2 = g2 - e * np.sin(d_theta)
        zvec = np.array([z1, z2, z_theta])
        self.zvec = zvec
        return zvec

    def cpt_lyap(self):
        """ 
        compute lyapunov function quadratic form
        """
        xvec = self.xvec
        diag_P = 0.5 * np.array([self._clf_paras['ke'], 1, self._clf_paras['kdphi']])
        self._lyap_P = np.diag(diag_P)
        V = xvec.T @ self._lyap_P @ xvec
        return V

    def generate_control(self, eps_angle=1e-6):
        """ 
        compute control signal using pre-design offline methods
        via Lyapunov function
        """
        e, e_phi, d_phi = self.xvec
        kv =  self._ctrl_gains['kv']
        kephi = self._ctrl_gains['kephi']
        kdphi = self._ctrl_gains['kdphi']

        v = kv * np.cos(e_phi) * e
        tmp = np.sin(e_phi)/e_phi if np.abs(e_phi) > eps_angle else 1
        w = kephi * e_phi + kv * np.cos(e_phi) * tmp * (e_phi + kdphi * d_phi)
        
        # check input constrains
        if v < self._ctrl_bds[0]:
            v = self._ctrl_bds[0]
        elif v > self._ctrl_bds[1]:
            v = self._ctrl_bds[1]
        if w < self._ctrl_bds[2]:
            w = self._ctrl_bds[2]
        elif w > self._ctrl_bds[3]:
            w = self._ctrl_bds[3]
        
        uvec = np.array([v, w])
        self.uvec = uvec
        return uvec

    def update(self, zg, debug_info=True):
        """ 
        update robot local error states and global pose with polar control signal
        """
        V = self.cpt_lyap()
        xvec = self.trans_gc2lp(zg)
        uvec = self.generate_control()

        self.tidx +=1
        t = self.tidx * self._dt

        # real-robot state update
        xvec_sol = solve_ivp(unicycle_polar, [t-self._dt,t], xvec, t_eval=[t], args=(uvec[0],uvec[1]))
        xvec_up = xvec_sol.y[:,0]
        self.xvec = xvec_up
        zvec_up = self.trans_lp2gc()
        self.zvec = zvec_up

        if debug_info:
            np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
            print("t = %4.2f" % (t))
            print('uvec(%s) = %s' % (uvec))
            print('xvec  = %s' % self.xvec)
            print('xvec+ = %s' % xvec_up)
            print('V(x) = %.4f' % V)
            print('--------------------------------------------')
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        
        self.log_xvec = np.vstack((self.log_xvec, xvec_up))
        self.log_zvec = np.vstack((self.log_zvec, zvec_up))
        self.log_uvec = np.vstack((self.log_uvec, uvec))
        self.log_t.append(t)

    def check_goal_reached_rbt(self, debug_level=-1):
        """ 
        check whether goal is reached for Unicycle
        """
        reached_flag_dist = False
        reached_flag = False
        e, e_phi, _ = self.xvec
        V = self.cpt_lyap()

        if e < self._eps_dist:
            if debug_level > 0 and (not reached_flag_dist):
                    print("Goal position reached")
                    reached_flag_dist = True
            if np.abs(e_phi) < self._eps_angle:
                if debug_level > 0:
                    print("Goal configuraiton reached")
                reached_flag = True
                V = self.cpt_lyap()
        else:
            if debug_level > 10:
                e_phi_deg = np.rad2deg(e_phi)
                print('e = %.4f e_phi = %.4f deg V = %.4f' %(e, e_phi_deg, V) )
        return reached_flag
