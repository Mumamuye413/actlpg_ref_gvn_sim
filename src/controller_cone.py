#!/usr/bin/env python

"""
Position tracking controller for unicycle-modeled robot 
  (angle convergence is not guranteed)

Ref:
https://arxiv.org/pdf/2209.12648.pdf
"""


import numpy as np
import matplotlib as mpl
from scipy.integrate import solve_ivp

from utils.tools_dynsys import unicycle_carts
from utils.tools_geometry import wrap_angle, wf2bf2D
from utils.tools_plot_common import create_arrowhead_patch


class ConeController:
    """ 
    Cone controller class to generate velocity control signal 
    given current and desired robot states.
    """
    
    # Running status table, higher number better status
    NORMAL = 1
    GOAL_LOC_REACHED = 2
    GOAL_POSE_REACHED = 10

    def __init__(self,
                 z_g,
                 z,
                 ctrl_bds=[],
                 ctrl_params=None,
                 ax=None,
                 dt=0.05,
                 eps_dist=0.5,
                 eps_angle=np.pi/20.0,
                 bi_direction=False):
        """
        Init cone controller
        System Dimensions:  
            states - global pose [z1, z2, z_theta]
            control input - local velocities [v, w]
            output - global trajectory [z1, z2, _],


        Inputs:
            z_g: goal pose [g1, g2, (g_theta)] in world frame
            z: current robot pose in world frame [z1, z2, z_theta]
            ctrl_bds: [v_min, v_max, w_min, w_max]
            dt: discretization time
            eps_dist: tolerence for position tracking distance error
            eps_angle: tolerence for position tracking heading error
        """

        print('--------Init Class ConeController Start------------')
        self._dt = dt
        self._eps_dist = eps_dist
        self._eps_angle = eps_angle
        self.bi_direction = bi_direction
        self.ax = ax
    
        self.gvec = z_g
        self.zvec = z
        self.xvec = np.zeros(3)
        self.uvec = np.zeros(2)
        self.v = 0.0
        self.zg_bf = np.zeros(2)

        # --------- log container-----------------
        self.log_uvec = np.array([self.uvec])
        self.log_zvec = np.array([z])
        self.log_xvec = np.array([self.xvec])
        self.log_t = [0]

        print('zvec0 [z1, z2, z_theta] =  %s' % self.zvec)
        print('gvec0 [g1, g2, g_theta] =  %s' % self.gvec)
        print('xvec0 [e, d_phi, z_phi] =  %s' % self.xvec)

        if ctrl_params is not None:
            # controller design parameters
            self.kv = ctrl_params["kv"]
            self.kw = ctrl_params["kw"]
        else:
            print("[ConeController] use default params")
            self.kv = 0.5
            self.kw = 1.5
        
        self._ctrl_bds = ctrl_bds

        self.tidx = 0
        self._nx = 3
        self._ng = len(z_g)
        self._nu = 2
        self._ny = len(z)
        self.warning_msg = None
        self.status = ConeController.NORMAL
    
    def cmp_tracking_error(self):

        zg = self.gvec
        z = self.zvec
        zg_bf = wf2bf2D(z, zg[0:2])
        
        e = zg_bf[0]
        if self.bi_direction:
            if abs(zg_bf[1])<self._eps_dist:
                e_phi = 0.0
            else:
                e_phi = np.arctan(zg_bf[1]/zg_bf[0])
        else:
            e_phi = np.arctan2(zg_bf[1], zg_bf[0]) # tracking heading error

        if len(zg)==3:
            z_phi = wrap_angle(zg[2] - z[2]) # tracking orientation error
        else:
            z_phi = 0.0

        xvec = np.array([e, e_phi, z_phi])

        self.zg_bf = zg_bf
        self.xvec = xvec
        self.log_xvec = np.vstack((self.log_xvec, xvec))
        return xvec

    def generate_control(self, debug=True):
        """
        Generate velocity control signal (v, w) given current robot states and desired robot states.
        Input:
            @self.zvec: current robot states (z1, z2, z_theta)
            @self.zvec_g: desired robot states (g1, g2, g_theta)
        """

        xvec = self.cmp_tracking_error()
        zg_bf = self.zg_bf
        e, e_phi, _ = xvec

        # ------------------ speical case  ----------------
        if np.linalg.norm(zg_bf) < self._eps_dist:
            v = 0.0
            w = 0.0
            msg = "[cone controller]  status = GOAL_LOC_REACHED"                
            self.warning_msg = msg
            
        else:
            self.warning_msg = None
            self.status = ConeController.NORMAL
            # ------------------ normal case  ----------------
            if self.bi_direction:
                v = self.kv * e
            else:
                v = self.kv * max(0, e)
            w = self.kw * e_phi

        if len(self._ctrl_bds)>0:
            # check input constrains
            if v < self._ctrl_bds[0]:
                v = self._ctrl_bds[0]
            elif v > self._ctrl_bds[1]:
                v = self._ctrl_bds[1]
            if w < self._ctrl_bds[2]:
                w = self._ctrl_bds[2]
            elif w > self._ctrl_bds[3]:
                w = self._ctrl_bds[3]

        if debug:
            print("input z = [%.2f, %.2f, %.2f]" % (self.zvec[0], self.zvec[1], self.zvec[2]))
            print("input z_g = [%.2f, %.2f, %.2f]" % (self.gvec[0], self.gvec[1], self.gvec[2]))
            print("[e, e_phi] = [%.2f, %.2f]" % (e, e_phi))
            print("[v, w] = [%.2f, %.2f]" % (v, w))

        uvec = np.array([v, w])
        self.v = v
        self.w = w
        self.uvec = uvec
        self.log_uvec = np.vstack((self.log_uvec, uvec))

        return uvec
    
    def update(self, zg, debug_info=False):
        """ update states of reference governor system via selected control
        law (pre-designed uvec_lyap, CLF-CBF-QP uvec_qp)
        This function is only for for unicyle now
        """
        self.gvec = zg
        zvec = self.zvec
        uvec = self.generate_control(debug=False)
        self.tidx +=1
        t = self.tidx * self._dt

        # real-robot state update
        zvec_sol = solve_ivp(unicycle_carts, [t-self._dt,t], zvec, t_eval=[t], args=(uvec[0], uvec[1]))
        zvec_up = zvec_sol.y[:,0]

        if debug_info:
            np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

            print("t = %4.2f" % (t))
            print('uvec = %s' % [uvec[0], uvec[1]])
            print('z  = %s' % zvec)
            print('z+ = %s' % zvec_up)
            print('--------------------------------------------')
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

        # update states and log
        self.zvec = zvec_up.copy()
        self.log_zvec = np.vstack((self.log_zvec, zvec_up))
        self.log_t.append(t)

        return zvec_up

    def check_goal_reached_rbt(self, debug_level=-1):
        """ check whether goal is reached for Unicycle
        """
        reached_flag = False
        
        if self.status == ConeController.GOAL_POSE_REACHED:
            if debug_level > 0:
                print("Goal configuraiton reached")
                reached_flag = True

        return reached_flag

    def plotting_robot_trj_init(self):
        """
        plot robot trojectory
        """
        self.awhead = create_arrowhead_patch()
        # show robot location and heading
        awhead_rot = self.awhead.transformed(mpl.transforms.Affine2D().rotate(self.zvec[2]))
        self.loc_rob, = self.ax.plot(self.zvec[0], self.zvec[1], marker=awhead_rot, ms=18, 
                               color='purple', label='robot pose')
        self.trj_rob, = self.ax.plot(self.log_zvec[:,0], self.log_zvec[:,1], 
                                     color='mediumpurple', lw=5, label='robot trajectory')
        self.annv = self.ax.annotate(r'$\|v\|$: 0.00s/m', xy=(5,85), fontsize='xx-large')
        pass

    def plotting_robot_trj_update(self):
        """
        plot robot trojectory
        """
        awhead_rot = self.awhead.transformed(mpl.transforms.Affine2D().rotate(self.zvec[2]))
        if self.tidx%50 == 0:
            self.ax.plot(self.zvec[0], self.zvec[1], marker=awhead_rot, ms=12, 
                         color='indigo', label='robot pose')
        self.loc_rob.set_data(self.zvec[0], self.zvec[1])
        self.loc_rob.set_marker(awhead_rot)
        self.trj_rob.set_data(self.log_zvec[:,0], self.log_zvec[:,1])
        v_norm = format(np.linalg.norm(self.v), '.2f')
        self.annv.set_text(r'$\|v\|$: '+str(v_norm)+'m/s')
        pass