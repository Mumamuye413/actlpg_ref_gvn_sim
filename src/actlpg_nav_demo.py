#%%
import time
import numpy as np
from matplotlib import pyplot as plt

from controller_cone import ConeController
from controller_polar import PolarController
from actlpg_movobs_optm import MovingObstacleCheck
from ref_gvn import RbtGovSys

from utils.env_plotting import Env, Plotting
from utils.tools_plot_common import my_default_plt_setting, save_fig_to_folder


my_default_plt_setting() 

class NavigationDemo:
    """
    Three simulation environments available by picking different [map_size]:
        small  - 1d map(path) with two moving obstacles
        medium - 2d map with two moving obstacles
        large  - 2d map with eight moving obstacles
    the medium and large map have the same static environment setup,
    but different dynamic obstacles
    """

    def __init__(self, 
                 map_size,              # pick [small], [medium] or [large] environment
                 controller_type,       # position tracking [cone] or pose converging [polar] controller
                 bi_direction=True,     # active backward driving or not
                 save_fig="none",       # save sampled-frame [video]; save figure at picked time [time]; no figure saved [none]
                 fig_folder="fig",      # save figures to this folder
                 frame_sample_rate=1,   # for making video from figure frames
                 time_save=None,        # save figures at picked time list
                 show_safety=False,     # show second figure with safety metrics values
                 classK_gamma=0.15,     # class-K function scale
                 dt = 0.05,             # seconds/step
                 t_max = 100000,         # max simulation time steps
                 dist_tol = 0.5,        # reaching goal (position) tolerance (Euclidean distance)
                 ):
        
        self.map_size = map_size
        self.controller_type = controller_type
        self.bi_direction = bi_direction
        self.save_fig = save_fig
        self.fig_folder = fig_folder
        self.frame_sample_rate = frame_sample_rate
        self.time_save = time_save
        self.show_safety = show_safety
        self.medium_env = False

        self.gamma = classK_gamma

        self.dt = dt
        self.t_max = t_max
        self.dist_tol = dist_tol

        pass

    def small_map(self):
        """
        set up small 1d map 
        """

        # pick map size (m)
        env_size_x, env_size_y = 80, 14
        env_map = Env(xrange=80, yrange=14)

        # pick path waypoints
        waypoints_x = np.array([np.arange(0,env_size_x+1,10)]).T
        waypoints_y = np.zeros((len(waypoints_x),1))+env_size_y/2
        waypoints_list = np.hstack((waypoints_x, waypoints_y))

        # set start & goal position
        start_goal = [(20, 7),  (55, 7)]

        # set figure size
        figsize_small = (12,5)

        map_dic = {
            'map': env_map,
            'path': waypoints_list,
            'start_goal': start_goal,
            'figsize': figsize_small
        }

        return map_dic
    
    def medium_map(self):
        """
        set up medium 2d map 
        """

        # place obstacles on map: rectangulars with the pivot point at its bottom-left corner
        # [pivot_point_x, pivot_point_y, x_length, y_length]
        medium_map_obs =[[30, 1, 10, 10],
                        [40, 1, 8, 47],
                        [1, 65, 20, 5],
                        [30, 85, 15, 15],
                        [100, 25, 20, 25]]
        
        # pick map size (m)
        env_map = Env(xrange=120, yrange=100, rec_obs=medium_map_obs)

        # set start & goal position
        start_goal = [(7, 14),  (110, 95)]

        # set figure size
        figsize_medium = (12,15)

        # pick path waypoints
        waypoints_list = np.array( [[  7,  14],
                                    [  9,  24],
                                    [ 17,  30],
                                    [ 21,  40],
                                    [ 22,  48],
                                    [ 26,  55],
                                    [ 34,  60],
                                    [ 45,  62],
                                    [ 57,  65],
                                    [ 67,  67],
                                    [ 78,  71],
                                    [ 85,  78],
                                    [ 93,  83],
                                    [100,  86],
                                    [103,  88],
                                    [110,  95],] )

        map_dic = {
            'map': env_map,
            'path': waypoints_list,
            'start_goal': start_goal,
            'figsize': figsize_medium
        }

        return map_dic

    def plot_map(self):
        """
        initilize figure and visualize environment features
        """
        if self.map_size == "small":
            map_dic = self.small_map()
        elif self.map_size == "medium" or "large":
            map_dic = self.medium_map()
            self.medium_env = True      # legends on
        else:
            print("Error: please pick among [small], [medium] and [large]")
        
        # read params from map dictionary
        env_map = map_dic['map']
        waypoints = map_dic['path']
        global_start, global_goal = map_dic['start_goal']
        figsize = map_dic['figsize']

        # initial figure
        fig1, ax_sim = plt.subplots(figsize=figsize)

        # plot map with path & waypoints
        PlotMap = Plotting(global_start, global_goal, ax_sim, env_map)
        PlotMap.plot_path(waypoints)
        PlotMap.plot_grid()

        self.global_start = global_start
        self.global_goal = global_goal
        self.fig1 = fig1
        self.ax_sim = ax_sim
        self.map_dic = map_dic
        return

    def small_movobs_list(self):
        # moving obstacle list [p_start_x, p_start_y, p_goal_x, p_goal_y, v, r]
        movobs_list = np.array([[10, 7, 36, 7, 2.10, 6],
                                [65, 7, 47, 7, 1.22, 6]])
        
        return movobs_list
    
    def medium_movobs_list(self):
        # moving obstacle list [p_start_x, p_start_y, p_goal_x, p_goal_y, v, r]
        movobs_list = np.array([[70, 20, 70, 90, 4.0, 6],
                                [90, 65, 30, 65, 2.75, 6]])
        
        return movobs_list

    def large_movobs_list(self):
        # moving obstacle list [p_start_x, p_start_y, p_goal_x, p_goal_y, v, r]
        movobs_list = np.array([[50, 85, 107, 85, 2.0, 3.5],
                                [80, 60, 15, 60, 2.6, 4],
                                [70, 20, 70, 95, 1.9, 3],
                                [88, 40, 88, 95, 2.5, 4],
                                [78, 30, 78, 95, 4.7, 3],
                                [55, 25, 55, 85, 4.0, 5],
                                [15, 40, 15, 10, 3.0, 6],
                                [30, 80, 30, 30, 2.4, 4],
                                ])
        return movobs_list

    def init_movobs_check(self):
        """
        initialize moving obstacle safety check & QP controller
        """
        if self.map_size == "small":
            movobs_info_list = self.small_movobs_list()
        elif self.map_size == "medium":
            movobs_info_list = self.medium_movobs_list()
        elif self.map_size == "large":
            movobs_info_list = self.large_movobs_list()

        self.MOCheck = MovingObstacleCheck(movobs_info_list)
        self.s0 = self.MOCheck.get_moving_obstacle_state(0)
        self.MOCheck.plot_mo_set_init(self.ax_sim)
        
        # set up CVXPY solver
        P_mat = np.eye(2)          # cost matrix for optimization object (p.d.)
        qp_controller = self.MOCheck.setup_cp_problem(P_mat)

        self.movobs_info_list = movobs_info_list
        pass

    def init_controller(self):
        """
        initialize low-level controller
        """
        ax_sim = self.ax_sim
        dt = self.dt
        bi_direction = self.bi_direction
        global_start = self.global_start
        global_goal = self.global_goal

        z_start = np.array([global_start[0], global_start[1], np.pi/2])
        z_goal = np.array([global_goal[0], global_goal[1], 0])
        ctrl_bds = np.array([-10, 10, -4, 4]) # u1min, u1max, u2min, u2max

        if self.controller_type == 'Cone':

            conec_param_dic = {'kv':2,   # linear velocity
                               'kw':5}   # angular velocity
            
            self.Ctrl = ConeController(z_start, z_start, ctrl_bds, 
                                       conec_param_dic, ax=ax_sim, dt=dt, 
                                       eps_dist=0.05, eps_angle=np.pi/20, 
                                       bi_direction=bi_direction)

        elif self.controller_type == 'Polar':

            ctrl_gains = {'kv':3.5,      # speed
                          'kephi':6,     # heading error
                          'kdphi':2 }     # heading goal
            clf_paras = {'ke':1, 
                         'kdphi':1.5 }
            
            self.Ctrl = PolarController(z_start, z_start, ctrl_gains=ctrl_gains, 
                                        clf_paras=clf_paras, ctrl_bds=ctrl_bds, 
                                        ax=ax_sim, dt=dt, eps_dist=0.05, 
                                        eps_angle=np.pi/20, bi_direction=bi_direction)
        else:
            print("Please pick between [Cone] and [Polar]")

        self.Ctrl.plotting_robot_trj_init()
        self.z_start = z_start
        self.z_goal = z_goal
        self.ax_sim = ax_sim
        pass

    def init_governor_system(self):
        """
        initialize reference governor
        """
        z_start = self.z_start
        z_goal = self.z_goal
        map_dic = self.map_dic
        ax_sim = self.ax_sim
        dt = self.dt
        ctrl_type = self.controller_type
        
        env_map = map_dic['map']
        waypoints = map_dic['path']

        gvn_param_dic = {
            'kg': 0.8,       # governor control gain
            'Lf': 0.8,     # looking ahead distance
            'eps': 1e-3,   # governor reach goal tolerance
            'eta': -1e-3    # local safe zone tolerance
        }

        self.RGsys = RbtGovSys(z_start, z_goal, gvn_param_dic, 
                               env_map, waypoints, ax_sim, dt=dt)
        self.RGsys.plotting_governor_tracking_init(ctrl_type=ctrl_type)
        pass

    @staticmethod
    def aug_robot_state(z, u):
        """
        combine robot pose and control input for differential drive model 
        to get augmented robot state x: [zx, zy, vz, vy]
        """
        zx, zy, ztheta = z
        v, _ = u
        vx = v*np.cos(ztheta)
        vy = v*np.sin(ztheta)
        x = np.array([zx, zy, vx, vy])
        return x
    
    def save_figure(self, ti):

        fig_folder = self.fig_folder
        save_fig = self.save_fig
        fig1 = self.fig1
        dt = self.dt
        t = round(ti*dt,2)

        if save_fig == "video":
            if (ti % self.frame_sample_rate == 0):
                fig1.canvas.flush_events()
                save_fig_to_folder(fig1, 
                                   fig_folder,
                                   str(1000000+ti),
                                   dpi=150)
        
        elif save_fig == "time":
            if (t in time_save):
                fig1.canvas.flush_events()
                save_fig_to_folder(fig1, 
                                   fig_folder, 
                                   'sim_t_'+str(1000000+ti),
                                   ftype_ext='.pdf')
        
        elif save_fig == "none":
            fig1.canvas.flush_events()
            time.sleep(dt)
            plt.pause(0.0001)

    def show_safety_figure(self, ti):

        time_save = self.time_save
        fig_folder = self.fig_folder

        dt = self.dt
        t_vec = np.arange(ti+1)*dt

        mo_safeM = self.MOCheck.ho_log      # dynamic environment safety value
        so_safeM = self.RGsys.deltaE_log    # static environment safety value

        # visualize safety values
        fig2, ax_vis = plt.subplots(figsize=(15, 3))
        ax_vis.plot(t_vec, mo_safeM, color='gray', 
                    label=r'$h_o$', lw=3)
        ax_vis.plot(t_vec, so_safeM*0.1, color='goldenrod', 
                    label=r'${\Delta}E/10$', lw=4)
        ax_vis.legend()
        ax_vis.set_ylim(-5,20)

        # save figure with reference lines at picked times
        if self.save_fig == "time":
            for t_note in time_save:
                ax_vis.vlines(t_note, -5, 20, linestyles='dotted', color='black', lw=3)
                ax_vis.annotate(str(t_note)+'s', (t_note+0.1, 16))
            save_fig_to_folder(fig2, fig_folder, 'metric_eval', ftype_ext='.pdf')
        plt.show()

    def drive(self):

        # initialize system
        self.plot_map()
        self.init_movobs_check()
        self.init_controller()
        self.init_governor_system()

        # plot initial environment map
        if self.medium_env:
            self.ax_sim.legend(loc='lower right')
        plt.pause(1)

        ctrl_type = self.controller_type
        z_goal = self.z_goal
        ti = 1
        t_max = self.t_max
        dt = self.dt

        # Start driving
        while ti< t_max:
            
            # ---- update governor state ----
            # get governor local projected goal 
            g_bar = self.RGsys.get_gvn_goal()
            # get nominal governor control input
            ug_nom = self.RGsys.get_governor_control(g_bar)

            # get moving obstacle state, robot state & governor state
            s = self.MOCheck.get_moving_obstacle_state(ti*dt)
            g = self.RGsys.g
            z = self.Ctrl.zvec
            u = self.Ctrl.uvec
            x = self.aug_robot_state(z, u)

            # plot moving obstacle
            self.MOCheck.plot_mo_set_update()
            # plot governor & robot tracking
            self.Ctrl.plotting_robot_trj_update()
            self.RGsys.plotting_governor_tracking_update()

            # check governor safery
            safe = self.MOCheck.safe_check_ho(x, g[0:2])

            #----use CVXPY solve for QCQP----
            # compute QCQP control input with moving obstacle
            ug = self.MOCheck.update_cp_problem(ug_nom)

            # update governor state
            g_next = self.RGsys.update_governor(x, ug)
            if ctrl_type == 'Polar':
                g_next[2] = np.arctan2(ug_nom[1], ug_nom[0])

            # update robot state
            x_next = self.Ctrl.update(g_next)

            # save figure
            self.save_figure(ti)

            # stop tracking when robot reached global goal
            tracking_error = np.linalg.norm(z_goal[0:2]-x_next[0:2])
            if tracking_error < self.dist_tol:
                break
            else:
                ti += 1

        if self.save_fig == "none":
            save_fig_to_folder(self.fig1, self.fig_folder, 
                               'sim_done', ftype_ext='.pdf')

        # plot safe metrics evaluation
        if self.show_safety:
            self.show_safety_figure(ti)


if __name__ == '__main__':

    frame_sample_rate = 1   # for making video from figure frames
    time_save = [6.15, 9.80, 11.90, 12.70, 15.60]   # e.g. save figures at picked time

    NavDemo = NavigationDemo(map_size="medium", 
                             controller_type="Cone",
                             bi_direction=True,
                             save_fig="video",
                             frame_sample_rate=1,)
    NavDemo.drive()
