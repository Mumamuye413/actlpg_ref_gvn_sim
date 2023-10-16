#%%
import time
import numpy as np
from matplotlib import pyplot as plt

from rrt import plotting, env
from llc.pd_controller import PDController
from llc.controller_cone import ConeController
from llc.controller_polar import PolarController
from moving_obstacle_check import MovingObstacleCheck
from gvn.ref_gvn import RbtGovSys

from utils.tools_geometry import wrap_angle
from utils.tools_plot_common import my_default_plt_setting, save_fig_to_folder

my_default_plt_setting() 

#%% medium 2d map

# ------------- medium 2d map---------------------
# map & way_points of path generated from rrt_star
medium_map_rec =[[30, 1, 10, 10],
                 [40, 1, 8, 47],
                 [1, 65, 20, 5],
                 [30, 85, 15, 15],
                 [100, 25, 20, 25]]
env_map_m = env.Env(xrange=120, yrange=100, rec_obs=medium_map_rec)
waypoints_m = np.array([[  7,  14],
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
                        [110,  95]])
# set start & goal position
sandg_m = [(7, 14),  (110, 95)]       # Starting & Goal Nodes
# set figure size
figsize_m = (12,15)

map_dic_m = {
    'map': env_map_m,
    'path': waypoints_m,
    'start_goal': sandg_m,
    'figsize': figsize_m
}

#-------------------1d map----------------------
# map & way_points of path generated from rrt_star
env_sx, env_sy = 80, 14
env_map_s = env.Env(xrange=env_sx, yrange=env_sy)
waypoints_sx = np.array([np.arange(0,env_sx+1,10)]).T
waypoints_sy = np.zeros((len(waypoints_sx),1))+env_sy/2
waypoints_s = np.hstack((waypoints_sx, waypoints_sy))
# set start & goal position
sandg_s = [(20, 7),  (55, 7)]
# set figure size
figsize_s = (12,5)

map_dic_s = {
    'map': env_map_s,
    'path': waypoints_s,
    'start_goal': sandg_s,
    'figsize': figsize_s
}

# set time & stop criteria
dt = 0.05    # s/step
t_max = 10000              # max simulation time steps
tol = 0.5                  # reaching goal tolerance

# ---------------------------------------------------------------------
# ------------------------ pick environment ---------------------------
# ---------------------------------------------------------------------
# <map_dic_s> for small 1d map, <map_dic_m> for medium 2d map
map_dic = map_dic_m
medium_env = True

env_map = map_dic['map']
way_points = map_dic['path']
global_start, global_goal = map_dic['start_goal']
figsize = map_dic['figsize']

# initial figure
fig1, ax = plt.subplots(figsize=figsize)
ax_sim = ax

# plot map with path & way_points
PlotMap = plotting.Plotting(global_start, global_goal, ax_sim, env_map)
PlotMap.plot_path(way_points)
PlotMap.plot_grid()
# plt.show()

#%%
# moving obstacle list [p_startx, p_starty, p_goalx, p_goaly, kv_mo, r_mo]
# kv_mo: moving obstacle speed gain; r_mo: moving obstacle prediction set radius
mo_list_m = np.array([
                    [70, 20, 70, 90, 0.042, 6],
                    [90, 65, 30, 65, 0.08, 6],
                    ])

mo_list_s = np.array([[10, 7, 36, 7, 0.0, 6],
                      [65, 7, 47, 7, 0.068, 6]])

mo_list_l = np.array([
                    [50, 85, 107, 85, 0.035, 3.5],
                    [80, 60, 15, 60, 0.04, 4],
                    [70, 20, 70, 95, 0.025, 3],
                    [88, 40, 88, 95, 0.045, 4],
                    [78, 30, 78, 95, 0.072, 3],
                    [55, 25, 55, 85, 0.066, 5],
                    [15, 40, 15, 10, 0.1, 6],
                    [30, 80, 30, 30, 0.048, 4],
                    ])

# ---------------------------------------------------------------------
# ------------------- pick moving obstacle list -----------------------
# ---------------------------------------------------------------------
# <mo_list_m> for 2 obstacles, <mo_list_l> for 8 obstacles
mo_list = mo_list_l

# set up moving obstacle & safe check
mo_param_dic = {
    'gamma': 0.15,           # class-K function scale   # governor parameters
}
t_restart = 0       # timer for moving obstacle starts driving backwards

MOCheck = MovingObstacleCheck(mo_list, mo_param_dic, GVN=True)
s = MOCheck.get_mo_state(0)
MOCheck.plot_mo_set_init(ax_sim)

# set up CVXPY solver
P_mat = np.eye(2)          # cost matrix for optimization object (p.d.)
qp_controller = MOCheck.setup_cp_problem(P_mat)

#%%
# --------------------------------------------------------------------------
# ------------------- pick low level controller type -----------------------
# --------------------------------------------------------------------------

ctrl_type_list = ['DInte', 'Cone', 'Polar']
ctrl_type = ctrl_type_list[1]
BWmode = True

#%%
# initialize controller
# input constraints
ctrl_bds = np.array([-10, 10, -4, 4]) # u1min, u1max, u2min, u2max

if ctrl_type == 'DInte':
    # set robot start & global goal
    x0 = np.zeros((4))
    x0[0], x0[1] = global_start[0], global_start[1]
    z_start = x0[0:2]
    z_goal = np.array([global_goal[0], global_goal[1]])

    pdc_param_dic = {
        'kp': 7,  # p-gain
        'kd': 2     # d-gain
    }
    Ctrl = PDController(x0, pdc_param_dic, ax_sim, dt=dt)

elif ctrl_type == 'Cone':
    z_start = np.array([global_start[0], global_start[1], np.pi/2])
    z_goal = np.array([global_goal[0], global_goal[1], 0])

    conec_param_dic = {
        'kv':2,   # linear velocity
        'kw':5   # angular velocity
    }

    Ctrl = ConeController(z_start, z_start, ctrl_bds, conec_param_dic, ax=ax_sim,
                          dt=dt, eps_dist=0.05, eps_angle=np.pi/20, backward=BWmode)

elif ctrl_type == 'Polar':
    z_start = np.array([global_start[0], global_start[1], np.pi/2])
    z_goal = np.array([global_goal[0], global_goal[1], 0])

    ctrl_gains={'kv':3.5,      # speed
                'kephi':6,     # heading error
                'kdphi':2}     # heading goal
    clf_paras={'ke':1, 
               'kdphi':1.5}
    
    Ctrl = PolarController(z_start, z_start, ctrl_gains=ctrl_gains, clf_paras=clf_paras, ctrl_bds=ctrl_bds, 
                           ax=ax_sim, dt=dt, eps_dist=0.05, eps_angle=np.pi/20, backward=BWmode)

Ctrl.plotting_robot_trj_init()

#%%
# initialize reference governor
gvn_param_dic = {
    'kg': 0.8,       # governor control gain
    'Lf': 0.8,     # looking ahead distance
    'eps': 1e-3,   # governor reach goal tolerance
    'eta': -1e-3    # local safe zone tolerance
}
RGsys = RbtGovSys(z_start, z_goal, gvn_param_dic, env_map, way_points, ax_sim, dt=dt)
RGsys.plotting_governor_tracking_init(ctrl_type=ctrl_type)

#%%
# Start driving
ti = 1
if medium_env:
    ax_sim.legend(loc='lower right')
plt.pause(1)
# fig1.savefig('env_lmap.pdf')
#-------------------------------------------------------------------
#----------------------------save fig/show safety-------------------------------
#-------------------------------------------------------------------
fig_sample_rate = 1
save_fig = False
save_video_fig = False
show_safe = True
t_show = [6.15, 9.80, 11.90, 12.70, 15.60]
ug_prev = np.zeros((2))
while ti< t_max:
    t = round(ti*dt,2)

    # ---- update governor state ----
    # get governor local projected goal 
    g_bar = RGsys.get_gvn_goal()
    # get nominal governor control input
    ug_nom = RGsys.get_governor_control(g_bar)

    # get moving obstacle state, robot state & governor state
    s = MOCheck.get_mo_state((ti-t_restart)*dt)
    g = RGsys.g
    if ctrl_type == 'DInte':
        x = Ctrl.x
    elif ctrl_type == 'Cone' or 'Polar':
        z = Ctrl.zvec
        u = Ctrl.uvec
        x = MOCheck.aug_robot_state(z, u)

    # plot moving obstacle
    MOCheck.plot_mo_set_update()
    # plot governor & robot tracking
    Ctrl.plotting_robot_trj_update()
    RGsys.plotting_governor_tracking_update()

    # check governor safery
    safe = MOCheck.safe_check_h1(x, g[0:2])

    #----use CVXPY solve for QCQP----
    # compute QCQP control input with moving obstacle
    cp_value_dic = MOCheck.update_cp_problem(ug_nom, ug_prev)
    ug = cp_value_dic['u']
    A_mat = cp_value_dic['A']
    ug_norm = cp_value_dic['u_norm']
    low_bd = cp_value_dic['l']
    up_bd = cp_value_dic['m']
    ug_prev = ug

    # update governor state
    g_next = RGsys.update_governor(x, ug)
    if ctrl_type == 'Polar':
        gp_bf = Ctrl.g_bf
        # g_next[2] = wrap_angle(np.arctan2(gp_bf[1], gp_bf[0]) + np.arctan2(x[1], x[0]))
        g_next[2] = np.arctan2(ug_nom[1], ug_nom[0])

    # update robot state
    x_next = Ctrl.update(g_next)

    # save figure
    fig_folder = 'fig'
    if (ti % fig_sample_rate ==0) and save_video_fig:
        fig1.canvas.flush_events()
        save_fig_to_folder(fig1, fig_folder, str(1000000+ti))
    elif (t in t_show) and save_fig:
        fig1.canvas.flush_events()
        save_fig_to_folder(fig1, fig_folder, 'sim_t_'+str(1000000+ti), ftype_ext='.pdf')
    elif (not save_video_fig) and (not save_fig):
        fig1.canvas.flush_events()
        time.sleep(dt)
        plt.pause(0.0001)

    # stop tracking when robot reached global goal
    tracking_error = np.linalg.norm(z_goal[0:2]-x_next[0:2])
    if tracking_error < tol:
        break
    else:
        ti += 1

if save_fig:
    save_fig_to_folder(fig1, fig_folder, 'sim_done', ftype_ext='.pdf')
# plot safe metrics evaluation
if show_safe:
    t_vec = np.arange(ti+1)*dt
    mo_safeM = MOCheck.hgo_log
    so_safeM = RGsys.deltaE_log
    fig2, ax_vis = plt.subplots(figsize=(15, 3))
    ax_vis.plot(t_vec, mo_safeM, color='gray', label=r'$h_o$', lw=3)
    ax_vis.plot(t_vec, so_safeM*0.1, color='goldenrod', label=r'${\Delta}E/10$', lw=4)
    ax_vis.legend()
    ax_vis.set_ylim(-5,20)
    for t_note in t_show:
        ax_vis.vlines(t_note, -5, 20, linestyles='dotted', color='black', lw=3)
        ax_vis.annotate(str(t_note)+'s', (t_note+0.1, 16))
    save_fig_to_folder(fig2, fig_folder, 'metric_eval', ftype_ext='.pdf')
    plt.show()