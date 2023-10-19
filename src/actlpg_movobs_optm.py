
import numpy as np
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.tools_plot_common import create_arrowhead_patch

class MovingObstacleCheck:
    """
    This class:
        1) generated current moving obstacle states for safety checking
        2) computed CBF values to select the least safe moving obstacle
        3) constructed & solved QCQP to get optimized governor control input
    """

    def __init__(self, movobs_info_list, gamma=0.15, r_robot=0.33, reach_goal_eps=0.15):
        """
        INPUT
        movobs_info_list:
            p_start         2d initial position [x, y] (m)
            p_goal          2d goal position [x, y] (m)
            v_movobs        linear velocity (m/s)
            r_movobs        prediction set radius
        gamma               classK scaler
        r_robot             radius of robot circumcircle
        reach_goal_eps      moving obstacle reached goal tolerance
        """
        
        self.n_movobs = len(movobs_info_list)  # number of moving obstacles
        
        # initialize high dimension moving obstacle info arrays containers
        self.p_start = np.zeros((self.n_movobs, 2, 1))
        self.p_goal = np.zeros((self.n_movobs, 2, 1))
        self.v_movobs = np.zeros((self.n_movobs, 1, 1))
        self.r_movobs = np.zeros((self.n_movobs, 1, 1))

        # load data
        self.p_start[:,:,0] = movobs_info_list[:,0:2]
        self.p_goal[:,:,0] = movobs_info_list[:,2:4]
        self.v_movobs[:,:,0] = movobs_info_list[:,4:5]
        self.r_movobs[:,:,0] = movobs_info_list[:,5:]

        # obstacle heading angle
        p_delta_x = movobs_info_list[:,2]-movobs_info_list[:,0]
        p_delta_y = movobs_info_list[:,3]-movobs_info_list[:,1]
        self.movobs_heading = np.array([np.arctan2(p_delta_y, p_delta_x)])

        self.gamma = gamma
        self.r_robot = r_robot
        self.eps = reach_goal_eps
        
        # inflate moving obstacle set with robot radius
        self.R_movobs = self.r_movobs + self.r_robot

        # timer for moving obstacles start driving backwards
        self.t_restart = np.zeros((self.n_movobs,1,1))

        # initialize list to store indeces of moving obstacles switching heading
        self.switch_idx = []

        # moving obstacle set feature matrix
        self.Ao = np.repeat(np.array([np.eye(2)]), self.n_movobs, axis=0)
        self.Ao[:,0:1,0:1] = (1/self.R_movobs**2)
        self.Ao[:,1:,1:] = (1/self.R_movobs**2)

        # safety value container
        self.ho_log = np.empty(1)
        pass

    def get_moving_obstacle_state(self, time):
        """
        INPUT
        time               time(s)

        OUTPUT
        s               4d moving obstacle state [px, py, vx, vy]
        """
        self.time = format(time,'.2f')
        
        # moving obstacle switch direction after achieved current goal
        if len(self.switch_idx)>0:
            for idx in self.switch_idx:
                self.drive_back(idx)
                self.t_restart[idx,:,:] = time
                self.movobs_heading[0,idx] += np.pi

        # high dimension time container array 
        time_high_dim = np.ones((self.n_movobs,1,1))*time-self.t_restart

        # get state vec [px, py, vx, vy]
        cos_heading = np.array([np.cos(self.movobs_heading)]).T
        sin_heading = np.array([np.sin(self.movobs_heading)]).T
        v_x = cos_heading*self.v_movobs
        v_y = sin_heading*self.v_movobs
        v = np.concatenate((v_x, v_y), axis=1)
        p_x = self.p_start[:,0:1,:]+v_x*time_high_dim
        p_y = self.p_start[:,1:,:]+v_y*time_high_dim
        p = np.concatenate((p_x, p_y), axis=1)
        s = np.concatenate((p, v), axis=1)

        # get moving obstacle status (if reached current goal)
        p_diff = np.linalg.norm(self.p_goal-p, axis=1)
        self.switch_idx = np.argwhere(np.abs(p_diff[:,0])<self.eps)

        self.s = s
        return s
    
    def drive_back(self, movis_idx):
        """
        flip goal & start position to drive moving obstacles backwards after goal reached
        """      
        new_start = self.p_goal[movis_idx,:,:]
        new_goal = self.p_start[movis_idx,:,:]
        self.p_start[movis_idx,:,:] = new_start
        self.p_goal[movis_idx,:,:] = new_goal
        pass
    
    def safe_check_ho(self, x, g):
        """
        CBF equation:
            ho = (g-z)^T*Ao*(g-z) - (g-p)^T*Ao*(g-p) - 1 -2||g-p||/R_mo
            ho_dot = (partial_h/partial_g)*ug+(partial_h/partial_z)*w+(partial_h/partial_p)*v

        INPUT
        x        4d robot state (2d-position, 2d-velocity) [zx, zy, wx, wy] 
        g        2d governor state (2d-position) [gx, gy]
        """
        z, w  = x[0:2], x[2:]       # robot position & velocity
        p, v = self.s[:,0:2,:], self.s[:,2:,:]       # moving obstacle position & velocity
        d_gz = np.linalg.norm(g-z)      # ||g-z||
        R_movobs = self.R_movobs
        gamma = self.gamma

        # augmented (higher dimension) arrays for matrix computation
        g_high_dim = np.repeat(np.array([np.array([g]).T]), self.n_movobs, axis=0)
        z_high_dim = np.repeat(np.array([np.array([z]).T]), self.n_movobs, axis=0)
        w_high_dim = np.repeat(np.array([np.array([w]).T]), self.n_movobs, axis=0)

        # compute ho_dot values for all moving obstacles
        Ao = self.Ao
        if d_gz<1e-3:
            bo = 0.0
            delta_gz = 0.0
        else:
            bo = 1/(R_movobs*d_gz)
            delta_gz = d_gz/R_movobs

        dgz_high_dim = g_high_dim - z_high_dim
        pdpg = bo*dgz_high_dim      # partial_dgz/partial_g
        pdpz = -pdpg                # partial_dgz/partial_z

        phopg = 2*Ao@(g_high_dim-p) + 2*Ao@(z_high_dim-g_high_dim) - pdpg   # partial_ho/partial_g
        phopz = 2*Ao@(g_high_dim-z_high_dim) - pdpz     # partial_ho/partial_z
        phopp = -2*Ao@(g_high_dim-p)                    # partial_ho/partial_p

        # compute ho values to select the least safe moving obstacle
        ho = np.transpose(g_high_dim-p,(0,2,1))@Ao@(g_high_dim-p) \
            -np.transpose(g_high_dim-z_high_dim,(0,2,1))@Ao@(g_high_dim-z_high_dim)-1-delta_gz
        minh_idx = np.argmin(ho[:,0,0])
        min_ho = ho[minh_idx,0,0]

        # compute h_dot of the least safe moving obstacle: ho_dot = ho_dot_g*ug+ho_dot_f
        ho_dot_f = phopz[minh_idx].T@w_high_dim[minh_idx] + phopp[minh_idx].T@v[minh_idx]
        ho_dot_g = phopg[minh_idx].T
        
        # safety status
        g_safe = min_ho>=0

        self.ho_dot_f = ho_dot_f[0,0]
        self.ho_dot_g = ho_dot_g[0]

        self.classk_func_g = gamma*(min_ho**2)
        self.ho_log = np.hstack((self.ho_log, min_ho))
        
        self.g = g
        self.z = z
        self.p = p
        return g_safe
    
    def setup_cp_problem(self, P_mat):
        """
        initially setup CVXPY problem
        """
        x = cp.Variable(2)
        P = cp.Parameter((2,2), value=P_mat)
        q = cp.Parameter(2)
        l = cp.Parameter()
        A = cp.Parameter((2))
        m = cp.Parameter()
        n = cp.Parameter()

        objective = cp.Minimize(cp.norm(x-P@q))
        constraints = [l<=A@x, cp.norm(x)<=m]
        prob = cp.Problem(objective, constraints)

        self.x = x
        self.q = q
        self.l = l
        self.A = A
        self.m = m
        self.n = n
        self.prob = prob

        return prob
    
    def update_cp_problem(self, u_nom):
        """
        update parameters in CVXPY problem
        """

        self.q.value = u_nom
        self.l.value = -self.classk_func_g-self.ho_dot_f
        self.A.value = self.ho_dot_g
        self.m.value = np.linalg.norm(u_nom)
        
        self.prob.solve(verbose=False)
        x_star = self.x.value
        if x_star is None:
            print(self.prob.status)
            x_star = np.zeros(2)

        return x_star

    def plot_mo_set_init(self, ax):
        """
        initial prediction set plot of moving obstacle
        """
    
        self.movis_dic_list=[]
        awhead = create_arrowhead_patch()

        for movis_i in range(self.n_movobs):

            # show moving obstacle trajectory and start point
            ax.plot([self.p_start[movis_i,0,0], self.p_goal[movis_i,0,0]], 
                         [self.p_start[movis_i,1,0], self.p_goal[movis_i,1,0]], 
                         color='gray', lw=4, dashes=[6, 2])
            ax.plot(self.p_start[movis_i,0,0], self.p_start[movis_i,1,0], 'o',
                         color='gray', ms=8)
            
            # circle patches as moving obstacle sets
            movis_dic={}
            circle_mo = patches.Circle(
                        (self.s[movis_i,0,0], self.s[movis_i,1,0]), 
                        self.r_movobs[movis_i,0,0],
                        edgecolor='black',
                        facecolor='gray',
                        alpha=0.4,
                        fill=True)
            movis_dic['circl'] = circle_mo
            patch_mo = ax.add_patch(circle_mo)
            movis_dic['patch'] = patch_mo

            # annotate Time(s)
            ann_t = ax.annotate('Time: 0.00s', xy=(5,90), fontsize='xx-large')
            movis_dic['time'] = ann_t

            # show moving obstacle location and heading
            awhead_rot = awhead.transformed(mpl.transforms.Affine2D().rotate(self.movobs_heading[0,movis_i]))
            loc_mo, = ax.plot(self.s[movis_i,0,0], self.s[movis_i,1,0], marker=awhead_rot, ms=15, color='black')
            movis_dic['loc'] = loc_mo

            self.movis_dic_list.append(movis_dic)

            # annotate moving obstacle velocity and radius
            ax.annotate(r'$v_{%s}=$'%(movis_i+1)+str(self.v_movobs[movis_i,0,0])+'m/s',     # annotate string e.g.: v_1 = 0.21 m/s
                        xy=(self.p_start[movis_i,0,0]+2, self.p_start[movis_i,1,0]-2),  # annotation location (p_start_x+2, p_start_y-2)
                        fontsize='large')
            ax.annotate(r'$r_{%s}=$'%(movis_i+1)+str(self.r_movobs[movis_i,0,0])+'m',   # annotate string e.g.: r_1 = 0.1 m
                        xy=(self.p_start[movis_i,0,0]+2, self.p_start[movis_i,1,0]-5),  # annotation location (p_start_x+2, p_start_y-5)
                        fontsize='large')
        
        self.ax = ax
        self.awhead = awhead
        pass

    def plot_mo_set_update(self):
        """
        update moving obstacle set vis markers
        """

        for movis_j in range(self.n_movobs):
            
            # get current state
            movis_dic = self.movis_dic_list[movis_j]
            
            # update moving obstacle circle
            movis_dic['circl'].center = (self.s[movis_j,0,0], self.s[movis_j,1,0])
            
            # update arrow heading & position
            awhead_rot = self.awhead.transformed(mpl.transforms.Affine2D().rotate(self.movobs_heading[0,movis_j]))
            movis_dic['loc'].set_data(self.s[movis_j,0], self.s[movis_j,1])
            movis_dic['loc'].set_marker(awhead_rot)
            
            # update time counter
            movis_dic['time'].set_text('Time: '+str(self.time)+'s')
        
        pass
    

if __name__ == '__main__':
    # init figure
    fig1, ax = plt.subplots(figsize=(12,5))
    ax_sim = ax
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 120)
    ax.set_aspect('equal',"box")

    # set time & stop criteria
    dt = 0.05    # s/step
    t_max = 10000              # max simulation time steps
    tol = 0.3                  # reaching goal tolerance

    # moving obstacle list [p_startx, p_starty, p_goalx, p_goaly, v_mo, r_mo]
    # kv_mo: moving obstacle speed gain; r_mo: moving obstacle prediction set radius
    movobs_info_list = np.array([[50, 85, 107, 85, 2.0, 3.5],
                                [80, 60, 15, 60, 2.6, 4],
                                [70, 20, 70, 95, 1.9, 3],
                                [88, 40, 88, 95, 2.5, 4],
                                [78, 30, 78, 95, 4.7, 3],
                                [55, 25, 55, 85, 4.0, 5],
                                [15, 40, 15, 10, 3.0, 6],
                                [30, 80, 30, 30, 2.4, 4],])
    
    # set up moving obstacle & safe check
    gvn_param_dic = {
    'kg': 1,       # governor control gain
    'Lf': 0,       # looking ahead distance
    'eps': 1e-3,   # governor reach goal tolerance
    'eta': -0.2    # local safe zone tolerance
    }
    mo_param_dic = {
        'gamma': 0.7,           # class-K function scale
        'gvn': gvn_param_dic    # governor parameters
    }
    mo_turn = False     # moving obstacle not driving backwards
    ti = 1       # initial timestep

    MOCheck = MovingObstacleCheck(movobs_info_list)
    s = MOCheck.get_moving_obstacle_state(0)
    MOCheck.plot_mo_set_init(ax_sim)

    # Start driving
    while ti< t_max:
        # t = round(ti*dt,2)
        s = MOCheck.get_moving_obstacle_state(ti*dt)   #((ti-t_restart)*dt)
        # plot moving obstacle
        MOCheck.plot_mo_set_update()
        ti +=1
        plt.pause(0.0001)
    plt.show()

