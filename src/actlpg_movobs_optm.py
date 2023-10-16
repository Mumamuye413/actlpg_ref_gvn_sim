"""
Simulating single moving obstacle abstracted as a circle, tracking goal in constant linear velocity
and ignoring orientation (no steering).
"""

import numpy as np
import osqp
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse import csc_matrix
from utils.tools_plot_common import create_arrowhead_patch

class MovingObstacleCheck:
    """
    Check safety with dynamic obstacles 
    if control input unsafe, formulate inputs for QP controller solver
    """

    def __init__(self, mo_list, param_dic, r_robot=0.33, eps_modist=0.15, GVN=False):
        """
        INPUT
        p_start         2d initial position [x, y] (m)
        p_goal          2d goal position [x, y] (m)
        v               linear velocity (m/s)
        r_mo            prediction set radius
        """
        self.mo_n = len(mo_list)
        self.p_start = np.zeros((self.mo_n, 2, 1))
        self.p_goal = np.zeros((self.mo_n, 2, 1))
        self.Ao = np.repeat(np.array([np.eye(2)]), self.mo_n, axis=0)
        self.kv_mo = np.zeros((self.mo_n, 1, 1))
        self.r_mo = np.zeros((self.mo_n, 1, 1))
        self.t_restart = np.zeros((self.mo_n,1,1))  # timer for moving obstacles start driving backwards
        self.eps = eps_modist
        self.r_robot = r_robot

        self.p_start[:,:,0] = mo_list[:,0:2]
        self.p_goal[:,:,0] = mo_list[:,2:4]
        self.kv_mo[:,:,0] = mo_list[:,4:5]
        self.r_mo[:,:,0] = mo_list[:,5:]
        self.R_mo = self.r_mo + self.r_robot

        # rotation angle for visualizing obstacle heading
        p_delta_norm = np.linalg.norm(mo_list[:,2:4]-mo_list[:,0:2], axis=1)
        p_delta_x = mo_list[:,2]-mo_list[:,0]
        p_delta_y = mo_list[:,3]-mo_list[:,1]
        self.mo_heading = np.arctan2(p_delta_y/p_delta_norm, p_delta_x/p_delta_norm)

        self.gamma = param_dic['gamma']
        self.db_idx = []    # index list of driving backwards obstacles

        self.Ao[:,0:1,0:1] = (1/self.R_mo**2)
        self.Ao[:,1:,1:] = (1/self.R_mo**2)
        self.GVN = GVN

        self.hgo_log = np.empty(1)
        pass

    def get_mo_state(self, t):
        """
        INPUT
        t               time(s)

        OUTPUT
        s               4d moving obstacle state [px, py, vx, vy]
        """
        self.t = format(t,'.2f')
        
        # moving obstacle return after achieved current goal
        if len(self.db_idx)>0:
            for idx in self.db_idx:
                self.drive_back(idx)
                self.t_restart[idx,:,:] = t
                self.mo_heading[idx] += np.pi
        t_aug = np.ones((self.mo_n,1,1))*t-self.t_restart

        # get state vec
        p_delta = self.p_goal - self.p_start
        self.p_delta = p_delta
        v_x = p_delta[:,0:1,:]*self.kv_mo
        v_y = p_delta[:,1:,:]*self.kv_mo
        v = np.concatenate((v_x, v_y), axis=1)
        p_x = self.p_start[:,0:1,:]+v_x*t_aug
        p_y = self.p_start[:,1:,:]+v_y*t_aug
        p = np.concatenate((p_x, p_y), axis=1)
        self.s = np.concatenate((p, v), axis=1)

        # get moving obstacle status (if reached current goal)
        p_diff = np.linalg.norm(self.p_goal-p, axis=1)
        self.db_idx = np.argwhere(np.abs(p_diff[:,0])<self.eps)

        s = self.s
        return s
    
    def drive_back(self, mo_idx):
        """
        flip goal & start position to drive moving obstacles backwards after goal reached
        """      
        new_start = self.p_goal[mo_idx,:,:]
        new_goal = self.p_start[mo_idx,:,:]
        self.p_start[mo_idx,:,:] = new_start
        self.p_goal[mo_idx,:,:] = new_goal
        pass
    
    def safe_check_h1(self, x, g):
        """
        Compute CBF h1=||g-p||^2-(R+||g-z||)^2

        INPUT
        x        4d robot state [zx, zy, wx, wy]
        g        2d governor state [gx, gy]
        """
        z, w  = x[0:2], x[2:]
        p, v = self.s[:,0:2,:], self.s[:,2:,:]
        d_gz = np.linalg.norm(g-z)
        R_mo = self.R_mo
        gamma = self.gamma

        aug_g = np.repeat(np.array([np.array([g]).T]), self.mo_n, axis=0)
        aug_z = np.repeat(np.array([np.array([z]).T]), self.mo_n, axis=0)
        aug_w = np.repeat(np.array([np.array([w]).T]), self.mo_n, axis=0)

        Ao = self.Ao
        if d_gz<1e-3:
            bo = 0.0
            delta_gz = 0.0
        else:
            bo = 1/(R_mo*d_gz)
            # delta_gz = (2*d_gz)/R_mo
            delta_gz = d_gz/R_mo
        aug_dgz = aug_g - aug_z
        pdpg = bo*aug_dgz
        pdpz = -pdpg

        phopg = 2*Ao@(aug_g-p) + 2*Ao@(aug_z-aug_g) - pdpg
        phopz = 2*Ao@(aug_g-aug_z) - pdpz
        phopp = -2*Ao@(aug_g-p)

        hgo = np.transpose(aug_g-p,(0,2,1))@Ao@(aug_g-p)-np.transpose(aug_g-aug_z,(0,2,1))@Ao@(aug_g-aug_z)-1-delta_gz
        minh_ind = np.argmin(hgo[:,0,0])
        min_hgo = hgo[minh_ind,0,0]
        min_hov = v[minh_ind]

        hgo_prime_f = phopz[minh_ind].T@aug_w[minh_ind] + phopp[minh_ind].T@v[minh_ind]
        hgo_prime_g = phopg[minh_ind].T

        self.hgo_prime_f = hgo_prime_f[0,0]
        self.hgo_prime_g = hgo_prime_g[0]

        self.classk_func_g = gamma*(min_hgo**2)

        g_safe = min_hgo>=0

        self.hgo_log = np.hstack((self.hgo_log, min_hgo))
        self.vmin = min_hov
        self.g = g
        self.z = z
        self.p = p
        return g_safe
    
    def safe_check_h2(self, x, g, beta=1.1):
        """
        Compute CBF h2=||g-p||-(R+||g-z||)

        INPUT
        x        4d robot state [zx, zy, wx, wy]
        g        2d governor state [gx, gy]
        """
        z, w  = x[0:2], x[2:]
        p, v = self.s[:,0:2,:], self.s[:,2:,:]
        R_mo = self.R_mo
        gamma = self.gamma
        dgz = np.linalg.norm(g-z)

        aug_g = np.repeat(np.array([np.array([g]).T]), self.mo_n, axis=0)
        aug_z = np.repeat(np.array([np.array([z]).T]), self.mo_n, axis=0)
        aug_w = np.repeat(np.array([np.array([w]).T]), self.mo_n, axis=0)

        dgz_norm = np.linalg.norm(aug_g-aug_z, axis=1)
        aug_dgz_norm = np.zeros((self.mo_n, 1, 1))
        aug_dgz_norm[:,:,0] = dgz_norm

        dgp_norm = np.linalg.norm(aug_g-p, axis=1)
        aug_dgp_norm = np.zeros((self.mo_n, 1, 1))
        aug_dgp_norm[:,:,0] = dgp_norm

        dgz_vec = (aug_g - aug_z)
        dgp_vec = (aug_g - p)

        phopp = -dgp_vec/aug_dgp_norm
        if dgz<1e-5:
            phopz = np.zeros((self.mo_n, 2, 1))
        else:
            phopz = beta*dgz_vec/aug_dgz_norm
        phopg = -phopp - phopz

        hgo = aug_dgp_norm - beta*(aug_dgz_norm + R_mo)
        minh_ind = np.argmin(hgo[:,0,0])
        min_hgo = hgo[minh_ind,0,0]
        min_hov = v[minh_ind]        

        hgo_prime_g = phopg[minh_ind].T
        hgo_prime_f = phopz[minh_ind].T@aug_w[minh_ind] + phopp[minh_ind].T@v[minh_ind]

        self.hgo_prime_f = hgo_prime_f[0,0]
        self.hgo_prime_g = hgo_prime_g[0]
        self.classk_func_g = gamma*(min_hgo**2)

        g_safe = min_hgo>=0

        self.hgo_log = np.hstack((self.hgo_log, min_hgo))
        self.vmin = min_hov
        self.g = g
        self.z = z
        self.p = p
        return g_safe
    
    def safe_check_h3(self, x, g, beta2=1.0, beta1=0.1):
        """
        Compute CBF h3=beta2*h2+beta1*h1

        INPUT
        x        4d robot state [zx, zy, wx, wy]
        g        2d governor state [gx, gy]
        """
        Ao = self.Ao
        z, w  = x[0:2], x[2:]
        p, v = self.s[:,0:2,:], self.s[:,2:,:]
        dgz = np.linalg.norm(g-z)
        R_mo = self.R_mo
        gamma = self.gamma

        aug_g = np.repeat(np.array([np.array([g]).T]), self.mo_n, axis=0)
        aug_z = np.repeat(np.array([np.array([z]).T]), self.mo_n, axis=0)
        aug_w = np.repeat(np.array([np.array([w]).T]), self.mo_n, axis=0)

        if dgz<1e-3:
            bo = 0.0
            delta_gz = 0.0
        else:
            bo = 1/(R_mo*dgz)
            delta_gz = (2*dgz)/R_mo
        aug_dgz = aug_g - aug_z
        pdpg = bo*aug_dgz
        pdpz = -pdpg

        phopg1 = 2*Ao@(aug_g-p) + 2*Ao@(aug_z-aug_g) - pdpg
        phopz1 = 2*Ao@(aug_g-aug_z) - pdpz
        phopp1 = -2*Ao@(aug_g-p)

        dgz_norm = np.linalg.norm(aug_g-aug_z, axis=1)
        aug_dgz_norm = np.zeros((self.mo_n, 1, 1))
        aug_dgz_norm[:,:,0] = dgz_norm

        dgp_norm = np.linalg.norm(aug_g-p, axis=1)
        aug_dgp_norm = np.zeros((self.mo_n, 1, 1))
        aug_dgp_norm[:,:,0] = dgp_norm

        dgz_vec = (aug_g-aug_z)
        dgp_vec = (aug_g-p)

        phopp2 = -dgp_vec/aug_dgp_norm
        if dgz<1e-5:
            phopz2 = np.zeros((self.mo_n, 2, 1))
        else:
            phopz2 = dgz_vec/aug_dgz_norm
        phopg2 = -phopp2 - phopz2

        hgo1 = np.transpose(aug_g-p,(0,2,1))@Ao@(aug_g-p)-np.transpose(aug_g-aug_z,(0,2,1))@Ao@(aug_g-aug_z)-1-delta_gz
        hgo2 = aug_dgp_norm - aug_dgz_norm - R_mo
        hgo = beta2*hgo2 + beta1*hgo1
        
        minh_ind = np.argmin(hgo[:,0,0])
        min_hgo = hgo[minh_ind,0,0]

        hgo_prime_f1 = phopz1[minh_ind].T@aug_w[minh_ind] + phopp1[minh_ind].T@v[minh_ind]
        hgo_prime_g1 = phopg1[minh_ind].T
        hgo_prime_g2 = phopg2[minh_ind].T
        hgo_prime_f2 = phopz2[minh_ind].T@aug_w[minh_ind] + phopp2[minh_ind].T@v[minh_ind]

        self.hgo_prime_f = beta1*hgo_prime_f1[0,0]+beta2*hgo_prime_f2[0,0]
        self.hgo_prime_g = beta1*hgo_prime_g1[0]+beta2*hgo_prime_g2[0]

        self.classk_func_g = gamma*(min_hgo**2)

        g_safe = min_hgo>=0

        self.hgo_log = np.hstack((self.hgo_log, min_hgo))

        self.g = g
        self.z = z
        self.p = p
        return g_safe

    def setup_osqp_init(self, input_bound=False):
        """
        Initially setup OSQP object.
        """
        self.input_bound = input_bound

        osqp_dic_init = {}
        osqp_dic_init['P'] = csc_matrix(np.eye(2))
        osqp_dic_init['q'] = np.ones((2,1))
        if input_bound:
            osqp_dic_init['A'] = csc_matrix(np.ones((3,2)))
            osqp_dic_init['m'] = np.ones((3,1))     
            osqp_dic_init['l'] = np.ones((3,1))
        else:
            osqp_dic_init['A'] = csc_matrix(np.ones((1,2)))
            osqp_dic_init['m'] = np.ones((1,1))
            osqp_dic_init['l'] = np.ones((1,1))   

        qp_object = osqp.OSQP()
        qp_object.setup(P=osqp_dic_init['P'], q=osqp_dic_init['q'], A=osqp_dic_init['A'], l=osqp_dic_init['l'], u=osqp_dic_init['m'])

        return qp_object
    
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

        # objective = cp.Minimize(cp.square(cp.norm(x-P@q))+n)
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
    
    def formulate_qp_inputs(self, u_nom, Px, u_range=None):
        """
        formulate inputs for OSQP solver
        OSQP solves convex quadratic programs (QPs) of the form
            minimize 1/2*x.T@P@x + q.T@x
            subject to l<=A@x<=m
        
        INPUT
        u_nom       nominal control input generated from pure pursuit controller [a, omega]
        Px           selected control cost p.d. matrix
        u_range     numpy array including input constraints [[a_min, omega_min], [a_max, omega_max]]
        """

        osqp_dic = {}
        osqp_dic['Px'] = Px
        osqp_dic['q'] = -np.array([u_nom]).T

        if self.GVN:
            l2 = -self.classk_func_g-self.hgo_prime_f
            m2 = np.inf
            A2 = self.hgo_prime_g
        else:
            l2 = -self.classK_func-self.ho_double_prime_f
            m2 = np.inf
            A2 = self.ho_double_prime_g

        if self.input_bound:     
            l1 = u_range[0,:]        
            m1 = u_range[1,:]
            A1 = np.eye(2)
            A = np.vstack((A1,A2))
            l = np.array([np.hstack((l1,l2))]).T
            m = np.array([np.hstack((m1,m2))]).T
        else:
            A = np.copy(A2)
            l = np.array([l2])
            m = np.array([m2])
        
        osqp_dic['A'] =  A
        osqp_dic['Ax'] =  A.flatten()
        osqp_dic['m'] = m
        osqp_dic['l'] = l

        return osqp_dic
    
    def update_cp_problem(self, u_nom, ug_prev, beta=0.5):
        """
        update parameters in CVXPY problem
        """

        self.q.value = u_nom

        # vmin = self.vmin[:,0]
        # v_norm = np.linalg.norm(vmin)
        # if v_norm<1e-5:
        #     self.q.value = u_nom
        #     self.n.value = 0
        # else:
        #     self.q.value = u_nom + (vmin*beta)/(2*v_norm)
        #     self.n.value = -beta*(u_nom@vmin)/v_norm-(beta**2)*v_norm/4

        # self.q.value = u_nom + ug_prev/2
        # self.n.value = -u_nom@ug_prev-(ug_prev@ug_prev)/4

        self.l.value = -self.classk_func_g-self.hgo_prime_f
        self.A.value = self.hgo_prime_g
        self.m.value = np.linalg.norm(u_nom)
        
        
        self.prob.solve(verbose=False)
        x_star = self.x.value
        if x_star is None:
            print(self.prob.status)
            x_star = np.zeros(2)

        cp_values = {
            'l': self.l.value,
            'm': self.m.value,
            'u': x_star,
            'A': self.A.value,
            'u_norm': np.linalg.norm(x_star)
        }

        return cp_values

    def plot_mo_set_init(self, ax):
        """
        initial prediction set plot of moving obstacle
        """
    
        self.mo_dic_list=[]
        awhead = create_arrowhead_patch()
        for mo_i in range(self.mo_n):

            # show moving obstacle trajectory and start point
            ax.plot([self.p_start[mo_i,0,0], self.p_goal[mo_i,0,0]], 
                         [self.p_start[mo_i,1,0], self.p_goal[mo_i,1,0]], 
                         color='gray', lw=4, dashes=[6, 2])
            ax.plot(self.p_start[mo_i,0,0], self.p_start[mo_i,1,0], 'o',
                         color='gray', ms=8)
            
            # circle patches as moving obstacle sets
            mo_dic={}
            circle_mo = patches.Circle(
                        (self.s[mo_i,0,0], self.s[mo_i,1,0]), self.r_mo[mo_i,0,0],
                        edgecolor='black',
                        facecolor='gray',
                        alpha=0.4,
                        fill=True)
            mo_dic['circl'] = circle_mo
            patch_mo = ax.add_patch(circle_mo)
            mo_dic['patch'] = patch_mo
            # annotate Time(s)
            ann_t = ax.annotate('Time: 0.00s', xy=(5,90), fontsize='xx-large')
            mo_dic['time'] = ann_t
            # show moving obstacle location and heading
            awhead_rot = awhead.transformed(mpl.transforms.Affine2D().rotate(self.mo_heading[mo_i]))
            loc_mo, = ax.plot(self.s[mo_i,0,0], self.s[mo_i,1,0], marker=awhead_rot, ms=15, color='black')
            mo_dic['loc'] = loc_mo

            self.mo_dic_list.append(mo_dic)

            # annotate moving obstacle velocity and radius
            ax.annotate(r'$v_{%s}=$'%(mo_i+1)+str(np.round(np.linalg.norm(self.s[mo_i,2:,0]),2))+'m/s', 
                        xy=(self.p_start[mo_i,0,0]+2,self.p_start[mo_i,1,0]-2), fontsize='large')
            ax.annotate(r'$r_{%s}=$'%(mo_i+1)+str(self.r_mo[mo_i,0,0])+'m', 
                        xy=(self.p_start[mo_i,0,0]+2,self.p_start[mo_i,1,0]-5), fontsize='large')
        
        self.ax = ax
        self.awhead = awhead
        pass

    def plot_mo_set_update(self):
        """
        update prediction set plot of moving obstacle
        """
        for mo_j in range(self.mo_n):
            mo_dic = self.mo_dic_list[mo_j]
            mo_dic['circl'].center = (self.s[mo_j,0,0], self.s[mo_j,1,0])
            awhead_rot = self.awhead.transformed(mpl.transforms.Affine2D().rotate(self.mo_heading[mo_j]))
            mo_dic['loc'].set_data(self.s[mo_j,0], self.s[mo_j,1])
            mo_dic['loc'].set_marker(awhead_rot)
            mo_dic['time'].set_text('Time: '+str(self.t)+'s')
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

    # moving obstacle list [p_startx, p_starty, p_goalx, p_goaly, kv_mo, r_mo]
    # kv_mo: moving obstacle speed gain; r_mo: moving obstacle prediction set radius
    mo_list = np.array([[55, 70, 20, 70, 0.05, 2],
                        [70, 10, 70, 90, 0.06, 2],
                        [75, 10, 75, 90, 0.07, 3],
                        [60, 15, 60, 75, 0.01, 5],
                        [65, 20, 65, 95, 0.02, 2],
                        [50, 30, 50, 60, 0.08, 5],
                        [45, 20, 45, 90, 0.04, 3],
                        [30, 90, 30, 20, 0.06, 4]])
    
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

    MOCheck = MovingObstacleCheck(mo_list, mo_param_dic, GVN=True)
    s = MOCheck.get_mo_state(0)
    MOCheck.plot_mo_set_init(ax_sim)

    # Start driving
    while ti< t_max:
        # t = round(ti*dt,2)
        s = MOCheck.get_mo_state(ti*dt)   #((ti-t_restart)*dt)
        # plot moving obstacle
        MOCheck.plot_mo_set_update()
        ti +=1
        plt.pause(0.0001)
    plt.show()

