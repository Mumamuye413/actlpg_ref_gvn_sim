import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path

def _get_theta_from_two_pts(ptA, ptB, debug=False):
    dy = ptB[1] - ptA[1]
    dx = ptB[0] - ptA[0]
    theta = np.arctan2(dy, dx)
    if debug:
        print(ptA)
        print(ptB)
        print("[dy, dx, theta] = [%.2f, %.2f, %.2f]" % (dy, dx, theta))
    return theta

def wrap_angle(angle):
    """
    remap angle in radian from [0, 2*pi) to [-pi, pi)
    angle: angle despcription in radian
    """
    #   remap angle in radian from [-2*pi, 0) to [-pi, pi)
    if angle<0:
        angle_w = (angle - np.pi) % (-2 * np.pi) + np.pi
    else:
        angle_w = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle_w

def bf2wf2D(bf_pose, bf_pt):
    """
    input: body frame origin pose in world frame (1d array [x, y, theta])
           position in body frame need to be converted to world frame coords (1d array [xb, yb])
    
    Output: position in world frame (1d array [xw, yw])
    """
    theta = bf_pose[2]
    wRb = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    wf_pt = bf_pt@(wRb.T)
    return wf_pt

def wf2bf2D(bf_pose, wf_pt):
    """
    input: bf_pose - body frame origin pose in world frame (1d array [x, y, theta])
           wf_pt - position in world frame need to be converted to body frame coords (1d array [xw, yw])
    
    Output: position in body frame (1d array [xw, yw])
    """
    theta = bf_pose[2]
    wRb = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    bf_pt = (wRb.T)@(wf_pt-bf_pose[0:2])
    return bf_pt

def get_iccone_params(z, z_g, tol_len, tol_ang):
    """
    Compute radius of ice cream ball from current robot pose -z- and goal robot pose -zg-;
    generate coordinates of two tangent 2d-points for ice cream cone in robot __Body Frame__.

    Input: z, zg - position & orientation: (1d array [x, y, phi])
           tol_len - when d(z-z_g)<tol_len, r=0, tangent points = z
           tol_ang - when abs(z_phi-arctan2(zg_x,zg_y))<tol_ang, r=0, tangent points = zg
    
    Output: [tangent_point1, tangent_point2, p, pg] - a list of 2d points coords needed
            for ice-cream cone set visualization (*using matplotlib)
            list elements - (1d array [x, y])    
    """

    p, p_g = z[0:2], z_g[0:2]
    theta = z[2]

    e = np.linalg.norm(p - p_g)
    d_theta = _get_theta_from_two_pts(p, p_g)   # position tracking vector orientation in wf
    e_theta = wrap_angle(d_theta - theta)   # position tracking heading error (relative angel)

    # ---------- special cases ----------
    # goal too close
    if e<tol_len:
        e = 0.0
    
    # current heading aline with goal position
    elif abs(e_theta)<tol_ang:
        e_theta = 0.0
    
    # current heading not pointing to goal located half plane 
    if abs(e_theta)>np.pi/2:
        e_theta = np.pi/2

    # ---------- compute radius and coords -----------
    icball_radius = abs(e*np.sin(e_theta))

    # compute coords for 2 tangent points (tanpt1, tanpt2)
    tanpt1_x = round(e*np.cos(e_theta),5)
    tanpt2_x = round(tanpt1_x*np.cos(2*e_theta),5)
    tanpt2_y = round(tanpt1_x*np.sin(2*e_theta),5)

    tpt2d1 = np.array([tanpt1_x, 0.0])
    tpt2d2 = np.array([tanpt2_x, tanpt2_y])

    # convert tangent points coords from bf to wf
    tpt1wf = bf2wf2D(z, tpt2d1)+p
    tpt2wf = bf2wf2D(z, tpt2d2)+p

    return [tpt1wf, tpt2wf, p, p_g], icball_radius

def ball_path_intersection(x, r, path):
    """
    Find furthest intersection between a ball and a path in 2D/3D. (d=2/3)
    Inputs:
        x: coordinates of the ball center (d, )
        r: radius of the ball
        path: a series of waypoints (num_pts, d)
    Output
        is_intsec: boolen of if found intersection

        x_star  : furthest intersection (d, )
        B_idx: the furthest index of path xg lies in
                Example: B_idx = 6
                 (0)               (5)                (6)
                 START---->....---> A-----x_star-----> B----> .... ---> END

    Ref: personal notes
    """
    # default return
    is_intsec = False
    x_star, B_idx = [], []
    if len(path) < 2:
        # raise FindIntrError("path len is too short, >=2 waypoints is required!")
        pass
    # loop backward
    for path_idx in range(len(path) - 1):

        B_idx = -path_idx - 1
        B = path[B_idx]
        A = path[B_idx - 1]
        
        dAB = np.linalg.norm(A - B)  # segment length ptA from ptB
        u = x - A
        v = B - A
        w1 = np.inner(u, v) / dAB ** 2
        w1hat = max(min(w1, 1), 0)  # normalized to [0,1]
        dist2segAB = np.linalg.norm(u - w1hat * v)

        if dist2segAB > r:
            continue
        else:
            # distance from x to line AB
            dist2lineAB = np.linalg.norm(u - w1 * v)
            # print('DEBUG dist to line AB is %.2f' %(dist2lineAB))
            w2 = np.sqrt(r ** 2 - dist2lineAB ** 2) / dAB  # ratio of |v|
            w = w1 + w2
            w = max(min(w, 1), 0)
            x_star = (1 - w) * A + w * B
            is_intsec = True
            break

    return is_intsec, x_star, B_idx

def dist_point2obs(point, obs):
    """
    INPUT
    point       2d position [x, y]
    obs         rectangle obstacle feature [position_x, position_y, length_x, length_y]
                position attach point: left bottom corner (closest to map origin)
    OUTPUT
    dist        shortest distance between point and obstacle edges
    """
    obs_px, obs_py, obs_lx, obs_ly = obs
    point_x, point_y = point

    obs_xline = np.arange(obs_px, obs_px+obs_lx)
    obs_yline = np.arange(obs_py, obs_py+obs_ly)

    if point_x>obs_xline[0] and point_x<obs_xline[-1]:
        dist = min(abs(point_y-obs_py), abs(point_y-obs_py-obs_ly))
    elif point_y>obs_yline[0] and point_y<obs_yline[-1]:
        dist = min(abs(point_x-obs_px), abs(point_x-obs_px-obs_lx))
    else:
        closest_x = np.argmin(np.abs(obs_xline-point_x))
        closest_y = np.argmin(np.abs(obs_yline-point_y))
        closest_p = np.array([obs_xline[closest_x], obs_yline[closest_y]])
        dist = np.linalg.norm(point-closest_p)
    return dist