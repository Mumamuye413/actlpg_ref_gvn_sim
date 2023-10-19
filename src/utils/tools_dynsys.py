import numpy as np


def unicycle_polar(t, xvec, v, w):
        """ 
        robot dynamics unicyle in polar system
        """
        eps_dist = 1e-8
        e, e_phi, d_phi = xvec
        if np.abs(e) < eps_dist:
            print("edist is %e, set dzdt = 0" % np.abs(e))
            return np.zeros(3)
        else:
            e_dot = -v * np.cos(e_phi)
            e_phi_dot = -w + v * np.sin(e_phi)/e
            d_phi_dot = v * np.sin(e_phi)/e
            x_dot = np.array([e_dot, e_phi_dot, d_phi_dot])
            return x_dot

def unicycle_polar_pcw(t, xvec, v, w):
    """ 
    robot dynamics unicyle in polar system (heading error not converge for position tracking)
    """
    eps_dist = 1e-8
    e, e_phi, z_phi = xvec
    if np.abs(e) < eps_dist:
        print("edist is %e, set dzdt = 0" % np.abs(e))
        return np.zeros(3)
    else:
        e_dot = -v * np.cos(e_phi)
        e_phi_dot = -w + v * np.sin(e_phi)/e
        z_phi_dot = w
        x_dot = np.array([e_dot, e_phi_dot, z_phi_dot])
        return x_dot

def unicycle_carts(t, zvec, v, w):
    """ 
    robot dynamics unicyle in Cartesian system
    """

    _, _, theta = zvec

    z_dot = [v * np.cos(theta), v * np.sin(theta), w]

    return z_dot

# function value evaluation
def feval(funcName, *args):
    return np.array(funcName(*args))


def rk4_update(h, func, tn, yn, un, *extra_func_args):
    """
    RK4 takes one Runge-Kutta step.
    This function is used to solve the forced initial value ode of the form
    dy/dt = f(t, y, uvec),  with y(t0) = yn
    User need to supply current values of t, y, a stepsize h, external input
    vector, and  dynamics of ackerman drive to evaluate the derivative,
    this function can compute the fourth-order Runge Kutta estimate
    to the solution at time t+h.

    Parameters:
    INPUT:
        func,   function handle     RHS of ODE equation,
                                    specifically ackerman drive dynamics

        tn,     scalar              the current time.
        yn,     1D array            y value at time tn
        un,     1D array            external input at time tn
        h,      scalar              step size, scalar


    OUTPUT:
        yn1,    1D array            the 4_th order Runge-Kutta estimate
                                    y value at next time step
    """
    k1 = feval(func, tn, yn, un, *extra_func_args)
    k2 = feval(func, tn + h/2.0, yn + k1 * h/2.0, un, *extra_func_args)
    k3 = feval(func, tn + h/2.0, yn + k2 * h/2.0, un, *extra_func_args)
    k4 = feval(func, tn + h,     yn + k3 * h,     un, *extra_func_args)

    # Estimate y at time t_n+1 using weight sum of 4 key sample pts
    yn1 = yn + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return yn1
