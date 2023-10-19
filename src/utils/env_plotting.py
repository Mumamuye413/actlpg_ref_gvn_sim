
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Env:
    """
    Set up map features including boundaries and static obstacles.
    """

    def __init__(self, xrange, yrange, rec_obs=[]):
        """
        INPUT
        xrange      map size along x-axis
        yrange      map size along y-axis
        rec_obs     lists of rectangular obstacle features [pivot_x, pivot_y, length_x, length_y]
                    (pivot point: bottom-left cornor)
        """
        self.x_range = xrange
        self.y_range = yrange
        self.rec_obs = rec_obs
        self.obs_boundary = self.obs_boundary()
        self.obs_rectangle = self.obs_rectangle()

    def obs_boundary(self):
        x_bound = self.x_range-1
        y_bound = self.y_range-1
        obs_boundary = [
            [0, 0, 1, y_bound],
            [0, y_bound, x_bound, 1],
            [1, 0, x_bound, 1],
            [x_bound, 1, 1, y_bound]
        ]
        return obs_boundary

    def obs_rectangle(self):
        if len(self.rec_obs)>0:
            obs_rectangle = self.rec_obs
        else:
            obs_rectangle=[]
        return obs_rectangle

class Plotting:
    """
    Plotting tools for map & path visualization.
    """

    def __init__(self, x_start, x_goal, ax, env):
        """
        INPUT
        x_start     robot 2d start position
        x_goal      robot 2d goal position
        ax          figure ax
        env         environment object formed with class Env
        """
        self.xI, self.xG = x_start, x_goal
        self.env = env
        self.obs_bound = self.env.obs_boundary
        self.obs_rectangle = self.env.obs_rectangle
        self.ax = ax

    def plot_grid(self, name=None, clean=False):
        
        if clean:
            plt.cla()
        
        for (ox, oy, w, h) in self.obs_bound:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='dimgray',
                    facecolor='dimgray',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='dimgray',
                    facecolor='dimgray',
                    fill=True
                )
            )

        self.ax.plot(self.xI[0], self.xI[1], "*", color='indianred', alpha=1, markersize=15)
        self.ax.plot(self.xG[0], self.xG[1], "*", color='springgreen', alpha=1, markersize=15)

        plt.title(name)
        self.ax.set_ylim(0, self.env.y_range)
        self.ax.set_xlim(0, self.env.x_range)
        self.ax.set_aspect('equal',"box")

    def plot_path(self, path):
        if len(path) != 0:
            self.ax.plot([x[0] for x in path], [x[1] for x in path], 
                         '-x', markersize=6, color='darkgreen', lw=4, label='nav path')
            plt.pause(0.001)
