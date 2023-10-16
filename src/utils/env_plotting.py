"""
Plotting tools for Sampling-based algorithms
@author: huiming zhou
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# from rrt import env

class Env:
    def __init__(self, xrange, yrange, rec_obs=[], cir_obs=[]):
        self.x_range = xrange
        self.y_range = yrange
        self.rec_obs = rec_obs
        self.cir_obs = cir_obs
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
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

    def obs_circle(self):
        if len(self.cir_obs)>0:
            obs_cir = self.cir_obs
        else:
            obs_cir = []

        return obs_cir

class Plotting:
    def __init__(self, x_start, x_goal, ax, env):
        self.xI, self.xG = x_start, x_goal
        self.env = env
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.ax = ax

    def animation(self, nodelist, path, name, animation=False):
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)

    def animation_connect(self, V1, V2, path, name):
        self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)

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

        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        self.ax.plot(self.xI[0], self.xI[1], "*", color='indianred', alpha=1, markersize=15)
        self.ax.plot(self.xG[0], self.xG[1], "*", color='springgreen', alpha=1, markersize=15)

        plt.title(name)
        self.ax.set_ylim(0, self.env.y_range)
        self.ax.set_xlim(0, self.env.x_range)
        self.ax.set_aspect('equal',"box")

    @staticmethod
    def plot_visited(nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    @staticmethod
    def plot_visited_connect(V1, V2):
        len1, len2 = len(V1), len(V2)

        for k in range(max(len1, len2)):
            if k < len1:
                if V1[k].parent:
                    plt.plot([V1[k].x, V1[k].parent.x], [V1[k].y, V1[k].parent.y], "-g")
            if k < len2:
                if V2[k].parent:
                    plt.plot([V2[k].x, V2[k].parent.x], [V2[k].y, V2[k].parent.y], "-g")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 2 == 0:
                plt.pause(0.001)

        plt.pause(0.01)

    # @staticmethod
    def plot_path(self, path):
        if len(path) != 0:
            self.ax.plot([x[0] for x in path], [x[1] for x in path], 
                         '-x', markersize=6, color='darkgreen', lw=4, label='nav path')
            plt.pause(0.001)



if __name__ == '__main__':
    
    x_start = (5, 5)  # Starting node
    x_goal = (110, 95)  # Goal node
    PlotMap = Plotting(x_start, x_goal)
    PlotMap.plot_grid('MAP')
    plt.show()