#!/usr/bin/env python
# module PLOTS

import matplotlib.pyplot as plt
import numpy as np

LINESTYLES = ['-', '-', ':', '--','-.','-.']
MARKERS = [".", "o", "v", ",", "^", ">", "1",
           "2", "3", "4", "8", "s", "p", "*", "h"]
COLORS = ["black", "blue", "fuchsia", "gray", "aqua", "green", "lime",
          "maroon", "navy", "olive", "purple", "red", "silver", "teal", "yellow"]

def plot_point_sets(point_sets, title='', size=[10, 10], filename='', names=None):
    import itertools
    if names is None:
        names = ['Original']
        for i in range(len(point_sets)):
            names.append('Method {}'.format(i + 1))
    f = plt.figure()
    ax = f.add_subplot(111)
    plt.gca().set_aspect('equal', adjustable='box')
    # Set range automatically
    delta = 1.0
    xmin = np.min(point_sets[0][:, 0]) - delta
    ymin = np.min(point_sets[0][:, 1]) - delta
    xmax = np.max(point_sets[0][:, 0]) + delta
    ymax = np.max(point_sets[0][:, 1]) + delta
    plt.axis((xmin, xmax, ymin, ymax))

    legend = []
    for p, points in enumerate(point_sets):
        N = points.shape[0]
        if p == 0:
            for pair in itertools.combinations(range(N), 2):
                plt.plot([points[pair[0], 0], points[pair[1], 0]], [points[pair[0], 1],
                                                                    points[pair[1], 1]], '-', color=COLORS[p + 1], linewidth=2.0)
            # Plot base line.
            plt.plot([points[0, 0], points[1, 0]], [points[0, 1],
                                                    points[1, 1]], color=COLORS[p], linewidth=2.0, linestyle=LINESTYLES[p])
            # Plot with label.
            plt.plot([points[0, 0], points[2, 0]], [points[0, 1],
                                                    points[2, 1]], color=COLORS[p + 1], linewidth=2.0, linestyle=LINESTYLES[p], label=names[p])
            # Plot point numbers.
            for i in range(N):
                ax.annotate('%s' % i, xy=(
                    points[i, 0], points[i, 1]), textcoords='data', size=20, weight='bold')
        else:
            for pair in itertools.combinations(range(N), 2):
                plt.plot([points[pair[0], 0], points[pair[1], 0]], [points[pair[0], 1],

                                                                    points[pair[1], 1]], linestyle=LINESTYLES[p + 2], color=COLORS[p + 2], linewidth=2.0)
            # Plot with label.
            plt.plot([points[0, 0], points[1, 0]], [points[0, 1], points[
                     1, 1]], linestyle=LINESTYLES[p + 2], color=COLORS[p + 2], linewidth=2.0, label=names[p])
    f.set_size_inches(size)
    if title == '':
        plt.title('N = %r' % N)
    else:
        plt.title(title)
    if filename != '':
        plt.savefig(filename)
    plt.legend(loc='best')
    plt.show()

if __name__=="__main__":
    print('nothing happens when running this module.')
