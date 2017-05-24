#!/usr/bin/env python
# module EXPERIMENTS
import numpy as np
import matplotlib.pyplot as plt
import plots_cti as plots


def read_experimental_data(path, name, runs, tag_ids):
    from plots_cti import create_multispan_plots
    # Independent of dataset
    anchors_full = np.genfromtxt(path + 'anchors.csv', delimiter=',')
    real = np.genfromtxt(path + 'real_position.csv', delimiter=',')
    n_positions = 5
    range_idx = 6
    locator_idx = 2
    point_idx = 1

    anchors = anchors_full[:, 1:]
    N = anchors.shape[0] + 1
    distances_all_points = [np.zeros((N, N)) for n in range(n_positions)]
    weights_all_points = [np.zeros((N, N)) for n in range(n_positions)]
    counter_all_points = [np.zeros((N, N)) for n in range(n_positions)]
    for i in range(n_positions):
        P_i = real[i, :]
        fig, ax, ax_total = create_multispan_plots(tag_ids)
        # Get samples corresponding to all pairs.
        for j, a in enumerate(anchors_full):
            distances_per_point = []
            P_a = a[1:]
            id_a = int(a[0])
            distance_real = np.linalg.norm(P_a - P_i)

            # Loop through different methods
            for count, id_i in enumerate(tag_ids):
                distances_ia = []
                distances_total = 0
                counter_total = 0
                for run in runs:
                    data = np.genfromtxt(
                        path + name.format(i, run), delimiter=',')
                    idx_a = data[:, locator_idx] == id_a
                    idx_i = data[:, point_idx] == id_i
                    for d in data[idx_a & idx_i, range_idx]:
                        distances_ia.append(d)
                        distances_total += d
                        counter_total += 1
                #print('adding distances from anchor {} to point {} at position {}'.format(id_a, ids, i))
                [distances_per_point.append(d) for d in distances_ia]
                if len(distances_ia) > 0:
                    vals, bins, __ = ax[count].hist(
                        distances_ia, alpha=0.2, label='anchor {}'.format(id_a), color=plots.COLORS[j])
                    ax[count].vlines(distance_real, 0, max(
                        vals), lw=5, color=plots.COLORS[j], linestyle=':')

                distances_all_points[i][j, -1] += distances_total
                distances_all_points[i][-1, j] += distances_total
                counter_all_points[i][j, -1] += counter_total
                counter_all_points[i][-1, j] += counter_total
                ax[count].legend(loc='best')
                ax[count].set_title('Tag with Id {}'.format(id_i))
            if len(distances_per_point) > 0:
                ax_total.vlines(distance_real, 0, max(vals),
                                lw=5, color=plots.COLORS[j], linestyle=':')
                vals, bins, __ = ax_total.hist(
                    distances_per_point, alpha=0.2, label='anchor {}'.format(id_a), color=plots.COLORS[j])
        fig.suptitle('Position {}'.format(i), fontsize=30)
        plt.show()
    return distances_all_points, weights_all_points, counter_all_points

if __name__ == "__main__":
    print('nothing happens.')
