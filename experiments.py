#!/usr/bin/env python
# module EXPERIMENTS
import numpy as np
import matplotlib.pyplot as plt
import plots_cti as plots
from basics import divide_where_nonzero

N_POSITIONS = 2

def read_experimental_data(path, name, runs, tag_ids, weighted=False):
    anchors_full = np.genfromtxt(path + 'anchors.csv', delimiter=',')
    real = np.genfromtxt(path + 'real_position.csv', delimiter=',')
    tag_idx = 1
    locator_idx = 2
    range_idx = 6
    if (weighted):
        abs_error_idx = 7
        var_idx = 8
    else:
        abs_error_idx = 0
        var_idx = 0

    anchors = anchors_full[:, 1:]
    N = anchors.shape[0] + 1
    d = anchors.shape[1]
    results = [ExperimentResults(n,N,d) for n in range(N_POSITIONS)]

    for i,result in enumerate(results):
        result.real = real[i, :]
        fig, ax, ax_total = plots.create_multispan_plots(tag_ids)

        # Get samples corresponding to all pairs.
        for j, a in enumerate(anchors_full):
            distances_per_point = []
            P_a = a[1:]
            id_a = int(a[0])
            distance_real = np.linalg.norm(P_a - result.real)

            # Loop through different methods
            for count, id_i in enumerate(tag_ids):
                distances_per_tag = []
                distances_total = 0
                abs_error_total = 0
                var_total = 0
                counter_total = 0
                for run in runs:
                    data = np.genfromtxt(
                        path + name.format(i, run), delimiter=',')
                    idx_a = data[:, locator_idx] == id_a
                    idx_i = data[:, tag_idx] == id_i
                    for this_data in data[idx_a & idx_i,:]:
                        d = this_data[range_idx]
                        var = this_data[var_idx]
                        abs_error = this_data[abs_error_idx]
                        distances_per_point.append(d)
                        distances_per_tag.append(d)
                        distances_total += d
                        var_total += var
                        abs_error_total += abs_error
                        counter_total += 1
                

                if len(distances_per_tag) > 0:
                    vals, bins, __ = ax[count].hist(
                        distances_per_tag, alpha=0.2, label='anchor {}'.format(id_a), color=plots.COLORS[j])
                    ax[count].vlines(distance_real, 0, max(
                        vals), lw=5, color=plots.COLORS[j], linestyle=':')

                result.edm_avg[j, -1] += distances_total
                result.edm_avg[-1, j] += distances_total
                result.abs_error_avg[j, -1] += abs_error_total
                result.abs_error_avg[-1, j] += abs_error_total
                result.var_avg[j, -1] += var_total
                result.var_avg[-1, j] += var_total
                result.counter[j, -1] += counter_total
                result.counter[-1, j] += counter_total
                [distances_per_point.append(d) for d in distances_per_tag]
                ax[count].legend(loc='best')

            if len(distances_per_point) > 0:
                ax_total.vlines(distance_real, 0, max(vals),
                                lw=5, color=plots.COLORS[j], linestyle=':')
                vals, bins, __ = ax_total.hist(
                    distances_per_point, alpha=0.2, label='anchor {}'.format(id_a), color=plots.COLORS[j])
                result.distances[j].append(distances_per_point) 
                #distances_all_points[i][j].append(distances_per_point)
        ax_total.legend(loc='best')
        fig.suptitle('Position {}'.format(i), fontsize=30)
        plt.show()
        result.render()
    return results, anchors

class ExperimentResults:
    def __init__(self, n_position,N,d):
        self.n_position = n_position
        self.real = np.zeros((d,1))
        self.edm_avg = np.zeros((N,N))
        self.counter = np.zeros((N,N))
        self.abs_error_avg = np.zeros((N,N))
        self.var_avg = np.zeros((N,N))
        self.distances = [[] for i in range(N-1)]
    def render(self):
        self.edm_avg = divide_where_nonzero(self.edm_avg, self.counter)
        self.edm_avg = np.power(self.edm_avg, 2)
        self.abs_error_avg = divide_where_nonzero(self.abs_error_avg, self.counter)
        self.var_avg = divide_where_nonzero(self.var_avg, self.counter)

if __name__ == "__main__":
    print('nothing happens.')
