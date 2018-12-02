import matplotlib.pyplot as plt
import numpy as np
from numpy import inf
from numpy.linalg import norm
import fileio

DEFAULT_FOLDER = "Results/"
DEFAULT_FILE_PREFIX = DEFAULT_FOLDER + "/Protected/results_problem_"

def histogram_norms(
    problem_num=5,
    output_filename="Images/Q5 x histograms",
    # input_filename="Results/Protected/results_problem_5.npz",
    # output_filename_prefix="Images/Q5_histogram_"
):
    A, b = fileio.load_A_b(problem_num)

    x_vals, _, _ = fileio.load_results(problem_num)
    xmin, xmax = min(
        min(A.dot(x_vals[0]) - b),
        min(A.dot(x_vals[1]) - b),
        min(A.dot(x_vals[2]) - b)
    ), max(
        max(A.dot(x_vals[0]) - b),
        max(A.dot(x_vals[1]) - b),
        max(A.dot(x_vals[2]) - b)
    )
    _, step = np.linspace(xmin, xmax, retstep=True)
    bins = np.linspace(xmin - step/3, xmax + step/3)
    plt.figure(figsize=[8, 6])
    plt.hist(A.dot(x_vals[1]) - b, bins, alpha=0.6)
    plt.hist(A.dot(x_vals[0]) - b, bins + step/3, alpha=0.6)
    plt.hist(A.dot(x_vals[2]) - b, bins - step/3, alpha=0.6)
    plt.xlim(-2, 2)
    plt.grid(True)
    plt.legend(["l_1", "l_2", "l_inf"])
    plt.xlabel("Residual component value")
    plt.ylabel("Frequency")
    plt.title("Histogram of the norm-approximation residuals for A5, b5")
    plt.savefig(output_filename)
    plt.close()

def plot_times(problem_list=range(1, 6), output_filename="Results/times"):
    times_list = [fileio.load_results(problem)[1] for problem in problem_list]
    n_list = [fileio.load_results(problem)[2] for problem in problem_list]
    # Plot l2 norm times
    plt.figure(figsize=[8, 6])
    plt.loglog(n_list, [t[0] for t in times_list], 'g', alpha=0.5)

    plt.grid(True)
    plt.xlabel("n")
    plt.ylabel("Time taken (s)")
    plt.title(
        "Computation times for different methods on different sized problems"
    )
    plt.savefig(output_filename)

if __name__ == "__main__":
    # plot_times()
    histogram_norms()



    # l1_table = [["n", "l1", "l2", "linf"]]
    # for i in range(len(problem_list)):
    #     l1_table.append([
    #         n_list[i], norm(x_vals_list[i][1], 1),
    #         norm(x_vals_list[i][1], 2), norm(x_vals_list[i][1], inf)
    #     ])
    # for row in l1_table: print(row)

    # l1_table, l2_table, linf_table = [[
    #     ["n", "l1", "l2", "linf"], *[[
    #         n_list[p],
    #         norm(x_vals_list[p][l], 1),
    #         norm(x_vals_list[p][l], 2),
    #         norm(x_vals_list[p][l], inf)
    #     ] for p in range(len(problem_list))]
    # ] for l in [1, 0, 2]]

    # for row in l1_table: print(row)
