import matplotlib.pyplot as plt
from numpy import load, inf
from numpy.linalg import norm
import fileio

DEFAULT_FOLDER = "Results/"
DEFAULT_FILE_PREFIX = DEFAULT_FOLDER + "/Protected/results_problem_"

def histogram_norms(
    input_filename="Results/Protected/results_problem_5.npz",
    output_filename_prefix="Results/Q5_histogram_"
):
    A, b = fileio.load_A_b(5)

    # TODO: put this in fileio.py:
    with load(input_filename) as data:
        x_vals = data["x_vals"]
    # TODO: add labels, titles and subplots
    plt.figure(figsize=[8, 6])
    plt.hist(A.dot(x_vals[0]) - b, 20)
    plt.grid(True)
    plt.savefig(output_filename_prefix + "l2")
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

def eval_tables(problem_list=range(1, 6)):
    # Load A and b matrices for each problem
    ab_list = [fileio.load_A_b(problem) for problem in problem_list]
    # Load results for each problem
    results_list = [fileio.load_results(problem) for problem in problem_list]
    
    norm_headers = ["n", "l1", "l2", "linf"]
    solution_norms = []
    l1_solution_norms = []
    l2_solution_norms = []
    linf_solution_norms = []
    time_headers = [
        "n", "l1 (IP)", "l1 (simplex)", "l2 (LS)",
        "linf (IP)", "linf (simplex)"
    ]
    time_table_vals = []

    for (A, b), (x_vals, t_vals, n) in zip(ab_list, results_list):
        solution_norms.append([
            n, norm(A.dot(x_vals[1]) - b, 1),
            norm(A.dot(x_vals[0]) - b, 2), norm(A.dot(x_vals[2]) - b, inf),
        ])
        l1_solution_norms.append([
            n, norm(A.dot(x_vals[1]) - b, 1),
            norm(A.dot(x_vals[1]) - b, 2), norm(A.dot(x_vals[1]) - b, inf),
        ])
        l2_solution_norms.append([
            n, norm(A.dot(x_vals[0]) - b, 1),
            norm(A.dot(x_vals[0]) - b, 2), norm(A.dot(x_vals[0]) - b, inf),
        ])
        linf_solution_norms.append([
            n, norm(A.dot(x_vals[2]) - b, 1),
            norm(A.dot(x_vals[2]) - b, 2), norm(A.dot(x_vals[2]) - b, inf),
        ])
        mean_times = t_vals.mean(axis=1)
        if mean_times.size == 5:
            time_table_vals.append([
                n, mean_times[1], mean_times[3],
                mean_times[0], mean_times[2], mean_times[4],
            ])
        else:
            time_table_vals.append([
                n, mean_times[1],  mean_times[0], mean_times[2]
            ])
            

    return (
        norm_headers, solution_norms, l1_solution_norms, l2_solution_norms,
        linf_solution_norms, time_headers, time_table_vals
    )

def print_norm_table(headers, values):
    print("{:<6}{:<8}{:<8}{:<8}".format(*headers))
    for row in values:
        print("{:<6}{:<8.4}{:<8.4}{:<8.4}".format(*row))

def print_time_table(headers, values):
    print("{:<6}{:<12}{:<12}{:<12}{:<12}{:<12}".format(*headers))
    for row in values:
        if len(row) == 6:
            print("{:<6}{:<12.4}{:<12.4}{:<12.4}{:<12.4}{:<12.4}".format(*row))
        else:
            print("{0:<6}{1:<12.4}{4:<12}{2:<12.4}{3:<12.4}{4:<12}".format(
                *row, "-"
            ))

if __name__ == "__main__":
    headers, solution_norms = eval_tables()[:2]
    print("Solution norms:")
    print_norm_table(headers, solution_norms)
    headers, times = eval_tables()[-2:]
    print("Times taken:")
    print_time_table(headers, times)
    print("n:")
    plot_times()
    # histogram_norms()



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
