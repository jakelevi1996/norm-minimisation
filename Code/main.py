import numpy as np
import matplotlib.pyplot as plt
from time import time
import results, fileio, linopt as lo

def display_time_taken(time_taken):
    if time_taken > 60:
        m, s = divmod(time_taken, 60)
        print("All problems analysed in {}m {:.3}s".format(int(m), s))
    else: print("All problems analysed in {:.4}s".format(time_taken))

def find_x_vals(
    problem_list=range(1, 6), output_filename="Results/x_vals.npz",
    save_results=True, verbose=True, very_verbose=True
):
    t_start = time()
    x_vals_list = []
    for problem in problem_list:
        A, b = fileio.load_A_b(problem)
        n = A.shape[1]
        x_vals = np.empty([3, n])
        x_vals[0], _, _ = lo.l1_min(
            A, b, method='interior-point', verbose=very_verbose
        )
        x_vals[1], _, _ = lo.l2_min(A, b, verbose=very_verbose)
        x_vals[2], _, _ = lo.linf_min(
            A, b, method='interior-point', verbose=very_verbose
        )
        x_vals_list.append(x_vals)
    
    time_taken = time() - t_start
    if save_results: fileio.save_vals_list(x_vals_list, output_filename)
    if verbose: display_time_taken(time_taken)

def find_t_vals(
    problem_list=range(1, 6), output_filename="Results/t_vals.npz",
    max_n_simplex=256, num_attempts=3,
    save_results=True, verbose=True, very_verbose=True
):
    t_start = time()
    t_vals_list = []
    for problem in problem_list:
        A, b = fileio.load_A_b(problem)
        n = A.shape[1]
        t_vals = np.zeros([5, num_attempts])
        for attempt in range(num_attempts):
            _, _, t_vals[0, attempt] = lo.l1_min(
                A, b, method='interior-point', verbose=very_verbose
            )
            if n <= max_n_simplex:
                _, _, t_vals[1, attempt] = lo.l1_min(
                    A, b, method='simplex', verbose=very_verbose
                )
            _, _, t_vals[2, attempt] = lo.l2_min(A, b, verbose=very_verbose)
            _, _, t_vals[3, attempt] = lo.linf_min(
                A, b, method='interior-point', verbose=very_verbose
            )
            if n <= max_n_simplex:
                _, _, t_vals[4, attempt] = lo.linf_min(
                    A, b, method='simplex', verbose=very_verbose
                )
        t_vals_list.append(t_vals)
    
    time_taken = time() - t_start
    if save_results: fileio.save_vals_list(t_vals_list, output_filename)
    if verbose: display_time_taken(time_taken)

def print_residuals_tables(
    x_vals_filename="Results/Protected/x_vals.npz",
    x_vals_list=None, problem_list=range(1, 6), p=None
):
    n_problems = len(problem_list)
    if x_vals_list is None:
        x_vals_list = fileio.load_vals_list(x_vals_filename)
    assert len(x_vals_list) == n_problems

    headers = ["n", "l1", "l2", "linf"]
    print("{:<6}{:<8}{:<8}{:<8}".format(*headers))
    for i in range(n_problems):
        x_vals = x_vals_list[i]
        A, b = fileio.load_A_b(problem_list[i])
        n = A.shape[1]
        if p is None:
            l1res = lo.residual(A, x_vals[0], b, 1)
            l2res = lo.residual(A, x_vals[1], b, 2)
            linfres = lo.residual(A, x_vals[2], b, np.inf)
        else:
            l1res = lo.residual(A, x_vals[0], b, p)
            l2res = lo.residual(A, x_vals[1], b, p)
            linfres = lo.residual(A, x_vals[2], b, p)
        print("{:<6}{:<8.4}{:<8.4}{:<8.4}".format(n, l1res, l2res, linfres))

def print_t_tables(
    t_vals_filename="Results/Protected/t_vals.npz",
    t_vals_list=None, problem_list=range(1, 6)
):
    n_problems = len(problem_list)
    if t_vals_list is None:
        t_vals_list = fileio.load_vals_list(t_vals_filename)
    assert len(t_vals_list) == n_problems

    headers = [
        "n", "l1 (IP)", "l1 (simplex)",
        "l2 (LS)", "linf (IP)", "linf (simplex)"
    ]
    print(("\n{:<6}" + 5*"{:<13}").format(*headers))

    for i in range(n_problems):
        t_vals = t_vals_list[i]
        A, _ = fileio.load_A_b(problem_list[i])
        n = A.shape[1]
        t_means = t_vals.mean(axis=1)
        print(("{:<6}" + 5*"{:<13.4}").format(n, *t_means))

def plot_t_tables(
    t_vals_list=None, t_vals_filename="Results/Protected/t_vals.npz",
    problem_list=range(1, 6), output_filename="Images/Norm minimisation times"
):
    # TODO: evaluate n_list from problem_list
    n_list=np.array([16, 64, 256, 512, 1024])
    n_problems = len(problem_list)
    if t_vals_list is None:
        t_vals_list = fileio.load_vals_list(t_vals_filename)
    assert len(t_vals_list) == n_problems
    n_attempts = t_vals_list[0].shape[1]
    fmt_list = ["bo:", "ro:", "go-", "bo-", "ro-"]

    plt.figure(figsize=[8, 6])
    for attempt in range(n_attempts):
        for p, fmt in enumerate(fmt_list):
            t = np.array(
                [t_vals_list[n][p, attempt] for n in range(n_problems)]
            )
            plt.loglog(n_list[t > 1.2e-3], t[t > 1.2e-3], fmt, alpha=0.3)
    plt.xlabel("n")
    plt.ylabel("Computation time (s)")
    title = "Computation time against problem size "
    title += "for different norm-minimisation methods"
    plt.title(title)
    plt.legend([
        "l_1 (IP)", "l_1 (simplex)", "l_2 (LS)",
        "l_inf (IP)", "l_inf (simplex)"
    ])
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()

def residual_histograms(
    problem_num=5, output_filename="Images/Residual histograms",
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


if __name__ == "__main__":
    # find_x_vals(problem_list=range(1, 3), save_results=False)
    # find_x_vals()
    # find_t_vals(problem_list=range(1, 2), save_results=True, num_attempts=1)
    # find_t_vals()
    print_residuals_tables()
    [print_residuals_tables(p=p) for p in [1, 2, np.inf]]

    print_t_tables()
    plot_t_tables()
    residual_histograms()
