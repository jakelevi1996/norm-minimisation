import numpy as np
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
    x_vals_list=None, x_vals_filename="Results/Protected/x_vals.npz",
    problem_list=range(1, 6), p=None
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

def print_times_tables(
    t_vals_list=None, t_vals_filename="Results/Protected/t_vals.npz",
    problem_list=range(1, 6)
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





if __name__ == "__main__":
    find_x_vals(problem_list=range(1, 3), save_results=False)
    # find_x_vals()
    find_t_vals(problem_list=range(1, 2), save_results=False, num_attempts=1)
    # find_t_vals()
    print_residuals_tables()
    print_times_tables()

