import numpy as np
from time import time
import results, fileio, linopt as lo

def display_time_taken(time_taken):
    if time_taken > 60:
        m, s = divmod(time_taken, 60)
        print("All problems analysed in {}m {}s".format(m, s))
    else: print("All problems analysed in {:.4g}s".format(time_taken))

def find_x_vals(
    problem_list=range(1, 6), output_filename="Results/x_vals.npz",
    verbose=True, very_verbose=True
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
    if verbose: display_time_taken(time_taken)

def analyse_methods(
    problem_list=range(1, 6), num_attempts=3, max_simplex_n=256,
    folder=results.DEFAULT_FOLDER, filename_prefix="results_problem_"
):
    """Analyse methods of norm-minimisation for different methods.

    One `.npz` results file is saved for each problem, containing 2
    `np.ndarray`s, 1 named `x_vals` and 1 named `t_vals`.

    `x_vals` contains 3 rows, which respectively contain the solution-vector
    found by minimising:
     - The l2 norm
     - The l1 norm
     - The linfinity norm
    
    `t_vals` contains either 3 or 5 rows, which respectively contain the times
    taken for differet attempts at minimisation using:
     - Least squares and the l2 norm
     - Interior point methods and the l1 norm
     - Interior point methods and the linfinity norm
     - [Simplex and the l1 norm]
     - [Simplex and the linfinity norm]
    """
    t_start = time()
    for problem in problem_list:
        A, b = fileio.load_A_b(problem)
        n = A.shape[1]
        x_vals = np.empty([3, n])
        if n <= max_simplex_n: t_vals = np.empty([5, num_attempts])
        else: t_vals = np.empty([3, num_attempts])
        for attempt in range(num_attempts):
            # Calculate solutions and time taken for interior-point and LS
            x_vals[0], t_vals[0, attempt], _ = lo.l2_min(A, b)
            x_vals[1], t_vals[1, attempt], _ = lo.l1_min(
                A, b, method='interior-point'
            )
            x_vals[2], t_vals[2, attempt], _ = lo.linf_min(
                A, b, method='interior-point'
            )
            # For small problems, calculate time taken for the simplex method
            if n <= max_simplex_n:
                _, t_vals[3, attempt], _ = lo.l1_min(
                    A, b, method='simplex'
                )
                _, t_vals[4, attempt], _ = lo.linf_min(
                    A, b, method='simplex'
                )
        # Save results for each problem in a `.npz` file
        np.savez(
            folder + filename_prefix + str(problem),
            x_vals=x_vals, t_vals=t_vals
        )
    time_taken = time() - t_start
    if time_taken > 60:
        m, s = divmod(time_taken, 60)
        print("All problems analysed in {}m {}s".format(m, s))
    else: print("All problems analysed in {:.4g}s".format(time_taken))