import numpy as np
from scipy.optimize import linprog
from time import time

import fileio, results

def l1_min(A, b, method='interior-point'):
    assert A.ndim == 2 and b.ndim == 1
    m, n = A.shape
    I = np.identity(m)
    A_lb = np.block([[-A, I], [A, I]])
    b_lb = np.block([-b, b])
    c = np.block([np.zeros(n), np.ones(m)])
    print(A_lb.shape, b_lb.shape, c.shape)
    start_time = time()
    res = linprog(
        c, A_ub=-A_lb, b_ub=-b_lb, bounds=(None, None),
        options={"maxiter": np.inf, "tol": 1e-7}, method=method
    )
    time_taken = time() - start_time
    print("{} Function value is {:.8g} after {} iterations in {:.4g} s".format(
        res.message, res.fun, res.nit, time_taken
    ))
    return res.x[:n], time_taken, res.status

def linf_min(A, b, method='interior-point'):
    assert A.ndim == 2 and b.ndim == 1
    m, n = A.shape
    ones = np.ones([m, m])
    A_lb = np.block([[-A, ones], [A, ones]])
    b_lb = np.block([-b, b])
    c = np.block([np.zeros(n), np.ones(m)])
    print(A_lb.shape, b_lb.shape, c.shape)
    start_time = time()
    res = linprog(
        c, A_ub=-A_lb, b_ub=-b_lb, bounds=(None, None),
        options={"maxiter": np.inf, "tol": 1e-7}, method=method
    )
    time_taken = time() - start_time
    print("{} Function value is {:.8g} after {} iterations in {:.4g} s".format(
        res.message, res.fun, res.nit, time_taken
    ))
    return res.x[:n], time_taken, res.success

def l2_min(A, b):
    assert A.ndim == 2 and b.ndim == 1
    n = A.shape[1]
    start_time = time()
    x, residual, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    time_taken = time() - start_time
    print("Residual is {:.8g}, rank is {}, time taken is {:.4g} s".format(
        *residual, rank, time_taken
    ))
    return x[:n], time_taken, rank

def analyse_methods(
    problem_list=range(1, 6), num_attempts=3, max_simplex_n=256,
    folder=results.DEFAULT_FOLDER, filename_prefix="results_problem_"
):
    start_time = time()
    for problem in problem_list:
        A, b = fileio.load_A_b(problem)
        n = A.shape[1]
        x_vals = np.empty([3, n])
        if n <= max_simplex_n: t_vals = np.empty([5, num_attempts])
        else: t_vals = np.empty([3, num_attempts])
        for attempt in range(num_attempts):
            # Calculate solutions and time taken for interior-point and LS
            x_vals[0], t_vals[0, attempt], _ = l2_min(A, b)
            x_vals[1], t_vals[1, attempt], _ = l1_min(
                A, b, method='interior-point'
            )
            x_vals[2], t_vals[2, attempt], _ = linf_min(
                A, b, method='interior-point'
            )
            # For small problems, calculate time taken for the simplex method
            if n <= max_simplex_n:
                _, t_vals[3, attempt], _ = l1_min(
                    A, b, method='simplex'
                )
                _, t_vals[4, attempt], _ = linf_min(
                    A, b, method='simplex'
                )
        # Save results for each problem in a `.npz` file
        np.savez(
            folder + filename_prefix + str(problem),
            x_vals=x_vals, t_vals=t_vals
        )
    time_taken = time() - start_time
    print("All problems analysed in {:.4g} s".format(time_taken))

if __name__ == "__main__":
    A, b = fileio.load_A_b(3)
    print(A.shape, b.shape)
    # x, _, _ = l1_min(A, b, method='interior-point')
    # print(x.shape)
    # linf_min(A, b)
    # l2_min(A, b)
    analyse_methods(problem_list=[1, 2, 3], max_simplex_n=0)
