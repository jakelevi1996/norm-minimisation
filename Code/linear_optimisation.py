import numpy as np
from scipy.optimize import linprog
from time import time

from fileio import load_A_b

def l1_min(A, b):
    assert A.ndim == 2 and b.ndim == 1
    I = np.identity(A.shape[0])
    A_lb = np.block([[-A, I], [A, I]])
    b_lb = np.block([b, -b])
    c = np.block([np.zeros(A.shape[1]), np.ones(A.shape[0])])
    print(A_lb.shape, b_lb.shape, c.shape)
    start_time = time()
    res = linprog(
        c, A_ub=-A_lb, b_ub=-b_lb, bounds=(None, None),
        options={"maxiter": np.inf, "tol": 1e-7},
        method='interior-point'
    )
    time_taken = time() - start_time
    print("{} Function value is {:.8g} after {} iterations in {:.4g} s".format(
        res.message, res.fun, res.nit, time_taken
    ))
    return res.x, time_taken, res.status

def linf_min(A, b):
    assert A.ndim == 2 and b.ndim == 1
    ones = np.ones((A.shape[0], A.shape[0]))
    A_lb = np.block([[-A, ones], [A, ones]])
    b_lb = np.block([b, -b])
    c = np.block([np.zeros(A.shape[1]), np.ones(A.shape[0])])
    print(A_lb.shape, b_lb.shape, c.shape)
    start_time = time()
    res = linprog(
        c, A_ub=-A_lb, b_ub=-b_lb, bounds=(None, None),
        options={"maxiter": np.inf, "tol": 1e-7},
        method='interior-point'
    )
    time_taken = time() - start_time
    print("{} Function value is {:.8g} after {} iterations in {:.4g} s".format(
        res.message, res.fun, res.nit, time_taken
    ))
    return res.x, time_taken, res.success

def l2_min(A, b):
    assert A.ndim == 2 and b.ndim == 1
    start_time = time()
    x, residual, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    time_taken = time() - start_time
    print("Residual is {:.8g}, rank is {}, time taken is {:.4g} s".format(
        *residual, rank, time_taken
    ))
    return x, time_taken, rank
    

if __name__ == "__main__":
    A, b = load_A_b(5)
    print(A.shape, b.shape)
    # l1_min(A, b)
    # linf_min(A, b)
    l2_min(A, b)
