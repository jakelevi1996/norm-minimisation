import numpy as np
from numpy.linalg import norm
from scipy.optimize import linprog
from scipy.linalg import solve
from time import time, perf_counter

import fileio

def ax_take_b(a, x, b): return a.dot(x) - b

def residual(A, x, b, p): return norm(ax_take_b(A, x, b), p)

def display_lp_min_results(n, p, solution_norm, time_taken):
    print("n={}\tMinimised {}-norm\t= {:.6}\tTime taken = {:.4} s".format(
        n, p, solution_norm, time_taken
    ))

def l1_min(A, b, method='interior-point', verbose=True):
    assert A.ndim == 2 and b.ndim == 1
    m, n = A.shape
    I = np.identity(m)
    A_ub = np.block([[A, -I], [-A, -I]])
    b_ub = np.block([b, -b])
    c = np.block([np.zeros(n), np.ones(m)])
    t_start = time()
    res = linprog(
        c, A_ub, b_ub, bounds=(None, None),
        options={"maxiter": np.inf, "tol": 1e-7}, method=method
    )
    time_taken = time() - t_start
    assert res.status == 0
    x = res.x[:n]
    solution_norm = residual(A, x, b, 1)
    if verbose: display_lp_min_results(n, 1, solution_norm, time_taken)

    return x, solution_norm, time_taken, res.nit

def linf_min(A, b, method='interior-point', verbose=True):
    assert A.ndim == 2 and b.ndim == 1
    m, n = A.shape
    ones = np.ones([m, 1])
    A_ub = np.block([[A, -ones], [-A, -ones]])
    b_ub = np.block([b, -b])
    c = np.block([np.zeros(n), 1])
    t_start = time()
    res = linprog(
        c, A_ub, b_ub, bounds=(None, None),
        options={"maxiter": np.inf, "tol": 1e-7}, method=method
    )
    time_taken = time() - t_start
    assert res.status == 0
    x = res.x[:n]
    solution_norm = residual(A, x, b, np.inf)
    if verbose: display_lp_min_results(n, np.inf, solution_norm, time_taken)

    return x, solution_norm, time_taken, res.nit

def l2_min(A, b, verbose=True):
    assert A.ndim == 2 and b.ndim == 1
    t_start = time()
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    time_taken = time() - t_start
    n = x.size
    solution_norm = residual(A, x, b, 2)
    if verbose: display_lp_min_results(n, 2, solution_norm, time_taken)
    
    return x, solution_norm, time_taken

def smooth_l1(A, x, b, epsilon):
    return np.sqrt(ax_take_b(A, x, b) ** 2 + epsilon ** 2).sum()

def smooth_l1_gradient(A, x, b, epsilon):
    Axb = ax_take_b(A, x, b)
    u = Axb / np.sqrt(Axb ** 2 + epsilon ** 2)
    return A.T.dot(u)

def smooth_l1_hessian(A, x, b, epsilon):
    Axb = ax_take_b(A, x, b)
    Lambda = (epsilon ** 2) / (np.sqrt(Axb ** 2 + epsilon ** 2) ** 3)
    return A.T.dot(Lambda.reshape(-1, 1) * A)

def smooth_l1_backtrack_condition(A, x, b, epsilon, t, delta, alpha, grad):
    old_val = smooth_l1(A, x, b, epsilon)
    new_val = smooth_l1(A, x + t * delta, b, epsilon)
    min_decrease = -alpha * t * grad.dot(delta)

    return old_val - new_val > min_decrease

def display_backtracking_progress(
    outer_step, inner_step, grad, A, x, b, epsilon
):
    print(
        "Outer = {:<3}, inner = {:<3}".format(outer_step, inner_step),
        "norm(grad) = {:.4}, func = {:.10}".format(
            norm(grad, 1), smooth_l1(A, x, b, epsilon)
        )
    )

def min_smooth_l1_gradient_descent(
    A, b, epsilon=0.01, t0=1e-2, alpha=0.5, beta=0.5, grad_tol=1e-3,
    random_init=False, forward_tracking=False, verbose=True
):
    n = A.shape[1]
    outer_step = 0
    t = t0
    t_start = perf_counter()
    if random_init: x = np.random.normal(size=n)
    else: x = np.zeros(shape=n)
    grad = smooth_l1_gradient(A, x, b, epsilon)
    while norm(grad) >= grad_tol:
        inner_step = 0
        if not forward_tracking: t = t0
        if not smooth_l1_backtrack_condition(
            A, x, b, epsilon, t, -grad, alpha, grad
        ):
            while not smooth_l1_backtrack_condition(
                A, x, b, epsilon, t, -grad, alpha, grad
            ):
                t = beta * t
                inner_step += 1
        elif forward_tracking:
            while smooth_l1_backtrack_condition(
                A, x, b, epsilon, t, -grad, alpha, grad
            ):
                t = t / beta
                inner_step -= 1
            t = beta * t
            inner_step += 1
            
        x = x - t * grad
        grad = smooth_l1_gradient(A, x, b, epsilon)
        if verbose: display_backtracking_progress(
            outer_step, inner_step, grad, A, x, b, epsilon
        )
        outer_step += 1
    time_taken = perf_counter() - t_start
    objective_value = smooth_l1(A, x, b, epsilon)
    return x, objective_value, time_taken, outer_step

def min_smooth_l1_newton(
    A, b, epsilon=0.01, t0=1.0, alpha=0.5, beta=0.5, grad_tol=1e-3,
    random_init=False, forward_tracking=False, diag_approx=False, verbose=True
):
    n = A.shape[1]
    outer_step = 0
    t = t0
    t_start = time()
    if random_init: x = np.random.normal(size=n)
    else: x = np.zeros(shape=n)
    grad = smooth_l1_gradient(A, x, b, epsilon)
    while norm(grad) >= grad_tol:
        hess = smooth_l1_hessian(A, x, b, epsilon)
        if diag_approx: v = - grad / np.diag(hess)
        else: v = -solve(hess, grad, assume_a="pos", check_finite=False)
        inner_step = 0
        if not forward_tracking: t = t0
        if not smooth_l1_backtrack_condition(
            A, x, b, epsilon, t, v, alpha, grad
        ):
            while not smooth_l1_backtrack_condition(
                A, x, b, epsilon, t, v, alpha, grad
            ):
                t = beta * t
                inner_step += 1
        elif forward_tracking:
            while smooth_l1_backtrack_condition(
                A, x, b, epsilon, t, v, alpha, grad
            ):
                t = t / beta
                inner_step -= 1
            t = beta * t
            inner_step += 1
        
        x = x + t * v
        grad = smooth_l1_gradient(A, x, b, epsilon)
        if verbose: display_backtracking_progress(
            outer_step, inner_step, grad, A, x, b, epsilon
        )
        outer_step += 1
    time_taken = time() - t_start
    objective_value = smooth_l1(A, x, b, epsilon)
    return x, objective_value, time_taken, outer_step

def smooth_card(A, x, b, epsilon, gamma):
    return residual(A, x, b, 2) + gamma * np.sqrt(x**2 + epsilon**2).sum()

def smooth_card_grad(A, x, b, epsilon, gamma):
    Axb = ax_take_b(A, x, b)
    return A.T.dot(Axb) / norm(Axb) + gamma * x / np.sqrt(x**2 + epsilon**2)

def smooth_card_backtrack_condition(
    A, x, b, epsilon, gamma, t, delta, alpha, grad
):
    old_val = smooth_card(A, x, b, epsilon, gamma)
    new_val = smooth_card(A, x + t * delta, b, epsilon, gamma)
    min_decrease = -alpha * t * grad.dot(delta)

    return old_val - new_val > min_decrease

def display_cardinality_results(x, gamma, cardinality, sparsity):
    print(
        "\nGamma = {:.4}, cardinality = {},".format(gamma, cardinality),
        "Sparsity pattern (indexing starts at 1) is:\n",
        np.arange(x.size)[sparsity] + 1
    )

def min_smooth_card_gd(
    A, b, epsilon=1e-3, gamma=2.09, t0=1e-2, alpha=0.5, beta=0.5,
    grad_tol=1e-5, random_init=False, forward_tracking=True,
    verbose=True, very_verbose=True
):
    n = A.shape[1]
    outer_step = 0
    t = t0
    t_start = time()
    if random_init: x = np.random.normal(size=n)
    else: x = np.zeros(shape=n)
    grad = smooth_card_grad(A, x, b, epsilon, gamma)
    while norm(grad) >= grad_tol:
        inner_step = 0
        if not forward_tracking: t = t0
        if not smooth_card_backtrack_condition(
            A, x, b, epsilon, gamma, t, -grad, alpha, grad
        ):
            while not smooth_card_backtrack_condition(
                A, x, b, epsilon, gamma, t, -grad, alpha, grad
            ):
                t = beta * t
                inner_step += 1
        elif forward_tracking:
            while smooth_card_backtrack_condition(
                A, x, b, epsilon, gamma, t, -grad, alpha, grad
            ):
                t = t / beta
                inner_step -= 1
            t = beta * t
            inner_step += 1
            
        x = x - t * grad
        grad = smooth_card_grad(A, x, b, epsilon, gamma)
        if very_verbose: display_backtracking_progress(
            outer_step, inner_step, grad, A, x, b, epsilon
        )
        outer_step += 1
    time_taken = time() - t_start
    sparsity = np.abs(x) >= epsilon
    cardinality = sparsity.sum()
    if verbose: display_cardinality_results(x, gamma, cardinality, sparsity)

    return x, sparsity, cardinality, time_taken, outer_step

def min_sparse_l2(A, b, sparsity):
    A = A[:, sparsity]
    x, residual, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    print(
        "Residual is {:.8g}, rank is {}, ".format(*residual, rank),
        "Values of x are:\n", *["{:.4}".format(i) for i in x]
    )
    return x

def fixed_its_gd(
    A, b, nits=300, epsilon=0.01, t0=1e-2, alpha=0.5, beta=0.5,
    random_init=False, forward_tracking=False, verbose=True
):
    n = A.shape[1]
    outer_step = 0
    t = t0
    # t_start = time()
    t_start = perf_counter()
    if random_init: x = np.random.normal(size=n)
    else: x = np.zeros(shape=n)
    f_list = np.empty(nits)
    t_list = np.empty(nits)
    i_list = np.empty(nits)
    grad = smooth_l1_gradient(A, x, b, epsilon)
    for i in range(nits):
        inner_step = 0
        if not forward_tracking: t = t0
        if not smooth_l1_backtrack_condition(
            A, x, b, epsilon, t, -grad, alpha, grad
        ):
            while not smooth_l1_backtrack_condition(
                A, x, b, epsilon, t, -grad, alpha, grad
            ):
                t = beta * t
                inner_step += 1
        elif forward_tracking:
            while smooth_l1_backtrack_condition(
                A, x, b, epsilon, t, -grad, alpha, grad
            ):
                t = t / beta
                inner_step -= 1
            t = beta * t
            inner_step += 1
            
        x = x - t * grad
        grad = smooth_l1_gradient(A, x, b, epsilon)
        if verbose: display_backtracking_progress(
            outer_step, inner_step, grad, A, x, b, epsilon
        )
        f_list[i] = smooth_l1(A, x, b, epsilon)
        # t_list[i] = time() - t_start
        t_list[i] = perf_counter() - t_start
        i_list[i] = i+1
        outer_step += 1
    return f_list, t_list, i_list

def fixed_its_newton(
    A, b, nits=30, epsilon=0.01, t0=1.0, alpha=0.5, beta=0.5,
    random_init=False, forward_tracking=False, diag_approx=False, verbose=True
):
    n = A.shape[1]
    outer_step = 0
    t = t0
    # t_start = time()
    t_start = perf_counter()
    if random_init: x = np.random.normal(size=n)
    else: x = np.zeros(shape=n)
    f_list = np.empty(nits)
    t_list = np.empty(nits)
    i_list = np.empty(nits)
    grad = smooth_l1_gradient(A, x, b, epsilon)
    for i in range(nits):
        hess = smooth_l1_hessian(A, x, b, epsilon)
        if diag_approx: v = - grad / np.diag(hess)
        else: v = -solve(hess, grad, assume_a="pos", check_finite=False)
        inner_step = 0
        if not forward_tracking: t = t0
        if not smooth_l1_backtrack_condition(
            A, x, b, epsilon, t, v, alpha, grad
        ):
            while not smooth_l1_backtrack_condition(
                A, x, b, epsilon, t, v, alpha, grad
            ):
                t = beta * t
                inner_step += 1
        elif forward_tracking:
            while smooth_l1_backtrack_condition(
                A, x, b, epsilon, t, v, alpha, grad
            ):
                t = t / beta
                inner_step -= 1
            t = beta * t
            inner_step += 1
        
        x = x + t * v
        grad = smooth_l1_gradient(A, x, b, epsilon)
        if verbose: display_backtracking_progress(
            outer_step, inner_step, grad, A, x, b, epsilon
        )
        f_list[i] = smooth_l1(A, x, b, epsilon)
        # t_list[i] = time() - t_start
        t_list[i] = perf_counter() - t_start
        i_list[i] = i+1
        outer_step += 1
    return f_list, t_list, i_list


if __name__ == "__main__":
    A, b = fileio.load_A_b(5, verbose=True)
    # for i in range(1, 3):
    #     A, b = fileio.load_A_b(i)
    #     # x, _, _, _ = l1_min(A, b, method='interior-point')
    #     x, solution_norm, t, _ = linf_min(A, b, method='interior-point')
    #     # x, _, _, _ = l2_min(A, b)
    # A, b = fileio.load_A_b(2, verbose=True)
    # min_smooth_l1_gradient_descent(A, b, epsilon=1e-1, forward_tracking=True)
    # min_smooth_l1_newton(A, b, forward_tracking=True)
    # min_smooth_l1_newton(A, b, forward_tracking=True, diag_approx=True)
    x, sparsity, _, _, _ = min_smooth_card_gd(A, b, gamma=2.1)
    min_sparse_l2(A, b, sparsity)
    