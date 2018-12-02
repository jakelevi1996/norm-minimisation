import numpy as np
from numpy.linalg import norm
from scipy.optimize import linprog
from scipy.linalg import solve
from time import time

import fileio, results

def ax_take_b(a, x, b): return A.dot(x) - b

def display_lp_min_results(solution_norm, time_taken):
    print("Minimised norm = {:.6}\tTime taken = {:.4} s".format(
        solution_norm, time_taken
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
    solution_norm = norm(ax_take_b(A, x, b), 1)
    if verbose: display_lp_min_results(solution_norm, time_taken)

    return x, solution_norm, time_taken

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
    solution_norm = norm(ax_take_b(A, x, b), np.inf)
    if verbose: display_lp_min_results(solution_norm, time_taken)

    return x, solution_norm, time_taken

def l2_min(A, b, verbose=True):
    assert A.ndim == 2 and b.ndim == 1
    t_start = time()
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    time_taken = time() - t_start
    solution_norm = norm(ax_take_b(A, x, b), 2)
    if verbose: display_lp_min_results(solution_norm, time_taken)
    
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
    if random_init: x = np.random.normal(size=n)
    else: x = np.zeros(shape=n)
    grad = smooth_l1_gradient(A, x, b, epsilon)
    outer_step = 0
    t = t0
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
    return x

def min_smooth_l1_newton(
    A, b, epsilon=0.01, t0=1.0, alpha=0.5, beta=0.5, grad_tol=1e-3,
    random_init=False, forward_tracking=False, diag_approx=False, verbose=True
):
    n = A.shape[1]
    if random_init: x = np.random.normal(size=n)
    else: x = np.zeros(shape=n)
    grad = smooth_l1_gradient(A, x, b, epsilon)
    outer_step = 0
    t = t0
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
    return x

def smooth_card(A, x, b, epsilon, gamma):
    Axb = ax_take_b(A, x, b)
    return norm(Axb, 2) + gamma * np.sqrt(x**2 + epsilon**2).sum()

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

def display_cardinality_results(gamma, cardinality, sparsity):
    print(
        "Gamma = {:.4}, cardinality = {},".format(gamma, cardinality),
        "Sparsity pattern (indexing starting at 1) is:\n",
        np.arange(x.size)[sparsity] + 1
    )

def min_smooth_card_gradient_descent(
    A, b, epsilon=1e-3, gamma=2.09, t0=1e-2, alpha=0.5, beta=0.5,
    grad_tol=1e-5, random_init=False, forward_tracking=True,
    verbose=True, very_verbose=True
):
    n = A.shape[1]
    if random_init: x = np.random.normal(size=n)
    else: x = np.zeros(shape=n)
    grad = smooth_card_grad(A, x, b, epsilon, gamma)
    outer_step = 0
    t = t0
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
    sparsity = np.abs(x) >= epsilon
    cardinality = (sparsity).sum()
    if verbose: display_cardinality_results(gamma, cardinality, sparsity)

    return x, sparsity, cardinality

def min_sparse_l2(A, b, sparsity):
    A = A[:, sparsity]
    x, residual, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    print(
        "Residual is {:.8g}, rank is {}, ".format(*residual, rank),
        "Values of x are\n:", x
    )
    return x


if __name__ == "__main__":
    A, b = fileio.load_A_b(1, verbose=True)
    for i in range(1, 5):
        A, b = fileio.load_A_b(i)
        # x, _, _ = l1_min(A, b, method='interior-point')
        x, solution_norm, t = linf_min(A, b, method='interior-point')
        # x, _, _ = l2_min(A, b)
    # analyse_methods(
    #     problem_list=[1, 2, 3], max_simplex_n=0, filename_prefix='test'
    # )
    # analyse_methods()
    # min_smooth_l1_backtracking(A, b)
    # min_smooth_l1_gradient_descent(A, b, epsilon=1e-1, forward_tracking=True)
    # min_smooth_l1_backtracking(A, b, beta=0.1)
    # min_smooth_l1_backtracking(A, b, t0=1e-3, alpha=0.9)
    # min_smooth_l1_newton(A, b)
    # min_smooth_l1_newton(A, b, forward_tracking=True, diag_approx=True)
    # x, sparsity, _ = min_smooth_card_gradient_descent(A, b, gamma=2.2)
    # min_sparse_l2(A, b, sparsity)