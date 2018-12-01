import numpy as np
from numpy.linalg import norm
from scipy.optimize import linprog
from scipy.linalg import solve
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
    start_time = time()
    x, residual, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    time_taken = time() - start_time
    print("Residual is {:.8g}, rank is {}, time taken is {:.4g} s".format(
        *residual, rank, time_taken
    ))
    return x, time_taken, rank

def smooth_l1(A, x, b, epsilon):
    Axb = A.dot(x) - b
    return np.sqrt(Axb ** 2 + epsilon ** 2).sum()

def smooth_l1_gradient(A, x, b, epsilon):
    Axb = A.dot(x) - b
    u = Axb / np.sqrt(Axb ** 2 + epsilon ** 2)
    return A.T.dot(u)

def smooth_l1_hessian(A, x, b, epsilon):
    Axb = A.dot(x) - b
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
            norm(grad), smooth_l1(A, x, b, epsilon)
        )
    )

def min_smooth_l1_gradient_descent(
    A, b, epsilon=0.01, t0=1e-2, alpha=0.5, beta=0.5, grad_tol=1e-3,
    random_init=False, forward_tracking=False
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
        display_backtracking_progress(
            outer_step, inner_step, grad, A, x, b, epsilon
        )
        outer_step += 1
    return x

def min_smooth_l1_newton(
    A, b, epsilon=0.01, t0=1.0, alpha=0.5, beta=0.5, grad_tol=1e-3,
    random_init=False, forward_tracking=False, diag_approx=False
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
        display_backtracking_progress(
            outer_step, inner_step, grad, A, x, b, epsilon
        )
        outer_step += 1
    return x

def smooth_card(A, x, b, epsilon, gamma):
    return norm(A.dot(x) - b, 2) + gamma * np.sqrt(x**2 + epsilon**2).sum()

def smooth_card_grad(A, x, b, epsilon, gamma):
    Axb = A.dot(x) - b
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
        "Gamma = {:.4}, cardinality = {}, ".format(gamma, cardinality),
        "Sparsity pattern (indexing starting at 1) is:\n",
        np.arange(x.size)[sparsity] + 1
    )

def min_smooth_card_gradient_descent(
    A, b, epsilon=1e-3, gamma=2.09, t0=1e-2, alpha=0.5, beta=0.5,
    grad_tol=1e-5, random_init=False, forward_tracking=True
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
        display_backtracking_progress(
            outer_step, inner_step, grad, A, x, b, epsilon
        )
        outer_step += 1
    sparsity = np.abs(x) >= epsilon
    cardinality = (sparsity).sum()
    display_cardinality_results(gamma, cardinality, sparsity)

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
    A, b = fileio.load_A_b(5)
    # print(A.shape, b.shape)
    # x, _, _ = l1_min(A, b, method='interior-point')
    # x, _, _ = linf_min(A, b, method='interior-point')
    # x, _, _ = l2_min(A, b)
    # print(x.shape)
    # print("{:.4}".format(norm(x, 1)))
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
    x, sparsity, _ = min_smooth_card_gradient_descent(A, b, gamma=2.2)
    min_sparse_l2(A, b, sparsity)