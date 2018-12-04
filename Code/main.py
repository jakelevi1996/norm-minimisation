import numpy as np
import matplotlib.pyplot as plt
from time import time
import fileio, linopt as lo

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
        x_vals[0], _, _, _ = lo.l1_min(
            A, b, method='interior-point', verbose=very_verbose
        )
        x_vals[1], _, _ = lo.l2_min(A, b, verbose=very_verbose)
        x_vals[2], _, _, _ = lo.linf_min(
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
            _, _, t_vals[0, attempt], _ = lo.l1_min(
                A, b, method='interior-point', verbose=very_verbose
            )
            if n <= max_n_simplex:
                _, _, t_vals[1, attempt], _ = lo.l1_min(
                    A, b, method='simplex', verbose=very_verbose
                )
            _, _, t_vals[2, attempt] = lo.l2_min(A, b, verbose=very_verbose)
            _, _, t_vals[3, attempt], _ = lo.linf_min(
                A, b, method='interior-point', verbose=very_verbose
            )
            if n <= max_n_simplex:
                _, _, t_vals[4, attempt], _ = lo.linf_min(
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

def plot_t_graphs(
    t_vals_list=None, t_vals_filename="Results/Protected/t_vals.npz",
    problem_list=range(1, 6), output_filename="Images/Norm minimisation times",
    t_res=1.2e-3
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
            plt.loglog(n_list[t > t_res], t[t > t_res], fmt, alpha=0.3)
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

# THINGS GO DOWN-HILL FROM HERE

def print_l1_perf(solution_strategy, final_value, nit, t):
    print(
        "Residual = {:.5} using {:<10} after {:<4} iters in {:.4} s".format(
            final_value, solution_strategy, nit, t
        )
    )

def compare_smooth_to_exact_l1(problem_num=1):
    A, b = fileio.load_A_b(problem_num)
    # Smooth gradient descent
    _, smooth_value, smooth_time, nit = lo.min_smooth_l1_gradient_descent(
        A, b, forward_tracking=False, verbose=False
    )
    print_l1_perf("Smooth GD", smooth_value, nit, smooth_time)
    # Smooth gradient descent with forward tracking
    _, smooth_value, smooth_time, nit = lo.min_smooth_l1_gradient_descent(
        A, b, forward_tracking=True, verbose=False
    )
    print_l1_perf("SmGD (fwd)", smooth_value, nit, smooth_time)
    # Solving exact linear program using IP
    _, lp_value, lp_time, nit = lo.l1_min(A, b, verbose=False)
    print_l1_perf("LP (IP)", lp_value, nit, lp_time)
    # Solving exact linear program using simplex
    _, lp_value, lp_time, nit = lo.l1_min(
        A, b, method="simplex", verbose=False
    )
    print_l1_perf("LP (simplex)", lp_value, nit, lp_time)
    # Newton method
    _, smooth_value, smooth_time, nit = lo.min_smooth_l1_newton(
        A, b, forward_tracking=True, verbose=False
    )
    print_l1_perf("Newton", smooth_value, nit, smooth_time)

def print_l1_perf_brief(solution_strategy, time):
    print("Finished {:<10} in {:.4} s".format(solution_strategy, time))

def find_t_vals_l1(
    problem_list=range(1, 6), output_filename="Results/t_vals_l1.npz",
    max_n=[256, 256, 512, 64, 1024, 1024], num_attempts=3,
    save_results=True, verbose=True, very_verbose=True
):
    t_start = time()
    t_vals_list = []
    for problem in problem_list:
        A, b = fileio.load_A_b(problem)
        n = A.shape[1]
        t_vals = np.zeros([6, num_attempts])
        for a in range(num_attempts):
            print("\n***Problem {}, attempt {}/{}...".format(
                problem, a+1, num_attempts
            ))
            # Smooth gradient descent
            if n <= max_n[0]:
                _, _, t_vals[0, a], _ = lo.min_smooth_l1_gradient_descent(
                    A, b, forward_tracking=False, verbose=False
                )
                print_l1_perf_brief("Smooth GD", t_vals[0, a])
            # Smooth gradient descent with forward tracking
            if n <= max_n[1]:
                _, _, t_vals[1, a], _ = lo.min_smooth_l1_gradient_descent(
                    A, b, forward_tracking=True, verbose=False
                )
                print_l1_perf_brief("SmGD fwd", t_vals[1, a])
            # Solving exact linear program using IP
            if n <= max_n[2]:
                _, _, t_vals[2, a], _ = lo.l1_min(A, b, verbose=False)
                print_l1_perf_brief("LP (IP)", t_vals[2, a])
            # Solving exact linear program using simplex
            if n <= max_n[3]:
                _, _, t_vals[3, a], _ = lo.l1_min(
                    A, b, method="simplex", verbose=False
                )
                print_l1_perf_brief("Simplex)", t_vals[3, a])
            # Newton method
            if n <= max_n[4]:
                _, _, t_vals[4, a], _ = lo.min_smooth_l1_newton(
                    A, b, forward_tracking=False, verbose=False
                )
                print_l1_perf_brief("Newton", t_vals[4, a])
            # Newton forward-tracking
            if n <= max_n[5]:
                _, _, t_vals[5, a], _ = lo.min_smooth_l1_newton(
                    A, b, forward_tracking=True, verbose=False
                )
                print_l1_perf_brief("Newt (fwd)", t_vals[5, a])
        t_vals_list.append(t_vals)
    
    time_taken = time() - t_start
    if save_results: fileio.save_vals_list(t_vals_list, output_filename)
    if verbose: display_time_taken(time_taken)

def plot_t_graphs_l1(
    t_vals_list=None, t_vals_filename="Results/Protected/t_vals_l1.npz",
    problem_list=range(1, 6), output_filename="Images/L1 minimisation times",
    t_res=1.2e-3
):
    # TODO: evaluate n_list from problem_list
    n_list=np.array([16, 64, 256, 512, 1024])
    n_problems = len(problem_list)
    if t_vals_list is None:
        t_vals_list = fileio.load_vals_list(t_vals_filename)
    assert len(t_vals_list) == n_problems
    n_attempts = t_vals_list[0].shape[1]
    fmt_list = ["ro-", "ro:", "bo-", "bo:", "go-", "go--"]

    plt.figure(figsize=[8, 6])
    for attempt in range(n_attempts):
        for p, fmt in enumerate(fmt_list):
            t = np.array(
                [t_vals_list[n][p, attempt] for n in range(n_problems)]
            )
            plt.loglog(n_list[t > t_res], t[t > t_res], fmt, alpha=0.3)
    plt.xlabel("n")
    plt.ylabel("Computation time (s)")
    title = "Computation time against problem size "
    title += "for different methods of L1 norm-minimisation"
    plt.title(title)
    plt.legend([
        "Smooth gradient-descent", "Smooth GD (forward-tracking)",
        "LP (interior point)", "LP (simplex)", "Newton method",
        "Newton (forward-tracking)"
    ])
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()


def plot_against_epsilon(
    f_filename="Images/f against epsilon",
    t_filename="Images/t against epsilon", eps_lims=[-3, -1], problem_num=1
):
    A, b = fileio.load_A_b(problem_num)
    epsilon_list = np.logspace(*eps_lims)
    t_back_list = np.zeros(epsilon_list.size)
    t_forwards_list = np.zeros(epsilon_list.size)
    f_list = np.zeros(epsilon_list.size)

    for i, eps in enumerate(epsilon_list):
        print("epsilon = {:.4}".format(eps))
        _, f_list[i], t_back_list[i], _ = lo.min_smooth_l1_gradient_descent(
            A, b, epsilon=eps, forward_tracking=False, verbose=False
        )
        _, _, t_forwards_list[i], _ = lo.min_smooth_l1_gradient_descent(
            A, b, epsilon=eps, forward_tracking=True, verbose=False
        )
    plt.figure(figsize=[8, 6])
    plt.semilogx(epsilon_list, f_list)
    plt.xlabel("Epsilon")
    plt.ylabel("Final objective function value")
    plt.title("Final performance against epsilon for fixed gradient tolerance")
    plt.grid(True)
    plt.savefig(f_filename)
    plt.close()

    plt.figure(figsize=[8, 6])
    plt.loglog(epsilon_list, t_back_list, epsilon_list, t_forwards_list)
    plt.xlabel("Epsilon")
    plt.ylabel("Time taken for convergence (s)")
    plt.title("Computation time against epsilon for fixed gradient tolerance")
    plt.legend([
        "Backtracking line-search", "Backtracking with forward-tracking"
    ])
    plt.grid(True)
    plt.savefig(t_filename)
    plt.close()

def plot_newton_vs_gradient_descent(
    i_filename="Images/f against i",
    t_filename="Images/f against t",
    problem_num=2, num_attempts=3, n_its_gd=1650, n_its_newton=23
):
    A, b = fileio.load_A_b(problem_num)
    f_list = [[None for _ in range(num_attempts)] for _ in range(5)]
    t_list = [[None for _ in range(num_attempts)] for _ in range(5)]
    i_list = [[None for _ in range(num_attempts)] for _ in range(5)]
    
    lo.fixed_its_gd(A, b, nits=n_its_gd, verbose=False)
    for a in range(num_attempts):
        print("attempt", a+1)
        f_list[0][a], t_list[0][a], i_list[0][a] = lo.fixed_its_gd(
            A, b, nits=n_its_gd, verbose=False
        )
        f_list[1][a], t_list[1][a], i_list[1][a] = lo.fixed_its_gd(
            A, b, nits=n_its_gd, forward_tracking=True, verbose=False
        )
        f_list[2][a], t_list[2][a], i_list[2][a] = lo.fixed_its_newton(
            A, b, nits=n_its_newton, verbose=False
        )
        f_list[3][a], t_list[3][a], i_list[3][a] = lo.fixed_its_newton(
            A, b, nits=n_its_newton, forward_tracking=True, verbose=False
        )
        f_list[4][a], t_list[4][a], i_list[4][a] = lo.fixed_its_newton(
            A, b, nits=n_its_gd, diag_approx=True, verbose=False
        )

    plt.figure(figsize=[8, 6])
    for a in range(num_attempts):
        plt.semilogx(
            i_list[0][a], f_list[0][a], "b-",
            i_list[1][a], f_list[1][a], "b:",
            i_list[2][a], f_list[2][a], "g-",
            i_list[3][a], f_list[3][a], "g:",
            i_list[4][a], f_list[4][a], "g--",
            alpha=0.3
        )
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.title("Performance against iteration")
    plt.legend([
        "Gradient descent ", "GD (forward-tracking)",
        "Newton method", "Newton (FT)", "Newton (diagonal)"
    ])
    plt.grid(True)
    plt.savefig(i_filename)
    plt.close()

    plt.figure(figsize=[8, 6])
    for a in range(num_attempts):
        plt.semilogx(
            t_list[0][a], f_list[0][a], "b-",
            t_list[1][a], f_list[1][a], "b:",
            t_list[2][a], f_list[2][a], "g-",
            t_list[3][a], f_list[3][a], "g:",
            t_list[4][a], f_list[4][a], "g--",
            alpha=0.3
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Objective function value")
    plt.title("Performance against time")
    plt.legend([
        "Gradient descent ", "GD (forward-tracking)",
        "Newton method", "Newton (FT)", "Newton (diagonal)"
    ])
    plt.grid(True)
    plt.savefig(t_filename)
    plt.close()

def card_vs_gamma(
    filename="Images/Cardinality as a function of gamma", problem_num=5,
    num_gamma=50, gamma_lims=[-0.5, 0.5], epsilon_list=[1e-3/4, 1e-3, 1e-3*4, ]
):
    A, b = fileio.load_A_b(problem_num)
    gamma_list = np.logspace(*gamma_lims, num_gamma)
    card_list = np.zeros([len(epsilon_list), gamma_list.size])
    for ie, e in enumerate(epsilon_list):
        for ig, g in enumerate(gamma_list):
            _, _, card_list[ie, ig], _, _ = lo.min_smooth_card_gd(
                A, b, epsilon=e, gamma=g, forward_tracking=True,
                verbose=False, very_verbose=False
            )
            print(g, card_list[ie, ig])
    plt.figure(figsize=[8, 6])
    for ie in range(len(epsilon_list)):
        plt.loglog(gamma_list, card_list[ie])
    plt.xlabel("gamma")
    plt.ylabel("Cardinality")
    plt.title("Cardinality as a function of gamma for varying epsilon")
    plt.legend(["Epsilon = {:.3}".format(e) for e in epsilon_list])
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # find_x_vals(problem_list=range(1, 3), save_results=False)
    # find_x_vals()
    # find_t_vals(problem_list=range(1, 2), save_results=True, num_attempts=1)
    # find_t_vals()
    print_t_tables()
    # print_residuals_tables()
    # [print_residuals_tables(p=p) for p in [1, 2, np.inf]]
    # plot_t_graphs()
    # residual_histograms()
    # compare_smooth_to_exact_l1(2)
    # find_t_vals_l1()
    # plot_t_graphs_l1()
    # plot_against_epsilon()
    # plot_newton_vs_gradient_descent(n_its_gd=250)
    card_vs_gamma()
