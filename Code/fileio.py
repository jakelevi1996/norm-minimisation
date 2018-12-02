import numpy as np

DEFAULT_FOLDER = "Data/"
DEFAULT_FILENAME = DEFAULT_FOLDER + "matrices.npz"

a_list = ["A" + str(i) for i in range(1, 6)]
b_list = ["b" + str(i) for i in range(1, 6)]
DEFAULT_FILES_LIST = a_list + b_list
DEFAULT_PROBLEM_NAMES_LIST = ["p" + str(i) for i in range(1, 6)]

def csv_to_npz(
    input_folder=DEFAULT_FOLDER,
    input_files=DEFAULT_FILES_LIST,
    output_filename=DEFAULT_FILENAME
):
    array_list = [np.loadtxt(
        input_folder + file + ".csv", delimiter=','
    ) for file in input_files]

    np.savez(output_filename, **{
        file: array_list[i] for i, file in enumerate(input_files)
    })

def load_A_b(index=1, filename=DEFAULT_FILENAME, verbose=False):
    with np.load(filename) as data:
        A = data["A" + str(index)]
        b = data["b" + str(index)]
    if verbose: print("Finished loading data")
    return A, b

def save_vals_list(
    vals_list, filename="Results/results.npz",
    problem_names_list=DEFAULT_PROBLEM_NAMES_LIST
):
    n = len(problem_names_list)
    assert len(vals_list) == n
    np.savez(filename, **{
        problem_names_list[i]: vals_list[i] for i in range(n)
    })

def load_vals_list(
    filename="Results/results.npz",
    problem_names_list=DEFAULT_PROBLEM_NAMES_LIST
):
    with np.load(filename) as data:
        vals_list = [data[p] for p in problem_names_list]

    return vals_list

def load_results(
    index=1, filename_prefix="Results/Protected/results_problem_"
):
    with np.load(filename_prefix + str(index) + ".npz") as data:
        x_vals = data["x_vals"]
        t_vals = data["t_vals"]
    
    n = x_vals.shape[1]
    return x_vals, t_vals, n


if __name__ == "__main__":
    print(DEFAULT_FILES_LIST)
    # csv_to_npz()
    A, b = load_A_b()
    print(A.shape, b.shape)
    # print(A)
    # print(b)
