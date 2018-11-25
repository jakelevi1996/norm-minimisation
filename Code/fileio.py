import numpy as np

DEFAULT_FOLDER = "Data/"
DEFAULT_FILENAME = DEFAULT_FOLDER + "matrices.npz"

a_list = ["A" + str(i) for i in range(1, 6)]
b_list = ["b" + str(i) for i in range(1, 6)]
DEFAULT_FILES_LIST = a_list + b_list

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

def load_A_b(index=1, filename=DEFAULT_FILENAME):
    with np.load(filename) as data:
        A = data["A" + str(index)]
        b = data["b" + str(index)]
    return A, b

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
