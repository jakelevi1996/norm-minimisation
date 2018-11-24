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

if __name__ == "__main__":
    print(DEFAULT_FILES_LIST)
    # csv_to_npz()
    A, b = load_A_b()
    print(A.shape, b.shape)
    # print(A)
    # print(b)
