import time
import numpy as np
import helper as helper


def matrix_multiplication(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    else:
        A11, A12, A21, A22 = helper.split_matrix(A)
        B11, B12, B21, B22 = helper.split_matrix(B)

        C11 = np.add(matrix_multiplication(A11, B11),
                     matrix_multiplication(A12, B21))
        C12 = np.add(matrix_multiplication(A11, B12),
                     matrix_multiplication(A12, B22))
        C21 = np.add(matrix_multiplication(A21, B11),
                     matrix_multiplication(A22, B21))
        C22 = np.add(matrix_multiplication(A21, B12),
                     matrix_multiplication(A22, B22))

        return helper.combine_matrices(C11, C12, C21, C22)


(A, B) = helper.readFromFile("input/input.txt")
start_time = time.time()
result = matrix_multiplication(A, B)
end_time = time.time()
total_time = end_time - start_time
helper.write_output_file(result, "out/output_straight.txt")
print("                                                                       time is " + str(total_time))
