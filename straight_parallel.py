
import time
import multiprocessing as mp
import numpy as np
import concurrent.futures
import helper as helper

global executor
executor = concurrent.futures.ProcessPoolExecutor()


def matrix_multiplication(A, B, P):
    global executor
    n = len(A)
    C = [[0 for _ in range(n)] for __ in range(n)]
    if n <= 128:
        C = np.matmul(A, B)

    else:
        A11, A12, A21, A22 = helper.split_matrix(A)
        B11, B12, B21, B22 = helper.split_matrix(B)

        if P >= 1:

            futures = []
            futures.append(executor.submit(
                matrix_multiplication, A11, B11, (P-1)))
            futures.append(executor.submit(
                matrix_multiplication, A12, B21, (P-1)))
            futures.append(executor.submit(
                matrix_multiplication, A11, B12, (P-1)))
            futures.append(executor.submit(
                matrix_multiplication, A12, B22, (P-1)))
            futures.append(executor.submit(
                matrix_multiplication, A21, B11, (P-1)))
            futures.append(executor.submit(
                matrix_multiplication, A22, B21, (P-1)))
            futures.append(executor.submit(
                matrix_multiplication, A21, B12, (P-1)))
            futures.append(executor.submit(
                matrix_multiplication, A22, B22, (P-1)))

            C11 = np.add(futures[0].result(), futures[1].result())
            C12 = np.add(futures[2].result(), futures[3].result())
            C21 = np.add(futures[4].result(), futures[5].result())
            C22 = np.add(futures[6].result(), futures[7].result())

        else:
            C11 = np.add(matrix_multiplication(
                A11, B11, 0), matrix_multiplication(A12, B21, 0))
            C12 = np.add(matrix_multiplication(
                A11, B12, 0), matrix_multiplication(A12, B22, 0))
            C21 = np.add(matrix_multiplication(
                A21, B11, 0), matrix_multiplication(A22, B21, 0))
            C22 = np.add(matrix_multiplication(
                A21, B12, 0), matrix_multiplication(A22, B22, 0))

        C = helper.combine_matrices(C11, C12, C21, C22)
    return C


(A, B) = helper.readFromFile("input/input.txt")
start_time = time.time()
result = matrix_multiplication(A, B, 1)
end_time = time.time()
total_time = end_time - start_time
helper.write_output_file(result, "out/output_straight_parallel.txt")
print("                                                                       time is " + str(total_time))
