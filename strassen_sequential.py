import time
import multiprocessing as mp
import numpy as np
import concurrent.futures
import helper as helper

global executor
executor = concurrent.futures.ProcessPoolExecutor()


def strassen(a, b):
    """
    Multiply two matrices using Strassen's algorithm.
    """
    n = len(a)

    # Base case: 1x1 matrices
    if n == 1:
        return a * b

    # Split matrices into quarters
    a11, a12, a21, a22 = helper.split_matrix(a)
    b11, b12, b21, b22 = helper.split_matrix(b)

    # Compute sub-products
    p1 = strassen(np.add(a11, a22), np.add(b11, b22))
    p2 = strassen(np.add(a21, a22), b11)
    p3 = strassen(a11, np.subtract(b12, b22))
    p4 = strassen(a22, np.subtract(b21, b11))
    p5 = strassen(np.add(a11, a12), b22)
    p6 = strassen(np.subtract(a21, a11), np.add(b11, b12))
    p7 = strassen(np.subtract(a12, a22), np.add(b21, b22))

    # # Compute final product
    c11 = np.add((np.add(p1, p4)), (np.add(np.negative(p5), p7)))
    c12 = np.add(p3, p5)
    c21 = np.add(p2, p4)
    c22 = np.add((np.subtract(p1, p2)), (np.add(p3, p6)))

    # Combine sub-products into result

    return helper.combine_matrices(c11, c12, c21, c22)

# Get user input for matrices A and B


(A, B) = helper.readFromFile("input/input.txt")
start_time = time.time()
result = strassen(A, B)
end_time = time.time()
total_time = end_time - start_time
helper.write_output_file(result, "out/strassen.txt")
print("                                                                       time is " + str(total_time))
