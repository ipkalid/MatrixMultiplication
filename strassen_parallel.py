import time
import multiprocessing as mp
import numpy as np
import concurrent.futures
import helper as helper

global executor
executor = concurrent.futures.ProcessPoolExecutor()


def strassen(a, b, p):
    n = len(a)
    c = [[0 for _ in range(n)] for __ in range(n)]
    # Base case: 1x1 matrices
    if n <= 128:
        return np.matmul(a, b)

    a11, a12, a21, a22 = helper.split_matrix(a)
    b11, b12, b21, b22 = helper.split_matrix(b)
    # Compute sub-products
    if (p >= 1):
        futures = []
        # p1 = strassen(a11 + a22, b11 + b22)
        futures.append(executor.submit(
            strassen, np.add(a11, a22), np.add(b11, b22), (p-1)
        ))
        # p2 = strassen(a21 + a22, b11)
        futures.append(executor.submit(
            strassen, np.add(a21, a22), b11, (p-1)
        ))
        # p3 = strassen(a11, b12 - b22)
        futures.append(executor.submit(
            strassen, a11, np.subtract(b12, b22), (p-1)
        ))
        # p4 = strassen(a22, b21 - b11)
        futures.append(executor.submit(
            strassen, a22, np.subtract(b21, b11), (p-1)
        ))
        # p5 = strassen(a11 + a12, b22)
        futures.append(executor.submit(
            strassen, np.add(a11, a12), b22, (p-1)
        ))
        # p6 = strassen(a21 - a11, b11 + b12)
        futures.append(executor.submit(
            strassen, np.subtract(a21, a11), np.add(b11, b12), (p-1)
        ))
        # p7 = strassen(a12 - a22, b21 + b22)
        futures.append(executor.submit(
            strassen, np.subtract(a12, a22), np.add(b21, b22), (p-1)
        ))

        c11 = np.add((np.add(futures[0].result(), futures[3].result())), (np.add(
            np.negative(futures[4].result()), futures[6].result())))
        c12 = np.add(futures[2].result(), futures[4].result())
        c21 = np.add(futures[1].result(), futures[3].result())
        c22 = np.add((np.subtract(futures[0].result(), futures[1].result())), (np.add(
            futures[2].result(), futures[5].result())))
    else:
        p1 = strassen(np.add(a11, a22), np.add(b11, b22), 0)
        p2 = strassen(np.add(a21, a22), b11, 0)
        p3 = strassen(a11, np.subtract(b12, b22), 0)
        p4 = strassen(a22, np.subtract(b21, b11), 0)
        p5 = strassen(np.add(a11, a12), b22, 0)
        p6 = strassen(np.subtract(a21, a11), np.add(b11, b12), 0)
        p7 = strassen(np.subtract(a12, a22), np.add(b21, b22), 0)

    # Compute final product
        c11 = np.add((np.add(p1, p4)), (np.add(np.negative(p5), p7)))
        c12 = np.add(p3, p5)
        c21 = np.add(p2, p4)
        c22 = np.add((np.subtract(p1, p2)), (np.add(p3, p6)))

    c = helper.combine_matrices(c11, c12, c21, c22)

    # Compute final product

    # Combine sub-products into result

    return c

# Get user input for matrices A and B


(A, B) = helper.readFromFile("input/input.txt")
start_time = time.time()
result = strassen(A, B, 1)
end_time = time.time()
total_time = end_time - start_time
helper.write_output_file(result, "out/strassen_parallel.txt")
print("                                                                       time is " + str(total_time))
