import numpy as np

import concurrent.futures


def pram_matrix_multiplication(A, B):

    # Get the dimensions of the matrices.

    n = A.shape[0]

    # Create a 2D mesh of processors.

    processors = [[None for _ in range(n)] for _ in range(n)]

    # Distribute the elements of the matrices to the processors.

    for i in range(n):

        for j in range(n):

            processors[i][j] = A[i, j] * B[i, j]

    # Start the parallel computation.

    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = [executor.submit(lambda x: x, processors[i][j])
                   for i in range(n) for j in range(n)]

        for future in futures:
            processors[i][j] = future.result()

    # Gather the results from the processors.

    C = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):

        for j in range(n):

            C[i][j] = processors[i][j]

    # Return the product of the matrices.

    return C


# Create two matrices.

A = np.array([[1, 2], [3, 4]])

B = np.array([[5, 6], [7, 8]])


# Multiply the matrices using the PRAM parallel matrix multiplication algorithm.

C = pram_matrix_multiplication(A, B)


# Print the product matrix.

print(C)
