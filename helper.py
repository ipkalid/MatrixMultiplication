
import numpy as np


def read_input(file_name):
    with open(file_name, 'r') as f:
        n = int(f.readline().strip())
        matrix_a = []
        matrix_b = []
        for i in range(2**n):
            row = list(map(int, f.readline().strip().split()))
            matrix_a.append(row)
        for i in range(2**n):
            row = list(map(int, f.readline().strip().split()))
            matrix_b.append(row)
    return matrix_a, matrix_b


def write_output_file(C, filename):
    with open(filename, 'w') as f:
        for row in C:
            f.write(' '.join([str(elem) for elem in row]))
            f.write('\n')


def write_info_file(T, filename):
    hours = int(T // 3600)
    minutes = int((T % 3600) // 60)
    seconds = int(T % 60)
    time_str = '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    with open(filename, 'w') as f:
        f.write(time_str)


def readFromFile(fileName):
    with open(fileName, "r") as f:
        n = int(f.readline())
        values = list(map(int, f.readline().split()))
        A = np.array(values[:n * n]).reshape(n, n)
        B = np.array(values[n * n:]).reshape(n, n)
    return (A, B)


def split_matrix(M):
    n = len(M)//2
    A11 = [[M[i][j] for j in range(n)] for i in range(n)]
    A12 = [[M[i][j] for j in range(n, len(M))] for i in range(n)]
    A21 = [[M[i][j] for j in range(n)] for i in range(n, len(M))]
    A22 = [[M[i][j] for j in range(n, len(M))] for i in range(n, len(M))]

    return A11, A12, A21, A22


def combine_matrices(C11, C12, C21, C22):
    n = len(C11)*2
    M = [[0 for i in range(n)] for j in range(n)]

    for i in range(len(C11)):
        for j in range(len(C11)):
            M[i][j] = C11[i][j]
            M[i][j+len(C11)] = C12[i][j]
            M[i+len(C11)][j] = C21[i][j]
            M[i+len(C11)][j+len(C11)] = C22[i][j]

    return M
