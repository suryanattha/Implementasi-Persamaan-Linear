import numpy as np

def lu_gauss_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()

    for k in range(n-1):
        for i in range(k+1, n):
            if U[i, k] != 0.0:
                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                U[i, k:n] -= factor * U[k, k:n]

    return L, U

def solve_lu_gauss(A, b):
    L, U = lu_gauss_decomposition(A)
    n = len(A)
    
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        y[i] /= L[i, i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

A = np.array([[2, -1, 1],
              [-3, 3, -1],
              [-2, 1, 2]])

b = np.array([2, -3, 3])

x = solve_lu_gauss(A, b)
print("Solution x:", x)
