import numpy as np

def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        U[k, k] = 1.0
        for j in range(k, n):
            sum_ = sum(L[j, s] * U[s, k] for s in range(k))
            L[j, k] = A[j, k] - sum_
        for i in range(k+1, n):
            sum_ = sum(L[k, s] * U[s, i] for s in range(k))
            U[k, i] = (A[k, i] - sum_) / L[k, k]

    return L, U

def solve_crout(L, U, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

def test_crout_method():
    A = np.array([[2, -1, 1],
                  [-3, 3, -1],
                  [-2, 1, 2]])
    b = np.array([2, -3, 3])

    L, U = crout_decomposition(A)
    x = solve_crout(L, U, b)

    Ax = np.dot(A, x)
    assert np.allclose(Ax, b), "Test failed: Ax is not equal to b"

    print("Test passed: Ax is equal to b")

test_crout_method()
