import numpy as np

def solve_linear_eq_with_inverse(A, b):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matriks koefisien harus berbentuk matriks persegi.")
    
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    
    return x

# Contoh penggunaan sistem persamaan linear:
# 3x + 2y = 7
# 5x - 3y = 8

# Matriks koefisien A
A = np.array([[3, 2],
              [5, -3]])

# Vektor hasil b
b = np.array([[7],
              [8]])

# Memanggil fungsi untuk menyelesaikan sistem persamaan linear
solusi = solve_linear_eq_with_inverse(A, b)

print("Solusi x dan y:")
print("x =", solusi[0][0])
print("y =", solusi[1][0])