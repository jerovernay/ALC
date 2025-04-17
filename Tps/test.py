# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:00:00 2025

@author: jeron
"""

import numpy as np
from scipy.linalg import lu, solve_triangular

def inversa_por_lu(A):
    n = A.shape[0]

    # Realizamos la factorización LU de la matriz A
    P, L, U = lu(A)

    # Inicializamos la matriz identidad I
    I = np.eye(n)
    A_inv = np.zeros_like(A, dtype=float)

    # Resolvemos para cada columna de la matriz inversa
    for i in range(n):
        b = I[:, i]  # La columna i de la identidad

        # Resolvemos L y U
        y = solve_triangular(L, P @ b, lower=True)
        x = solve_triangular(U, y)

        A_inv[:, i] = x  # Guardamos el resultado en la columna i de A_inv

    return A_inv

# Ejemplo de uso:
A = np.array([
    [4, 7, 2],
    [3, 5, 1],
    [2, 3, 6]
], dtype=float)

A_inv = inversa_por_lu(A)
print("Matriz A:")
print(A)
print("Matriz A inversa:")
print(A_inv)

# Verificamos que A @ A_inv = I
print("Chequeo A @ A_inv:")
print(np.dot(A, A_inv))  # Debería dar la identidad