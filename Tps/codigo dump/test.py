from scipy.linalg import solve_triangular
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX



def calculaLU(A):

    def construir_P(A):

        n = A.shape[0]
        P = np.eye(n) # comentar aca
        A_permutada = A.copy()

        for k in range(n):
            #Tomamos los valores de la columna k desde la fila k  hasta el final
            columna = A_permutada[k:, k]

            #Hacemos que todos los valores de la columna sean su absoluto
            largo_columna_abs = np.abs(columna)

            #Buscamos el indice de la columna al que le pertenece el valor mas grande
            max_indice_columna = 0
            maxValor = largo_columna_abs[0]

            for i in range(1, len(columna)):

                if largo_columna_abs[i] > maxValor:
                    maxValor = largo_columna_abs[i]
                    max_indice_columna = i

            #Calculamos el indice correcto de la fila en A
            p = k + max_indice_columna


            # Intercambiamos filas en A_permutada y en P si es necesario
            if p != k:

                #Intercambiamos en A_copia
                A_permutada[[k, p], :] = A_permutada[[p, k], :]

                #Intercambiamos en P
                P[[k, p], :] = P[[p, k], :]

        return P, A_permutada


    P, A_permutada = construir_P(A) #Consigo la P, y en caso de que P != I la A con la filas reordenadas
    m = A.shape[0]
    n = A.shape[1]

    if m!=n:
        print('Matriz no cuadrada')
        return

    U = A_permutada # Comienza siendo una copia de A y deviene en U (triangulado superiormente)
    L = np.identity(n)  # comentar aca !!!



    for j in range(n):
        for i in range(j+1,n):
            L[i,j] = U[i,j] / U[j,j]
            U[i,:] = U[i,:] - L[i,j] * U[j,:]

    return L, U, P



# Calculo de la inversa usando descomposicion de LU para cualquier matriz inversible

def inversa_por_lu(A):
    n = A.shape[0]

    # Realizamos la factorización LU de la matriz A
    L, U, P = calculaLU(A)

    # Inicializamos la matriz identidad I
    I = np.eye(n)
    A_inv = np.zeros_like(A, dtype=float) #comentar aca !!!!

    # Resolvemos para cada columna de la matriz inversa
    for i in range(n):

        b = I[:, i]  # La columna i de la identidad

        # Resolvemos L y U
        y = solve_triangular(L, P @ b, lower=True)

        x = solve_triangular(U, y)

        A_inv[:, i] = x  # Guardamos el resultado en la columna i de A_inv

    return A_inv




def calcula_matriz_C(A):

    # Primero creo K a partir de la matriz A

    def crearK (A):

        n = A.shape[0]
        m = A.shape[1]
        K = np.zeros((m, n))
        sumaFilasA = np.sum(A, axis = 1)


        for i in range (len (sumaFilasA)):
            K[i, i] = sumaFilasA[i]

        return K

    # Nuestra matriz de transicion esta definida por A_traspuesta y K_inv, como nos pide la ecuacion (2)

    A_traspuesta = np.transpose(A) # Trasponemos A
    K = crearK(A) # Creamos K
    K_inv = inversa_por_lu(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = A_traspuesta @ K_inv # Calcula C multiplicando A_traspuesta y K_inv

    return C

def calcula_matriz_C_continua(D):


    def calcula_Cji(D, j , i):
        N = D.shape[0]
        num = 1 / D[i,j]
        den = 0

        for k in range(1, N):
            if k!= i:
                den = den + (1 / D[i,k])

        return num / den

    n = D.shape[0]
    C = np.zeros((n,n))

    for j in range(n):
        for i in range(n):
            if i != j:
              C[j,i] = calcula_Cji(D, j, i)

    return C

def calcula_B(C,r):
    n = C.shape[0]
    B = np.eye(n)

    for k in range(1,r):
        C_elevada_k = np.linalg.matrix_power(C,k)
        B = B + C_elevada_k

    return B

def nro_condicion_B(C,r):
    
    B = calcula_B(C, r)
    B_inv = inversa_por_lu(B)
    
    norma1_de_B = np.linalg.norm(B, 1)
    norma1_de_B_inv = np.linalg.norm(B_inv, 1)
    
    return norma1_de_B @ norma1_de_B_inv

r = 3
C = calcula_matriz_C_continua(D)
B = calcula_B(C,r)

B_inversa = inversa_por_lu(B)


visitas = [1159, 1078, 1137, 1073, 1097, 1083, 1096, 1088, 1126, 1083, 1151, 1105, 1110, 1102, 1110, 1170, 1084, 1120, 1120, 1106, 1160, 1146, 1073, 1087, 1056, 1113, 1149, 1082, 1088, 1053, 1115, 1115, 1146, 1133, 1137, 1090, 1092, 1031, 1085, 1103, 1077, 1128, 1172, 1116, 1130, 1085, 1013, 1173, 1120, 1081, 1116, 1052, 1118, 1070, 1087, 1089, 1125, 1034, 1105, 1124, 1117, 1090, 1103, 1163, 1076, 1086, 1063, 1138, 1120, 1085, 1053, 1105, 1128, 1094, 1084, 1123, 1052, 1070, 1069, 1119, 1194, 1085, 1124, 1089, 1101, 1090, 1096, 1176, 1115, 1061, 1134, 1049, 1097, 1069, 1068, 1081, 1094, 1075, 1110, 1129, 1115, 1086, 1083, 1128, 1097, 1124, 1100, 1118, 1073, 1121, 1127, 1126, 1086, 1073, 1060, 1083, 1077, 1037, 1072, 1046, 1161, 1138, 1104, 1108, 1086, 1081, 1094, 1131, 1116, 1116, 1168, 1115, 1110, 1118, 1156, 1124]

    
    
    
    