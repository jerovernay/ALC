import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def proyectarPts(ptos, T):
    assert(T.shape == (3,3) or T.shape == (2,2)) # chequeo de matriz  o 3x3
    assert(T.shape[1] == ptos.shape[0]) # multiplicacion matricial valida   
    xy = None
        
    xy = np.matmul(T,ptos)

    return xy

def pointsGrid(corners):
    
    """
Crea una grilla de puntos para visualizar transformaciones.

Parámetros:
corners -- array con las esquinas [[x_min, y_min], [x_max, y_max]]

Retorna:
Array de 2 x n con los puntos de la grilla
    """
    
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(corners[0,0], corners[1,0], 46),
                        np.linspace(corners[0,1], corners[1,1], 10))

    [w2, z2] = np.meshgrid(np.linspace(corners[0,0], corners[1,0], 10),
                        np.linspace(corners[0,1], corners[1,1], 46))

    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
                         
    return wz
          
def grid_plot(ax, ab, limits, a_label, b_label):
    
    """
    Dibuja una grilla de puntos en el eje especificado.
    """
    
    ax.plot(ab[0,:], ab[1,:], '.')
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)
    
    
def vistform(T, wz, titulo=''):
    
    """
    Visualiza una transformación lineal.
    Muestra los puntos originales y transformados lado a lado.

    Parámetros:
        T -- matriz de transformación 2x2
        wz -- puntos originales (2xn)
        titulo -- título del gráfico

    """
    
    # transformar los puntos de entrada usando T
    xy = proyectarPts(wz, T)
    
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
               [minlim[1]-bump[1], maxlim[1]+bump[1]]]             

    # Crea los subplots para mostrar la diferencia entre el original y el transformado

    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    
    # Dibujar los puntos originales y transformadoa
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')    
    
    #Mostrar la matriz de transformacion en el titulo
    
    matrix_str = f"T = [[[{T[0,0]}, {T[0,1]}], [{T[1,0]}, {T[1,1]}]] "
    plt.figtext(0.5, 0.1, matrix_str, ha ='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)    #Hacer espacio para el texto de la matriz
    plt.show()
    
def visualizar_transformacion(T, region= None):
    """
    Visualiza una transformación lineal en una región específica.

    Parámetros:
        T -- matriz de transformación 2x2
        region -- región a visualizar (por defecto [0,0] a [100,100])
    """
    
    if region is None:
        corners = np.array([[0, 0], [100, 100]])
    else:
        corners = np.array(region)
        
    # Generar puntos de la grilla
    wz = pointsGrid(corners)
    
    # Visualizar la transformación
    vistform(T, wz, f'Transformación Lineal: {T[0,0]:.1f}x{T[1,1]:.1f} + {T[0,1]:.1f}z')


def main():
    print('Ejecutar el programa de transformacion lineal')
    
    # generar el tipo de transformacion dando valores a la matriz T
    T = pd.read_csv('T.csv', header=None).values
    corners = np.array([[0,0],[100,100]])
    # corners = np.array([[-100,-100],[100,100]]) array con valores positivos y negativos
    wz = pointsGrid(corners)
    vistform(T, wz, 'Deformar coordenadas')
    
    
if __name__ == "__main__":
    main()
