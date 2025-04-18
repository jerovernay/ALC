# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
from scipy.linalg import lu, solve_triangular




#Para graficar la red con un conjunto de puntajes (como el Page Rank)

# factor_escala = 1e4 # Escalamos los nodos 10 mil veces para que sean bien visibles
# fig, ax = plt.subplots(figsize=(10, 10)) # Visualización de la red en el mapa
# barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
# pR = np.random.uniform(0,1,museos.shape[0])# Este va a ser su score Page Rank. Ahora lo reemplazamos con un vector al azar !!!
# pR = pR/pR.sum() # Normalizamos para que sume 1
# Nprincipales = 5 # Cantidad de principales
# principales = np.argsort(pR)[-Nprincipales:] # Identificamos a los N principales
# labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos
# nx.draw_networkx(G,G_layout,node_size = pR*factor_escala, ax=ax,with_labels=False) # Graficamos red
# nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k") # Agregamos los nombres

#Ejemplo Claudio

# Calcular PageRank para m=3 y alpha=1/5
m = 3
alpha = 1/5
N = museos.shape[0]  # número de museos

# Construir matriz de adyacencia
A = construye_adyacencia(D, m)

# Calcular PageRank
page_rank = calculo_Page_Rank(A, alpha, N)

# Crear el grafo
G = nx.from_numpy_array(A)
G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],
                                          museos.to_crs("EPSG:22184").get_coordinates()['y']))}

# Identificar los 3 museos con mayor PageRank
N_principales = 3
principales = np.argsort(page_rank)[-N_principales:]


# Crear figura y graficar
fig, ax = plt.subplots(figsize=(10, 10))
barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)

# Factor de escala ajustado para visualizar mejor
factor_escala = 1e4

# Etiquetar solo los N principales museos
labels = {n: f"Museo {n}\nRank: {page_rank[i]:.4f}" if i in principales else ""
          for i, n in enumerate(G.nodes)}

# Dibujar la red
nx.draw_networkx(G, G_layout,
                 node_size=page_rank*factor_escala,  # Tamaño proporcional al PageRank
                 node_color=page_rank,  # Color según PageRank
                 cmap=plt.cm.viridis,  # Usar un mapa de colores
                 with_labels=False,
                 ax=ax)

# Agregar etiquetas solo para los principales
nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=10, font_color="red")

# Añadir título y leyenda
plt.title(f'Red de Museos - PageRank (m={m}, α={alpha})')
plt.axis('off')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Imprimir información sobre los 3 museos principales
print("Los 3 museos con mayor PageRank son:")
for i, idx in enumerate(principales[::-1]):  # Invertir para mostrar en orden descendente
    print(f"{i+1}. Museo {idx}: PageRank = {page_rank[idx]:.6f}")