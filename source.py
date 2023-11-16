import numpy as np
from numpy import random

def matrizAleatoria(n):
    m = random.randint(low=1, high=1000, size=(n, n))
    return (np.tril(m, -1) + np.tril(m, -1).T)

def minDistintoDeCero(l):
    res = min(l)
    if min 
    pass

def dijkstra(matriz):
    print(matriz)
    n = len(matriz)
    # generar la matriz Distancias con el
    # tamaño correcto, que es lo que vamos a devolver
    distancias = np.zeros((n, n))
        
    for m in range(n):
        noVisitados = np.arange(n)
        noVisitados = np.delete(noVisitados, m)

        for i in range(n):
            distancias[m][i] = matriz[m][i]
        
        for i in range(n-1):
            # v es el nodo que tiene la menor distancia a m Y que no esté visitado
            fila = distancias[m]
            v = 00000000000000
            print(v)
            
        print('----')

#dijkstra(matrizAleatoria(5))

lista = [0,1,2,3,4]