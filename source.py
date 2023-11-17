import numpy as np
from numpy import random

def matrizAleatoria(n):
    m = random.randint(low=1, high=1000, size=(n, n))
    return (np.tril(m, -1) + np.tril(m, -1).T)

def minDistintoDeCero(l):
    # debemos devolver el índice del mínimo distinto de cero
    zeroIndex = 0
    for i in range(len(l) - 1):
        if l[i] == 0:
            zeroIndex = i
            break # asumimos que solo hay uno y nos quedamos con el primero
    l = np.delete(l, zeroIndex)
    minIndex = np.where(l == min(l))[0][0]
    # hay que coger los índices así porque si no es un objeto array
    
    if minIndex >= zeroIndex:
        minIndex += 1
        # si estaba a la derecha del cero, hay que compensar y sumarle 1
    
    return minIndex

def dijkstra(matriz):
    n = len(matriz)
    # generar la matriz Distancias con el
    # tamaño correcto, que es lo que vamos a devolver
    distancias = np.zeros((n, n))
        
    for m in range(n):
        noVisitados = np.arange(n)
        noVisitados = np.delete(noVisitados, m)

        for i in range(n):
            distancias[m][i] = matriz[m][i]
        
        for i in range(n - 1):
            # "v es el nodo que tiene la menor distancia a m Y que no esté visitado"
            # EL PROBLEMA ES QUE EL MÍNIMO que estamos cogiendo, ya está en novisitados
            # despues de la primera iteracion. falta la segunda condición
            fila = distancias[m]
            v = minDistintoDeCero(fila) # índice, no valor
            print()
            print(f'fila es {fila}')
            print(f'noVisitados es {noVisitados}')
            print(f'el índice del mínimo es {v}')
            print()
            
            noVisitados = np.delete(noVisitados, v - 1)
            for w in noVisitados:
                if distancias[m][w] > distancias[m][v] + matriz[v][w]:
                    distancias[m][w] = distancias[m][v] + matriz[v][w]
    return distancias

matrizOriginal = matrizAleatoria(10)
resultado = dijkstra(matrizOriginal)

print(f'la matriz original era {matrizOriginal}')
print(f'el resultado fue {resultado}')