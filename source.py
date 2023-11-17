import numpy as np
from numpy import random

def matrizAleatoria(n):
    m = random.randint(low=1, high=1000, size=(n, n))
    return (np.tril(m, -1) + np.tril(m, -1).T)

def minDistintoDeCero(l):
    l2 = l.copy()
    for i in range(len(l2) - 1):
        if l2[i] == 0:
            l2[i] = np.nan
    result = int(np.nanmin(l2))
    resultIndex = np.where(l==result)[0][0]
    return resultIndex

def dijkstra(matriz):
    n = len(matriz)
    # generar la matriz Distancias con el
    # tamaño correcto, que es lo que vamos a devolver
    distancias = np.zeros((n, n))
        
    for m in range(n):
        noVisitados = np.arange(n)
        noVisitados = np.delete(noVisitados, m)

        for i in range(n):
            # sobreescribe la fila m de distancias con la fila m de la matriz
            distancias[m][i] = matriz[m][i]
        
        print('---------')
        for i in range(n - 1):
            # "v es el nodo que tiene la menor distancia a m Y que no esté visitado"
            
            fila = distancias[m]
            v = minDistintoDeCero(fila) # índice, no valor
            
            print()
            print(f'fila es        {fila}')
            print(f'noVisitados es {noVisitados}')
            print(f'el índice del mínimo es {v}')
            print()                  
                        
            noVisitados = np.delete(noVisitados, np.argwhere(noVisitados == v))
            
            for w in noVisitados:
                if distancias[m][w] > distancias[m][v] + matriz[v][w]:
                    distancias[m][w] = distancias[m][v] + matriz[v][w]
    return distancias

def test(): # cubrir con los ejemplos del pdf
    # primer ejemplo
    print('primer ejemplo')
    matrizOriginal = np.array([[0,1,8,4,7],
                               [1,0,2,6,5],
                               [8,2,0,9,5],
                               [4,6,9,0,3],
                               [7,5,5,3,0]])
    
    solucion = np.array([[0,1,3,4,6],
                         [1,0,2,5,5],
                         [3,2,0,7,5],
                         [4,5,7,0,3],
                         [6,5,5,3,0]])
    
    resultado = dijkstra(matrizOriginal)
    
    print('la matriz original era')
    print(matrizOriginal)
    
    print('la solución correcta es')
    print(solucion)

    print('el resultado es')
    print(resultado)
    
    # segundo ejemplo
    print('segundo ejemplo')
    matrizOriginal = np.array([[0,1,4,7],
                               [1,0,2,8],
                               [4,2,0,3],
                               [7,8,3,0]])
    
    solucion = np.array([[0,1,3,6],
                         [1,0,2,5],
                         [3,2,0,3],
                         [6,5,3,0]])
    
    resultado = dijkstra(matrizOriginal)
    
    print('la matriz original era')
    print(matrizOriginal)
    
    print('la solución correcta es')
    print(solucion)

    print('el resultado es')
    print(resultado)

test()