import numpy as np
from numpy import random

def matrizAleatoria(n):
    m = random.randint(low=1, high=1000, size=(n, n))
    return (np.tril(m, -1) + np.tril(m, -1).T)

def minDistintoDeCero(l):
    for i in range(len(l)): # sustituimos los 0s por NaNs
        if l[i] == 0:
            l[i] = np.nan
    result = int(np.nanmin(l)) # obtenemos el mínimo ignorando los NaNs
    resultIndex = np.where(l==result)[0][0] # obtenemos su índice
    return resultIndex

def dijkstra(matriz):
    n = len(matriz) # necesitamos el tamaño para operar más tarde
    # generar la matriz Distancias con el
    # tamaño correcto, que es lo que vamos a devolver
    distancias = np.zeros((n, n))
        
    for m in range(n):
        # enumeramos todos los nodos
        noVisitados = np.arange(n)
        # y quitamos m, que es sobre el que estamos operando
        noVisitados = np.delete(noVisitados, m)

        for i in range(n):
            # sobreescribimos la fila m de distancias con la fila m de la matriz
            distancias[m][i] = matriz[m][i]
        
        for i in range(n - 1):
            # "v es el nodo que tiene la menor distancia a m Y que no esté visitado"
            
            # para crear 'fila', se crea una matriz de ceros del tamaño adecuado
            # y sustituimos por los valores reales que tenemos en 'distancias'
            # solo si no los hemos visitado. no nos interesa buscar el mínimo 
            # considerando elementos que ya hemos visitado
            fila = np.zeros(n)
            for i in range(n):
                if i in noVisitados:
                    fila[i] = distancias[m][i]
            
            # obtenemos el índice del mínimo ignorando los ceros
            v = minDistintoDeCero(fila)
            # lo borramos de 'noVisitados', porque lo acabamos de visitar
            noVisitados = np.delete(noVisitados, np.argwhere(noVisitados == v))
            
            for w in noVisitados:
                if distancias[m][w] > distancias[m][v] + matriz[v][w]:
                    distancias[m][w] = distancias[m][v] + matriz[v][w]
    return distancias.astype(int) # para que los valores sean int y no float

def test():
    # primer ejemplo
    print()
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
    print()
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