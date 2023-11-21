import random, time
import numpy as np
from prettytable import PrettyTable
from line_profiler import LineProfiler

def calcular_tiempo(func, m):
    start = time.perf_counter_ns()
    func(m)
    finish = time.perf_counter_ns()
    t = finish - start
    # solo se ejecuta si no cumplimos con el umbral de confianza
    if t < 500000: # 500 microsegundos = 500000 ns
        n = len(m)
        k = 100
        
        start = time.perf_counter_ns()
        for i in range(k):
            matriz = matrizAleatoria(n)
            func(matriz)
        finish = time.perf_counter_ns()
        t1 = finish - start

        start = time.perf_counter_ns()
        for i in range(k):
            matriz = matrizAleatoria(n)
        finish = time.perf_counter_ns()
        t2 = finish - start

        t = (t1 - t2) / k
        if t < 0:
            print(f'///// Valor negativo con n={n}, anomalía.')

    return t

# ejercicio 1
def matrizAleatoria(n):
    m = np.random.randint(low=1, high=1000, size=(n, n))
    return (np.tril(m, -1) + np.tril(m, -1).T)

def minDistintoDeCero(l):
    l = l.copy()
    l[l == 0] = np.nan # sustituimos los ceros por NaNs
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
        # y quitamos m, ya que no se puede visitar a sí mismo
        noVisitados = np.delete(noVisitados, m)

        for i in range(n):
            # sobreescribimos la fila m de distancias con la fila m de la matriz
            distancias[m][i] = matriz[m][i]
        
        for i in range(n - 2):
            # para crear 'fila', se crea un array de ceros del tamaño adecuado
            # y sustituimos por los valores reales que tenemos en 'distancias'
            # solo si están en noVisitados. no nos interesa buscar
            # el mínimo entre elementos que ya hemos visitado
            fila = np.zeros(n)
            fila[noVisitados] = distancias[m, noVisitados]
            
            # obtenemos los índices de los elementos distintos que cero
            nonzero_indices = np.nonzero(fila)[0]
            # obtenemos el índice del mínimo en la lista de índices
            v = nonzero_indices[np.argmin(fila[nonzero_indices])]

            # lo borramos de 'noVisitados', porque lo acabamos de visitar
            noVisitados = noVisitados[noVisitados != v]
            
            # guardamos los índices de los nodos no visitados que tienen un
            # camino más corto que pasa por el mínimo en una máscara de numpy
            mask = distancias[m, noVisitados] > (distancias[m, v] + matriz[v, noVisitados])
            # después los sustituimos
            distancias[m, noVisitados[mask]] = distancias[m, v] + matriz[v, noVisitados[mask]]
            
            '''
            teníamos una versión idéntica a la del pseudocódigo pero esta
            comparación era significativamente más lenta/costosa
            
            for w in noVisitados:
                if distancias[m][w] > distancias[m][v] + matriz[v][w]:
                    distancias[m][w] = distancias[m][v] + matriz[v][w]
            '''
            
    return distancias.astype(int) # para que los valores sean int y no float

# ejercicio 2
# TODO: implementar MÁS casos que los del pdf
def test():
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

print('\n///// Ejercicio 2')
test()

# ejercicio 3
# TODO: cálculo empírico de la complejidad
print('\n///// Ejercicio 3')

start_time = time.time()
sizes = [2**i for i in range(9+1)]
table = PrettyTable()
table.title = 'Matrices de adyacencia aleatorias con n vértices'
table.field_names = ['n', 't(n)(ns)', 't(n)/n**2', 't(n)/n**2.5', 't(n)/n**3']

for n in sizes:
    # matriz aleatoria
    matriz = matrizAleatoria(n)
    executionTime = calcular_tiempo(dijkstra, matriz)
    table.add_row([n, 
                 executionTime,                     # la sintaxis de "%.nf"      
                 "%.2f" % (executionTime / n**2),   # sirve para redondear
                 "%.2f" % (executionTime / n**2.5), # a exactamente n decimales,
                 "%.2f" % (executionTime / n**3)])  # sean ceros o no

table.align = 'r' # alineamos la tabla a la derecha
print(table)
print('Tiempo total de ejecución del ejercicio 3 (en segundos):',
      round(time.time() - start_time, 2))

lp = LineProfiler()                     # creamos un objeto LineProfiler()     
lp.add_function(dijkstra)               # y le añadimos funciones para ver
lp.run('dijkstra(matrizAleatoria(64))') # el tiempo de ejecución por línea
lp.print_stats()                        # (y lo imprimimos)