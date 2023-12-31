import time
import numpy as np
from prettytable import PrettyTable
from line_profiler import LineProfiler

def calcular_tiempo(func, m):
    # medimos el tiempo de la función que se le pasa, y si está por debajo del
    # umbral de confianza, medimos la media de k iteraciones
    start = time.perf_counter_ns()
    func(m)
    finish = time.perf_counter_ns()
    t = finish - start
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

def fM(m):
    # esta función solo está para que las matrices queden bien alineadas
    # cuando se imprimen. para ello, le sumamos un espacio y le quitamos el
    # primer y último caracter (corchetes)
    return ' ' + str(m)[1:-1] 

# ejercicio 1
print('''
****Ejercicio 1****
En este ejercicio tan solo implementamos las funciones matrizAleatoria() y
dijkstra() en el código de Python.
''')

def matrizAleatoria(n):
    # inicializamos una matriz de tamaño n x n con pesos aleatorios
    # entre 1 y 1000. después, cogemos su triangular inferior y le sumamos la
    # propia traspuesta para que sea simétrica y sea equivalente a la matriz de
    # adyacencia de un grafo completo. la diagonal es 0
    m = np.random.randint(low=1, high=1000, size=(n, n))
    return (np.tril(m, -1) + np.tril(m, -1).T)

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
            distancias[m, i] = matriz[m, i]
        
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
            
            # para cada nodo en la fila comprobamos si el camino
            # a través del mínimo es más corto
            for w in noVisitados:
                if distancias[m][w] > distancias[m][v] + matriz[v][w]:
                    distancias[m][w] = distancias[m][v] + matriz[v][w]
            
    return distancias.astype(int) # para que los valores sean int y no float

# ejercicio 2
def test():
    tablaPrimerEjemplo = PrettyTable()
    tablaPrimerEjemplo.title = 'Primer ejemplo'
    tablaPrimerEjemplo.field_names = ['Matriz original', 'Solución',
                                      'Resultado', '¿Funciona?']
    
    mOriginal = np.array([[0,1,8,4,7],
                          [1,0,2,6,5],
                          [8,2,0,9,5],
                          [4,6,9,0,3],
                          [7,5,5,3,0]])
    solucion = np.array([[0,1,3,4,6],
                         [1,0,2,5,5],
                         [3,2,0,7,5],
                         [4,5,7,0,3],
                         [6,5,5,3,0]])
    resultado = dijkstra(mOriginal)
    tablaPrimerEjemplo.add_row([fM(mOriginal), fM(solucion), fM(resultado),
                                ('✓' if solucion.all() == resultado.all()
                                     else 'x')])
    print(tablaPrimerEjemplo)
    
    tablaSegundoEjemplo = PrettyTable()
    tablaSegundoEjemplo.title = 'Segundo ejemplo'
    tablaSegundoEjemplo.field_names = ['Matriz original', 'Solución',
                                       'Resultado', '¿Funciona?']
    
    mOriginal = np.array([[0,1,4,7],
                          [1,0,2,8],
                          [4,2,0,3],
                          [7,8,3,0]])
    solucion = np.array([[0,1,3,6],
                         [1,0,2,5],
                         [3,2,0,3],
                         [6,5,3,0]])
    resultado = dijkstra(mOriginal)
    tablaSegundoEjemplo.add_row([fM(mOriginal), fM(solucion), fM(resultado),
                                ('✓' if solucion.all() == resultado.all()
                                     else 'x')])
    print(tablaSegundoEjemplo)

print('\n**** Ejercicio 2****')
test()

# ejercicio 3
print('\n**** Ejercicio 3****')

start_time = time.time()
n = 8 # empezamos en 2^3
table = PrettyTable()
table.title = 'Matrices de adyacencia aleatorias con n vértices'
table.field_names=['n','t(n)(ns)','t(n)/n**2.75','t(n)/n**3','t(n)/n**3.25']

while True:
    # matriz aleatoria
    matriz = matrizAleatoria(n)
    executionTime = calcular_tiempo(dijkstra, matriz)
    table.add_row([n,
                executionTime,
                "%.2f" % (executionTime / n**2.75),
                "%.2f" % (executionTime / n**3),
                "%.2f" % (executionTime / n**3.25)])
    # la sintaxis de "%.nf" sirve para redondear a n decimales, sean ceros o no
    n *= 2
    if executionTime > 10**9: # si nos hemos pasado de un segundo, salimos
        break

table.align = 'r' # alineamos la tabla a la derecha
print(table)
print('Tiempo total de ejecución del ejercicio 3:',
      f'{round(time.time() - start_time, 2)} segundos.')