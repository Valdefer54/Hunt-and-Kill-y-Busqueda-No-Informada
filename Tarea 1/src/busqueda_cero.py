"""
Versión 1: Algoritmos de Búsqueda No Informada Implementados Desde Cero.
Sin bibliotecas externas de grafos ni de búsqueda.
Algoritmos: DFS, BFS, UCS, DLS, IDDFS, Bidireccional y Lee.
"""

from collections import deque
import heapq
import time
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from src.comun import ACCIONES_DELTA, DELTA_ACCIONES, Nodo, ResultadoBusqueda


class ProblemaLaberinto:
    """Modela el entorno del laberinto como un espacio de estados formal."""

    def __init__(
        self,
        grafo: Dict[Tuple[int, int], Set[Tuple[int, int]]],
        inicio: Tuple[int, int],
        meta: Tuple[int, int],
        costo_arista_fn: Optional[Callable[[Tuple[int, int], Tuple[int, int]], float]] = None,
    ):
        self.grafo = grafo
        self.inicio = inicio
        self.meta = meta
        # Función de costo de paso; por defecto costo unitario 1.0
        self.costo_arista_fn = costo_arista_fn or (lambda u, v: 1.0)

    def acciones(self, estado: Tuple[int, int]) -> List[str]:
        """Devuelve las acciones legales disponibles desde un estado."""
        r, c = estado
        acciones_legales = []
        # Orden determinista: ARRIBA, ABAJO, IZQUIERDA, DERECHA
        for nombre_acc, (dr, dc) in ACCIONES_DELTA.items():
            vecino = (r + dr, c + dc)
            if vecino in self.grafo.get(estado, set()):
                acciones_legales.append(nombre_acc)
        return acciones_legales

    def resultado(self, estado: Tuple[int, int], accion: str) -> Tuple[int, int]:
        """Calcula el estado sucesor dado un estado y una acción."""
        dr, dc = ACCIONES_DELTA[accion]
        return (estado[0] + dr, estado[1] + dc)

    def es_meta(self, estado: Tuple[int, int]) -> bool:
        """Comprueba si el estado dado es el estado meta."""
        return estado == self.meta

    def costo_paso(self, estado1: Tuple[int, int], accion: str, estado2: Tuple[int, int]) -> float:
        """Devuelve el costo de transición entre dos estados adyacentes."""
        return self.costo_arista_fn(estado1, estado2)


# ==============================================================================
# 1. BÚSQUEDA EN PROFUNDIDAD (DFS ITERATIVA)
# ==============================================================================
def dfs_cero(problema: ProblemaLaberinto) -> ResultadoBusqueda:
    t0 = time.perf_counter()
    nodo_raiz = Nodo(estado=problema.inicio, padre=None, accion=None, costo_camino=0.0, profundidad=0)

    if problema.es_meta(problema.inicio):
        t1 = time.perf_counter()
        camino, acciones = nodo_raiz.reconstruir_camino()
        return ResultadoBusqueda(
            encontrado=True,
            camino=camino,
            acciones=acciones,
            costo=0.0,
            profundidad=0,
            expandidos=0,
            generados=1,
            repetidos_descartados=0,
            frontera_maxima=1,
            tiempo_ms=(t1 - t0) * 1000.0,
        )

    pila: List[Nodo] = [nodo_raiz]
    alcanzados: Set[Tuple[int, int]] = {problema.inicio}
    expandidos = 0
    generados = 1
    repetidos_descartados = 0
    frontera_maxima = 1

    while pila:
        nodo = pila.pop()

        if problema.es_meta(nodo.estado):
            t1 = time.perf_counter()
            camino, acciones = nodo.reconstruir_camino()
            return ResultadoBusqueda(
                encontrado=True,
                camino=camino,
                acciones=acciones,
                costo=nodo.costo_camino,
                profundidad=nodo.profundidad,
                expandidos=expandidos,
                generados=generados,
                repetidos_descartados=repetidos_descartados,
                frontera_maxima=frontera_maxima,
                tiempo_ms=(t1 - t0) * 1000.0,
            )

        expandidos += 1
        # Invertimos acciones para explorar en orden canónico (ARRIBA, ABAJO, ...) al hacer pop
        for accion in reversed(problema.acciones(nodo.estado)):
            sucesor_estado = problema.resultado(nodo.estado, accion)
            costo = nodo.costo_camino + problema.costo_paso(nodo.estado, accion, sucesor_estado)
            hijo = Nodo(
                estado=sucesor_estado,
                padre=nodo,
                accion=accion,
                costo_camino=costo,
                profundidad=nodo.profundidad + 1,
            )
            generados += 1

            if sucesor_estado not in alcanzados:
                alcanzados.add(sucesor_estado)
                pila.append(hijo)
            else:
                repetidos_descartados += 1

        if len(pila) > frontera_maxima:
            frontera_maxima = len(pila)

    t1 = time.perf_counter()
    return ResultadoBusqueda(
        encontrado=False,
        camino=[],
        acciones=[],
        costo=0.0,
        profundidad=0,
        expandidos=expandidos,
        generados=generados,
        repetidos_descartados=repetidos_descartados,
        frontera_maxima=frontera_maxima,
        tiempo_ms=(t1 - t0) * 1000.0,
    )


# ==============================================================================
# 2. BÚSQUEDA EN ANCHURA (BFS)
# ==============================================================================
def bfs_cero(problema: ProblemaLaberinto) -> ResultadoBusqueda:
    t0 = time.perf_counter()
    nodo_raiz = Nodo(estado=problema.inicio, padre=None, accion=None, costo_camino=0.0, profundidad=0)

    # Verificación anticipada de meta para la raíz
    if problema.es_meta(problema.inicio):
        t1 = time.perf_counter()
        camino, acciones = nodo_raiz.reconstruir_camino()
        return ResultadoBusqueda(
            encontrado=True,
            camino=camino,
            acciones=acciones,
            costo=0.0,
            profundidad=0,
            expandidos=0,
            generados=1,
            repetidos_descartados=0,
            frontera_maxima=1,
            tiempo_ms=(t1 - t0) * 1000.0,
        )

    cola: deque[Nodo] = deque([nodo_raiz])
    alcanzados: Set[Tuple[int, int]] = {problema.inicio}
    expandidos = 0
    generados = 1
    repetidos_descartados = 0
    frontera_maxima = 1

    while cola:
        nodo = cola.popleft()
        expandidos += 1

        for accion in problema.acciones(nodo.estado):
            sucesor_estado = problema.resultado(nodo.estado, accion)
            costo = nodo.costo_camino + problema.costo_paso(nodo.estado, accion, sucesor_estado)
            hijo = Nodo(
                estado=sucesor_estado,
                padre=nodo,
                accion=accion,
                costo_camino=costo,
                profundidad=nodo.profundidad + 1,
            )
            generados += 1

            if sucesor_estado not in alcanzados:
                # Verificación anticipada de meta (estándar para BFS no ponderado)
                if problema.es_meta(sucesor_estado):
                    t1 = time.perf_counter()
                    camino, acciones = hijo.reconstruir_camino()
                    return ResultadoBusqueda(
                        encontrado=True,
                        camino=camino,
                        acciones=acciones,
                        costo=hijo.costo_camino,
                        profundidad=hijo.profundidad,
                        expandidos=expandidos,
                        generados=generados,
                        repetidos_descartados=repetidos_descartados,
                        frontera_maxima=frontera_maxima,
                        tiempo_ms=(t1 - t0) * 1000.0,
                    )
                alcanzados.add(sucesor_estado)
                cola.append(hijo)
            else:
                repetidos_descartados += 1

        if len(cola) > frontera_maxima:
            frontera_maxima = len(cola)

    t1 = time.perf_counter()
    return ResultadoBusqueda(
        encontrado=False,
        camino=[],
        acciones=[],
        costo=0.0,
        profundidad=0,
        expandidos=expandidos,
        generados=generados,
        repetidos_descartados=repetidos_descartados,
        frontera_maxima=frontera_maxima,
        tiempo_ms=(t1 - t0) * 1000.0,
    )


# ==============================================================================
# 3. BÚSQUEDA DE COSTO UNIFORME (UCS)
# ==============================================================================
def ucs_cero(problema: ProblemaLaberinto) -> ResultadoBusqueda:
    t0 = time.perf_counter()
    nodo_raiz = Nodo(estado=problema.inicio, padre=None, accion=None, costo_camino=0.0, profundidad=0)

    # Heap: elementos de la forma (costo, id_contador, nodo)
    contador = 0
    frontera: List[Tuple[float, int, Nodo]] = [(0.0, contador, nodo_raiz)]
    mejor_costo: Dict[Tuple[int, int], float] = {problema.inicio: 0.0}

    expandidos = 0
    generados = 1
    repetidos_descartados = 0
    frontera_maxima = 1

    while frontera:
        g, _, nodo = heapq.heappop(frontera)

        # Lazy deletion: ignorar entradas obsoletas si ya encontramos un camino más barato
        if g > mejor_costo.get(nodo.estado, float("inf")):
            continue

        # En UCS, la meta DEBE comprobarse al expandir (late goal test) para garantizar optimalidad
        if problema.es_meta(nodo.estado):
            t1 = time.perf_counter()
            camino, acciones = nodo.reconstruir_camino()
            return ResultadoBusqueda(
                encontrado=True,
                camino=camino,
                acciones=acciones,
                costo=nodo.costo_camino,
                profundidad=nodo.profundidad,
                expandidos=expandidos,
                generados=generados,
                repetidos_descartados=repetidos_descartados,
                frontera_maxima=frontera_maxima,
                tiempo_ms=(t1 - t0) * 1000.0,
            )

        expandidos += 1

        for accion in problema.acciones(nodo.estado):
            sucesor_estado = problema.resultado(nodo.estado, accion)
            nuevo_costo = nodo.costo_camino + problema.costo_paso(nodo.estado, accion, sucesor_estado)
            hijo = Nodo(
                estado=sucesor_estado,
                padre=nodo,
                accion=accion,
                costo_camino=nuevo_costo,
                profundidad=nodo.profundidad + 1,
            )
            generados += 1

            if sucesor_estado not in mejor_costo or nuevo_costo < mejor_costo[sucesor_estado]:
                mejor_costo[sucesor_estado] = nuevo_costo
                contador += 1
                heapq.heappush(frontera, (nuevo_costo, contador, hijo))
            else:
                repetidos_descartados += 1

        if len(frontera) > frontera_maxima:
            frontera_maxima = len(frontera)

    t1 = time.perf_counter()
    return ResultadoBusqueda(
        encontrado=False,
        camino=[],
        acciones=[],
        costo=0.0,
        profundidad=0,
        expandidos=expandidos,
        generados=generados,
        repetidos_descartados=repetidos_descartados,
        frontera_maxima=frontera_maxima,
        tiempo_ms=(t1 - t0) * 1000.0,
    )


# ==============================================================================
# 4. BÚSQUEDA LIMITADA EN PROFUNDIDAD (DLS)
# ==============================================================================
# Señales para distinguir corte de fracaso
CORTE = "CORTE"
FRACASO = "FRACASO"


def dls_cero(
    problema: ProblemaLaberinto,
    limite: int,
) -> Tuple[Union[Nodo, str], Dict[str, int]]:
    """
    Búsqueda limitada en profundidad iterativa.
    Devuelve (Nodo|'CORTE'|'FRACASO', metricas).
    """
    nodo_raiz = Nodo(estado=problema.inicio, padre=None, accion=None, costo_camino=0.0, profundidad=0)

    if problema.es_meta(problema.inicio):
        return nodo_raiz, {
            "expandidos": 0,
            "generados": 1,
            "repetidos_descartados": 0,
            "frontera_maxima": 1,
        }

    # Pila de tuplas: (nodo, iterador_acciones)
    pila: List[Nodo] = [nodo_raiz]
    # Control de camino actual para evitar ciclos en la rama
    camino_actual: Set[Tuple[int, int]] = {problema.inicio}
    mejor_profundidad_alcanzada: Dict[Tuple[int, int], int] = {problema.inicio: 0}

    expandidos = 0
    generados = 1
    repetidos_descartados = 0
    frontera_maxima = 1
    hubo_corte = False

    while pila:
        nodo = pila.pop()

        if problema.es_meta(nodo.estado):
            return nodo, {
                "expandidos": expandidos,
                "generados": generados,
                "repetidos_descartados": repetidos_descartados,
                "frontera_maxima": frontera_maxima,
            }

        if nodo.profundidad >= limite:
            hubo_corte = True
            continue

        expandidos += 1
        for accion in reversed(problema.acciones(nodo.estado)):
            sucesor = problema.resultado(nodo.estado, accion)
            nueva_prof = nodo.profundidad + 1
            costo = nodo.costo_camino + problema.costo_paso(nodo.estado, accion, sucesor)
            hijo = Nodo(
                estado=sucesor,
                padre=nodo,
                accion=accion,
                costo_camino=costo,
                profundidad=nueva_prof,
            )
            generados += 1

            if sucesor not in mejor_profundidad_alcanzada or nueva_prof < mejor_profundidad_alcanzada[sucesor]:
                mejor_profundidad_alcanzada[sucesor] = nueva_prof
                pila.append(hijo)
            else:
                repetidos_descartados += 1

        if len(pila) > frontera_maxima:
            frontera_maxima = len(pila)

    metricas = {
        "expandidos": expandidos,
        "generados": generados,
        "repetidos_descartados": repetidos_descartados,
        "frontera_maxima": frontera_maxima,
    }
    return (CORTE if hubo_corte else FRACASO), metricas


def dls_wrapper(problema: ProblemaLaberinto, limite: int) -> ResultadoBusqueda:
    t0 = time.perf_counter()
    resultado, metricas = dls_cero(problema, limite)
    t1 = time.perf_counter()

    if isinstance(resultado, Nodo):
        camino, acciones = resultado.reconstruir_camino()
        return ResultadoBusqueda(
            encontrado=True,
            camino=camino,
            acciones=acciones,
            costo=resultado.costo_camino,
            profundidad=resultado.profundidad,
            expandidos=metricas["expandidos"],
            generados=metricas["generados"],
            repetidos_descartados=metricas["repetidos_descartados"],
            frontera_maxima=metricas["frontera_maxima"],
            tiempo_ms=(t1 - t0) * 1000.0,
        )
    return ResultadoBusqueda(
        encontrado=False,
        camino=[],
        acciones=[],
        costo=0.0,
        profundidad=0,
        expandidos=metricas["expandidos"],
        generados=metricas["generados"],
        repetidos_descartados=metricas["repetidos_descartados"],
        frontera_maxima=metricas["frontera_maxima"],
        tiempo_ms=(t1 - t0) * 1000.0,
    )


# ==============================================================================
# 5. PROFUNDIZACIÓN ITERATIVA (IDDFS)
# ==============================================================================
def iddfs_cero(problema: ProblemaLaberinto, max_limite: int = 2000) -> ResultadoBusqueda:
    t0 = time.perf_counter()
    total_expandidos = 0
    total_generados = 0
    total_repetidos = 0
    max_frontera_global = 0

    for limite in range(max_limite + 1):
        resultado, metricas = dls_cero(problema, limite)
        total_expandidos += metricas["expandidos"]
        total_generados += metricas["generados"]
        total_repetidos += metricas["repetidos_descartados"]
        if metricas["frontera_maxima"] > max_frontera_global:
            max_frontera_global = metricas["frontera_maxima"]

        if isinstance(resultado, Nodo):
            t1 = time.perf_counter()
            camino, acciones = resultado.reconstruir_camino()
            return ResultadoBusqueda(
                encontrado=True,
                camino=camino,
                acciones=acciones,
                costo=resultado.costo_camino,
                profundidad=resultado.profundidad,
                expandidos=total_expandidos,
                generados=total_generados,
                repetidos_descartados=total_repetidos,
                frontera_maxima=max_frontera_global,
                tiempo_ms=(t1 - t0) * 1000.0,
            )
        elif resultado == FRACASO:
            break

    t1 = time.perf_counter()
    return ResultadoBusqueda(
        encontrado=False,
        camino=[],
        acciones=[],
        costo=0.0,
        profundidad=0,
        expandidos=total_expandidos,
        generados=total_generados,
        repetidos_descartados=total_repetidos,
        frontera_maxima=max_frontera_global,
        tiempo_ms=(t1 - t0) * 1000.0,
    )


# ==============================================================================
# 6. BÚSQUEDA BIDIRECCIONAL
# ==============================================================================
def bidireccional_cero(problema: ProblemaLaberinto) -> ResultadoBusqueda:
    t0 = time.perf_counter()

    if problema.es_meta(problema.inicio):
        t1 = time.perf_counter()
        return ResultadoBusqueda(
            encontrado=True,
            camino=[problema.inicio],
            acciones=[],
            costo=0.0,
            profundidad=0,
            expandidos=0,
            generados=2,
            repetidos_descartados=0,
            frontera_maxima=2,
            tiempo_ms=(t1 - t0) * 1000.0,
        )

    nodo_inicio = Nodo(estado=problema.inicio, padre=None, accion=None, costo_camino=0.0, profundidad=0)
    nodo_meta = Nodo(estado=problema.meta, padre=None, accion=None, costo_camino=0.0, profundidad=0)

    cola_fwd = deque([nodo_inicio])
    cola_bwd = deque([nodo_meta])

    alcanzados_fwd: Dict[Tuple[int, int], Nodo] = {problema.inicio: nodo_inicio}
    alcanzados_bwd: Dict[Tuple[int, int], Nodo] = {problema.meta: nodo_meta}

    expandidos = 0
    generados = 2
    repetidos_descartados = 0
    frontera_maxima = 2

    nodo_encuentro_fwd: Optional[Nodo] = None
    nodo_encuentro_bwd: Optional[Nodo] = None

    while cola_fwd and cola_bwd:
        # Alternar de manera balanceada expandiendo el frente con menos elementos
        if len(cola_fwd) <= len(cola_bwd):
            # Paso hacia adelante
            nodo = cola_fwd.popleft()
            expandidos += 1
            for accion in problema.acciones(nodo.estado):
                sucesor = problema.resultado(nodo.estado, accion)
                costo = nodo.costo_camino + problema.costo_paso(nodo.estado, accion, sucesor)
                hijo = Nodo(
                    estado=sucesor,
                    padre=nodo,
                    accion=accion,
                    costo_camino=costo,
                    profundidad=nodo.profundidad + 1,
                )
                generados += 1

                if sucesor in alcanzados_bwd:
                    # ¡Colisión detectada!
                    nodo_encuentro_fwd = hijo
                    nodo_encuentro_bwd = alcanzados_bwd[sucesor]
                    break

                if sucesor not in alcanzados_fwd:
                    alcanzados_fwd[sucesor] = hijo
                    cola_fwd.append(hijo)
                else:
                    repetidos_descartados += 1
        else:
            # Paso hacia atrás desde la meta
            nodo = cola_bwd.popleft()
            expandidos += 1
            for accion in problema.acciones(nodo.estado):
                sucesor = problema.resultado(nodo.estado, accion)
                costo = nodo.costo_camino + problema.costo_paso(nodo.estado, accion, sucesor)
                hijo = Nodo(
                    estado=sucesor,
                    padre=nodo,
                    accion=accion,
                    costo_camino=costo,
                    profundidad=nodo.profundidad + 1,
                )
                generados += 1

                if sucesor in alcanzados_fwd:
                    # ¡Colisión detectada!
                    nodo_encuentro_fwd = alcanzados_fwd[sucesor]
                    nodo_encuentro_bwd = hijo
                    break

                if sucesor not in alcanzados_bwd:
                    alcanzados_bwd[sucesor] = hijo
                    cola_bwd.append(hijo)
                else:
                    repetidos_descartados += 1

        frontera_actual = len(cola_fwd) + len(cola_bwd)
        if frontera_actual > frontera_maxima:
            frontera_maxima = frontera_actual

        if nodo_encuentro_fwd is not None:
            break

    t1 = time.perf_counter()

    if nodo_encuentro_fwd is not None and nodo_encuentro_bwd is not None:
        # Reconstruir camino hacia adelante: inicio -> punto_encuentro
        camino_fwd, acciones_fwd = nodo_encuentro_fwd.reconstruir_camino()

        # Reconstruir camino hacia atrás: meta -> punto_encuentro
        camino_bwd, _ = nodo_encuentro_bwd.reconstruir_camino()
        # camino_bwd tiene: [meta, ..., punto_encuentro].
        # Invertir para tener [punto_encuentro, ..., meta]
        camino_bwd.reverse()

        # Unir respetando la orientación: camino_fwd + camino_bwd[1:] (sin duplicar encuentro)
        camino_completo = camino_fwd + camino_bwd[1:]

        # Reconstruir la secuencia unificada de acciones
        acciones_completas = []
        costo_total = 0.0
        for i in range(len(camino_completo) - 1):
            s1 = camino_completo[i]
            s2 = camino_completo[i + 1]
            delta = (s2[0] - s1[0], s2[1] - s1[1])
            nombre_acc = DELTA_ACCIONES[delta]
            acciones_completas.append(nombre_acc)
            costo_total += problema.costo_paso(s1, nombre_acc, s2)

        return ResultadoBusqueda(
            encontrado=True,
            camino=camino_completo,
            acciones=acciones_completas,
            costo=costo_total,
            profundidad=len(acciones_completas),
            expandidos=expandidos,
            generados=generados,
            repetidos_descartados=repetidos_descartados,
            frontera_maxima=frontera_maxima,
            tiempo_ms=(t1 - t0) * 1000.0,
        )

    return ResultadoBusqueda(
        encontrado=False,
        camino=[],
        acciones=[],
        costo=0.0,
        profundidad=0,
        expandidos=expandidos,
        generados=generados,
        repetidos_descartados=repetidos_descartados,
        frontera_maxima=frontera_maxima,
        tiempo_ms=(t1 - t0) * 1000.0,
    )


# ==============================================================================
# 7. ALGORITMO DE LEE (EXPANSIÓN POR FRENTE DE ONDA)
# ==============================================================================
def lee_cero(
    problema: ProblemaLaberinto,
    guardar_historial_frentes: bool = False,
) -> Tuple[ResultadoBusqueda, Dict[Tuple[int, int], int], List[List[Tuple[int, int]]]]:
    """
    Algoritmo de Lee: propagación simultánea del frente de onda.
    Retorna: (ResultadoBusqueda, etiquetas_dict, historial_frentes).
    """
    t0 = time.perf_counter()
    etiquetas: Dict[Tuple[int, int], int] = {problema.inicio: 0}
    frente_actual: List[Tuple[int, int]] = [problema.inicio]
    historial_frentes: List[List[Tuple[int, int]]] = [[problema.inicio]] if guardar_historial_frentes else []

    expandidos = 0
    generados = 1
    repetidos_descartados = 0
    frontera_maxima = 1

    while frente_actual and problema.meta not in etiquetas:
        frente_siguiente: List[Tuple[int, int]] = []
        for celda in frente_actual:
            expandidos += 1
            for accion in problema.acciones(celda):
                vecino = problema.resultado(celda, accion)
                generados += 1
                if vecino not in etiquetas:
                    etiquetas[vecino] = etiquetas[celda] + 1
                    frente_siguiente.append(vecino)
                else:
                    repetidos_descartados += 1

        frente_actual = frente_siguiente
        if guardar_historial_frentes and frente_actual:
            historial_frentes.append(list(frente_actual))
        if len(frente_actual) > frontera_maxima:
            frontera_maxima = len(frente_actual)

    t1 = time.perf_counter()

    if problema.meta not in etiquetas:
        res = ResultadoBusqueda(
            encontrado=False,
            camino=[],
            acciones=[],
            costo=0.0,
            profundidad=0,
            expandidos=expandidos,
            generados=generados,
            repetidos_descartados=repetidos_descartados,
            frontera_maxima=frontera_maxima,
            tiempo_ms=(t1 - t0) * 1000.0,
        )
        return res, etiquetas, historial_frentes

    # Reconstrucción descendente desde meta siguiendo etiquetas decrecientes:
    # meta -> ... -> inicio
    camino_reverso = [problema.meta]
    actual = problema.meta
    while actual != problema.inicio:
        etiqueta_esperada = etiquetas[actual] - 1
        vecino_encontrado = None
        for accion in problema.acciones(actual):
            vecino = problema.resultado(actual, accion)
            if etiquetas.get(vecino) == etiqueta_esperada:
                vecino_encontrado = vecino
                break
        if vecino_encontrado is None:
            raise RuntimeError(f"Fallo en reconstrucción de Lee desde {actual}")
        actual = vecino_encontrado
        camino_reverso.append(actual)

    camino = list(reversed(camino_reverso))
    acciones = []
    costo_total = 0.0
    for i in range(len(camino) - 1):
        s1 = camino[i]
        s2 = camino[i + 1]
        delta = (s2[0] - s1[0], s2[1] - s1[1])
        nombre_acc = DELTA_ACCIONES[delta]
        acciones.append(nombre_acc)
        costo_total += problema.costo_paso(s1, nombre_acc, s2)

    res = ResultadoBusqueda(
        encontrado=True,
        camino=camino,
        acciones=acciones,
        costo=costo_total,
        profundidad=len(acciones),
        expandidos=expandidos,
        generados=generados,
        repetidos_descartados=repetidos_descartados,
        frontera_maxima=frontera_maxima,
        tiempo_ms=(t1 - t0) * 1000.0,
    )
    return res, etiquetas, historial_frentes
