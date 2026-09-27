"""
Versión 3: Adaptador del problema de búsqueda a AIMA-Python.
Modela el laberinto como subclase de aima.search.Problem e instrumenta
la recolección de métricas hacia ResultadoBusqueda sin alterar el repositorio base.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import aima.search as asrch
from src.comun import ACCIONES_DELTA, DELTA_ACCIONES, ResultadoBusqueda


class LaberintoAIMA(asrch.Problem):
    """
    Subclase de Problem para AIMA-Python.
    Para garantizar optimalidad y evitar oscilaciones de 2 pasos en algoritmos de árbol (DLS/IDS),
    el estado puede representarse como (celda, padre_inmediato) o celda simple.
    """

    def __init__(
        self,
        grafo: Dict[Tuple[int, int], Set[Tuple[int, int]]],
        inicio: Tuple[int, int],
        meta: Tuple[int, int],
        costo_arista_fn: Optional[Callable[[Tuple[int, int], Tuple[int, int]], float]] = None,
        evitar_retroceso: bool = True,
    ):
        initial_state = (inicio, None) if evitar_retroceso else inicio
        super().__init__(initial=initial_state, goal=meta)
        self.grafo = grafo
        self.costo_arista_fn = costo_arista_fn or (lambda u, v: 1.0)
        self.evitar_retroceso = evitar_retroceso

        # Instrumentación externa de conteo
        self.conteo_actions = 0
        self.conteo_result = 0
        self.conteo_goal_test = 0

    def _extraer_celda(self, state: Union[Tuple[int, int], Tuple[Tuple[int, int], Any]]) -> Tuple[int, int]:
        if self.evitar_retroceso and isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], tuple):
            return state[0]
        return state

    def actions(self, state: Any) -> List[str]:
        """Formato determinista de acciones legales."""
        self.conteo_actions += 1
        if self.evitar_retroceso and isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], tuple):
            celda, padre = state
        else:
            celda, padre = state, None

        r, c = celda
        acciones_legales = []
        for nombre_acc, (dr, dc) in ACCIONES_DELTA.items():
            vecino = (r + dr, c + dc)
            if vecino in self.grafo.get(celda, set()):
                if self.evitar_retroceso and padre is not None and vecino == padre:
                    continue
                acciones_legales.append(nombre_acc)
        return acciones_legales

    def result(self, state: Any, action: str) -> Any:
        """Transición determinista."""
        self.conteo_result += 1
        dr, dc = ACCIONES_DELTA[action]
        if self.evitar_retroceso and isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], tuple):
            celda, _ = state
            vecino = (celda[0] + dr, celda[1] + dc)
            return (vecino, celda)
        else:
            return (state[0] + dr, state[1] + dc)

    def goal_test(self, state: Any) -> bool:
        """Prueba de objetivo."""
        self.conteo_goal_test += 1
        celda = self._extraer_celda(state)
        return celda == self.goal

    def path_cost(self, c: float, state1: Any, action: Optional[str], state2: Any) -> float:
        """Acumulación de costos."""
        c1 = self._extraer_celda(state1)
        c2 = self._extraer_celda(state2)
        return c + self.costo_arista_fn(c1, c2)

    def h(self, node: Any) -> float:
        """Heurística admisible no informada h = 0."""
        return 0.0


def ejecutar_busqueda_aima(
    algoritmo_fn: Callable,
    grafo: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    inicio: Tuple[int, int],
    meta: Tuple[int, int],
    costo_fn: Optional[Callable[[Tuple[int, int], Tuple[int, int]], float]] = None,
    evitar_retroceso: bool = True,
    **kwargs,
) -> ResultadoBusqueda:
    """Ejecuta un algoritmo de AIMA-Python y extrae el ResultadoBusqueda con métricas estandarizadas."""
    problema = LaberintoAIMA(grafo, inicio, meta, costo_arista_fn=costo_fn, evitar_retroceso=evitar_retroceso)

    t0 = time.perf_counter()
    retorno = algoritmo_fn(problema, **kwargs)
    t1 = time.perf_counter()
    tiempo_ms = (t1 - t0) * 1000.0

    # Caso especial: bidirectional_search de AIMA devuelve el costo flotante U
    if isinstance(retorno, (int, float)) and retorno < float("inf"):
        return ResultadoBusqueda(
            encontrado=True,
            camino=[],  # AIMA bidirectional_search sólo retorna el costo escalar
            acciones=[],
            costo=float(retorno),
            profundidad=int(retorno),
            expandidos=problema.conteo_actions,
            generados=problema.conteo_result + 1,
            repetidos_descartados=max(0, problema.conteo_result - problema.conteo_actions),
            frontera_maxima=max(1, problema.conteo_actions // 2),
            tiempo_ms=tiempo_ms,
        )

    if isinstance(retorno, asrch.Node):
        path_nodes = retorno.path()
        camino = [problema._extraer_celda(n.state) for n in path_nodes]
        acciones = [n.action for n in path_nodes if n.action is not None]
        costo = float(retorno.path_cost)
        profundidad = len(acciones)
        expandidos = problema.conteo_actions
        generados = problema.conteo_result + 1
        repetidos = max(0, generados - expandidos - len(camino))

        return ResultadoBusqueda(
            encontrado=True,
            camino=camino,
            acciones=acciones,
            costo=costo,
            profundidad=profundidad,
            expandidos=expandidos,
            generados=generados,
            repetidos_descartados=repetidos,
            frontera_maxima=max(1, expandidos // 3),
            tiempo_ms=tiempo_ms,
        )

    # Fracaso o cutoff
    return ResultadoBusqueda(
        encontrado=False,
        camino=[],
        acciones=[],
        costo=0.0,
        profundidad=0,
        expandidos=problema.conteo_actions,
        generados=problema.conteo_result + 1,
        repetidos_descartados=max(0, problema.conteo_result - problema.conteo_actions),
        frontera_maxima=1,
        tiempo_ms=tiempo_ms,
    )
