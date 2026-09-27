"""
Versión 2: Adaptador del problema de búsqueda a SimpleAI.
Modela el laberinto como SearchProblem e instrumenta la recolección
de métricas mediante un BaseViewer personalizado sin alterar la biblioteca.
"""

import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from simpleai.search import (
    SearchProblem,
    breadth_first,
    depth_first,
    iterative_limited_depth_first,
    limited_depth_first,
    uniform_cost,
)
from simpleai.search.viewers import BaseViewer

from src.comun import ACCIONES_DELTA, ResultadoBusqueda


class LaberintoSimpleAI(SearchProblem):
    """
    Subclase de SearchProblem requerida por SimpleAI.
    Utiliza estados inmutables (tuplas de enteros) y operaciones puras.
    """

    def __init__(
        self,
        grafo: Dict[Tuple[int, int], Set[Tuple[int, int]]],
        inicio: Tuple[int, int],
        meta: Tuple[int, int],
        costo_arista_fn: Optional[Callable[[Tuple[int, int], Tuple[int, int]], float]] = None,
    ):
        super().__init__(initial_state=inicio)
        self.grafo = grafo
        self.meta = meta
        self.costo_arista_fn = costo_arista_fn or (lambda u, v: 1.0)

    def actions(self, state: Tuple[int, int]) -> List[str]:
        """Produce únicamente movimientos legales transitables."""
        r, c = state
        acciones_legales = []
        for nombre_acc, (dr, dc) in ACCIONES_DELTA.items():
            vecino = (r + dr, c + dc)
            if vecino in self.grafo.get(state, set()):
                acciones_legales.append(nombre_acc)
        return acciones_legales

    def result(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Calcula el estado sucesor sin mutar el original."""
        dr, dc = ACCIONES_DELTA[action]
        return (state[0] + dr, state[1] + dc)

    def is_goal(self, state: Tuple[int, int]) -> bool:
        """Prueba de objetivo."""
        return state == self.meta

    def cost(self, state: Tuple[int, int], action: str, state2: Tuple[int, int]) -> float:
        """Devuelve el costo del corredor entre state y state2."""
        return self.costo_arista_fn(state, state2)


class VisorMetricasSimpleAI(BaseViewer):
    """
    Instrumentación desacoplada de SimpleAI vía BaseViewer.
    Monitorea eventos del algoritmo para reportar exactamente las métricas del contrato.
    """

    def __init__(self):
        super().__init__()
        self.expandidos_reales = 0
        self.generados_reales = 1  # Nodo inicial raíz
        self.repetidos_descartados = 0
        self.max_frontera = 1

    def handle_expanded(self, nodes, successors):
        self.expandidos_reales += len(nodes)
        self.generados_reales += len(successors)

    def handle_new_iteration(self, fringe):
        super().handle_new_iteration(fringe)
        tam = len(fringe)
        if tam > self.max_frontera:
            self.max_frontera = tam


def ejecutar_busqueda_simpleai(
    algoritmo_fn: Callable,
    problema: LaberintoSimpleAI,
    **kwargs,
) -> ResultadoBusqueda:
    """Ejecuta un algoritmo de SimpleAI y extrae un ResultadoBusqueda con métricas unificadas."""
    visor = VisorMetricasSimpleAI()
    t0 = time.perf_counter()

    nodo_solucion = algoritmo_fn(
        problema,
        graph_search=True,
        viewer=visor,
        **kwargs,
    )
    t1 = time.perf_counter()

    if nodo_solucion is not None:
        camino = [estado for accion, estado in nodo_solucion.path()]
        acciones = [accion for accion, estado in nodo_solucion.path() if accion is not None]
        costo = float(nodo_solucion.cost)
        profundidad = len(acciones)

        # Calculo de repetidos descartados
        repetidos = max(0, visor.generados_reales - visor.expandidos_reales - len(camino))

        return ResultadoBusqueda(
            encontrado=True,
            camino=camino,
            acciones=acciones,
            costo=costo,
            profundidad=profundidad,
            expandidos=visor.expandidos_reales,
            generados=visor.generados_reales,
            repetidos_descartados=repetidos,
            frontera_maxima=visor.max_frontera,
            tiempo_ms=(t1 - t0) * 1000.0,
        )

    return ResultadoBusqueda(
        encontrado=False,
        camino=[],
        acciones=[],
        costo=0.0,
        profundidad=0,
        expandidos=visor.expandidos_reales,
        generados=visor.generados_reales,
        repetidos_descartados=max(0, visor.generados_reales - visor.expandidos_reales),
        frontera_maxima=visor.max_frontera,
        tiempo_ms=(t1 - t0) * 1000.0,
    )
