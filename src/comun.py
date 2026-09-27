"""
Definiciones comunes para los algoritmos de búsqueda no informada:
- Estructura de Nodo de búsqueda
- Contrato obligatorio ResultadoBusqueda
- Mapeo de acciones y utilidades
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict


ACCIONES_DELTA: Dict[str, Tuple[int, int]] = {
    "ARRIBA": (-1, 0),
    "ABAJO": (1, 0),
    "IZQUIERDA": (0, -1),
    "DERECHA": (0, 1),
}

DELTA_ACCIONES: Dict[Tuple[int, int], str] = {
    v: k for k, v in ACCIONES_DELTA.items()
}


@dataclass
class ResultadoBusqueda:
    """Contrato común y estricto para todas las versiones del taller."""
    encontrado: bool
    camino: List[Tuple[int, int]]       # Secuencia de estados (de inicio a meta inclusive)
    acciones: List[str]                 # Secuencia de acciones aplicadas
    costo: float                        # Costo acumulado total del camino
    profundidad: int                    # Profundidad del nodo solución (camino - 1)
    expandidos: int                     # Nodos retirados de frontera y cuyos sucesores se generaron
    generados: int                      # Total de nodos sucesores construidos
    repetidos_descartados: int          # Nodos sucesores descartados por ya haber sido visitados
    frontera_maxima: int                # Tamaño pico alcanzado por la estructura de frontera
    tiempo_ms: float                    # Tiempo de ejecución medido en milisegundos

    def __str__(self) -> str:
        return (
            f"ResultadoBusqueda(\n"
            f"  encontrado={self.encontrado},\n"
            f"  pasos={len(self.camino)-1 if self.encontrado else 0},\n"
            f"  costo={self.costo},\n"
            f"  profundidad={self.profundidad},\n"
            f"  expandidos={self.expandidos},\n"
            f"  generados={self.generados},\n"
            f"  repetidos_descartados={self.repetidos_descartados},\n"
            f"  frontera_maxima={self.frontera_maxima},\n"
            f"  tiempo_ms={self.tiempo_ms:.3f} ms\n"
            f")"
        )


class Nodo:
    """
    Representa un nodo en el árbol de búsqueda.
    Encapsula el estado, nodo padre, acción, costo g(n) y profundidad.
    """
    __slots__ = ("estado", "padre", "accion", "costo_camino", "profundidad")

    def __init__(
        self,
        estado: Tuple[int, int],
        padre: Optional["Nodo"] = None,
        accion: Optional[str] = None,
        costo_camino: float = 0.0,
        profundidad: int = 0,
    ):
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.costo_camino = costo_camino
        self.profundidad = profundidad

    def __lt__(self, otro: "Nodo") -> bool:
        """Comparador por costo acumulado g(n) para colas de prioridad."""
        return self.costo_camino < otro.costo_camino

    def reconstruir_camino(self) -> Tuple[List[Tuple[int, int]], List[str]]:
        """Reconstruye la secuencia de estados y acciones desde la raíz hasta este nodo."""
        estados: List[Tuple[int, int]] = []
        acciones: List[str] = []
        actual: Optional[Nodo] = self

        while actual is not None:
            estados.append(actual.estado)
            if actual.accion is not None:
                acciones.append(actual.accion)
            actual = actual.padre

        estados.reverse()
        acciones.reverse()
        return estados, acciones
