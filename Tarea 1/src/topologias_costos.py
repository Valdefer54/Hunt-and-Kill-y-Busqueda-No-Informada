"""
Segunda Familia de Problemas: Topologías con Ciclos y Costos Ponderados.
Implementa transformaciones puras sobre grafos sin alterar la clase original.
"""

from collections import deque
import copy
import random
from typing import Callable, Dict, List, Set, Tuple

from src.busqueda_cero import ProblemaLaberinto, bfs_cero, dfs_cero, ucs_cero
from src.comun import ACCIONES_DELTA


def agregar_ciclos(
    grafo: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    filas: int,
    columnas: int,
    porcentaje: float = 0.08,
    semilla: int = 2026,
) -> Tuple[Dict[Tuple[int, int], Set[Tuple[int, int]]], int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Transformación A: Abre aleatoriamente entre 5% y 12% de las paredes internas que no son corredores.
    Conserva simetría ortogonal, reproducibilidad y genera ciclos comprobables.
    """
    assert 0.05 <= porcentaje <= 0.12, "El porcentaje de paredes abiertas debe estar entre 5% y 12%."
    rng = random.Random(semilla)
    grafo_ciclos = {k: set(v) for k, v in grafo.items()}

    # Localizar todas las paredes internas (adyacencias ortogonales no conectadas)
    paredes_candidatas: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    direcciones = [(0, 1), (1, 0)]  # Solo derecha y abajo para no duplicar no dirigidas

    for r in range(filas):
        for c in range(columnas):
            u = (r, c)
            for dr, dc in direcciones:
                v = (r + dr, c + dc)
                if 0 <= v[0] < filas and 0 <= v[1] < columnas:
                    if v not in grafo_ciclos[u]:
                        paredes_candidatas.append((u, v))

    total_paredes = len(paredes_candidatas)
    num_a_abrir = max(1, int(round(total_paredes * porcentaje)))

    paredes_abiertas = rng.sample(paredes_candidatas, num_a_abrir)

    for u, v in paredes_abiertas:
        grafo_ciclos[u].add(v)
        grafo_ciclos[v].add(u)

    return grafo_ciclos, num_a_abrir, paredes_abiertas


def detectar_ciclos_dfs(grafo: Dict[Tuple[int, int], Set[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """Detecta y extrae ciclos en el grafo mediante búsqueda en profundidad."""
    visitados = set()
    padres = {}
    ciclos = []

    for nodo_inicio in grafo:
        if nodo_inicio in visitados:
            continue
        pila = [(nodo_inicio, None)]
        while pila:
            actual, p = pila.pop()
            if actual in visitados:
                continue
            visitados.add(actual)
            padres[actual] = p

            for vecino in grafo[actual]:
                if vecino == p:
                    continue
                if vecino in visitados:
                    # Ciclo detectado: reconstruir el ciclo
                    ciclo = [vecino, actual]
                    curr = actual
                    while curr != vecino and curr is not None:
                        curr = padres.get(curr)
                        if curr is not None:
                            ciclo.append(curr)
                    ciclos.append(ciclo)
                    if len(ciclos) >= 5:  # Limitar para reporte conciso
                        return ciclos
                else:
                    pila.append((vecino, actual))
    return ciclos


def crear_escenario_dilema_costos(
    filas: int = 7,
    columnas: int = 7,
) -> Tuple[Dict[Tuple[int, int], Set[Tuple[int, int]]], Tuple[int, int], Tuple[int, int], Callable]:
    """
    Transformación B: Construye deliberadamente un escenario controlado donde:
    Camino con menos pasos != Camino de menor costo.

    Estructura:
    - Atajo corto: 2 aristas de alto costo (costo = 20.0 cada una, total = 40.0, 2 pasos).
    - Rodeo largo: 5 aristas de bajo costo (costo = 1.0 cada una, total = 5.0, 5 pasos).
    """
    grafo: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {
        (r, c): set() for r in range(filas) for c in range(columnas)
    }

    inicio = (1, 1)
    meta = (1, 3)

    # 1. Atajo directo: (1,1) -> (1,2) -> (1,3) [2 pasos]
    grafo[(1, 1)].add((1, 2))
    grafo[(1, 2)].add((1, 1))
    grafo[(1, 2)].add((1, 3))
    grafo[(1, 3)].add((1, 2))

    # 2. Desvío despejado: (1,1) -> (2,1) -> (2,2) -> (2,3) -> (1,3) [4 pasos]
    desvio = [(1, 1), (2, 1), (2, 2), (2, 3), (1, 3)]
    for i in range(len(desvio) - 1):
        u, v = desvio[i], desvio[i + 1]
        grafo[u].add(v)
        grafo[v].add(u)

    # Conectar el resto para formar una cuadrícula conexa básica
    for r in range(filas):
        for c in range(columnas):
            if c + 1 < columnas and (r, c + 1) not in grafo[(r, c)]:
                grafo[(r, c)].add((r, c + 1))
                grafo[(r, c + 1)].add((r, c))

    # Definición de la función de costos por arista
    costos_especiales = {
        ((1, 1), (1, 2)): 25.0,
        ((1, 2), (1, 1)): 25.0,
        ((1, 2), (1, 3)): 25.0,
        ((1, 3), (1, 2)): 25.0,
    }

    def costo_fn(u: Tuple[int, int], v: Tuple[int, int]) -> float:
        return costos_especiales.get((u, v), 1.0)

    return grafo, inicio, meta, costo_fn
