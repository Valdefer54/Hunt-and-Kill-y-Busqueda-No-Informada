"""
Batería de Pruebas de Aceptación, Concordancia y Casos Límite.
Valida formalmente los requisitos de la Sección 8 del taller.
"""

from typing import Dict, List, Set, Tuple

import aima.search as asrch
from simpleai.search import breadth_first as simpleai_bfs
from simpleai.search import uniform_cost as simpleai_ucs

from src.adaptador_aima import ejecutar_busqueda_aima
from src.adaptador_simpleai import LaberintoSimpleAI, ejecutar_busqueda_simpleai
from src.busqueda_cero import (
    CORTE,
    ProblemaLaberinto,
    bfs_cero,
    bidireccional_cero,
    dfs_cero,
    dls_cero,
    dls_wrapper,
    lee_cero,
    ucs_cero,
)
from src.comun import ACCIONES_DELTA, ResultadoBusqueda


class ValidadorCamino:
    """Validador estricto para los resultados de búsqueda."""

    @staticmethod
    def validar(problema: ProblemaLaberinto, resultado: ResultadoBusqueda) -> Tuple[bool, str]:
        if not resultado.encontrado:
            if resultado.camino != []:
                return False, "Si encontrado es False, camino debe ser una lista vacía."
            return True, "No se encontró solución (consistente)."

        camino = resultado.camino
        acciones = resultado.acciones

        # 1. Primer estado es el inicio
        if camino[0] != problema.inicio:
            return False, f"El primer estado {camino[0]} no coincide con el inicio {problema.inicio}"

        # 2. Último estado es la meta
        if camino[-1] != problema.meta:
            return False, f"El último estado {camino[-1]} no coincide con la meta {problema.meta}"

        # 3. Cada pareja consecutiva aparece como arista en el grafo
        costo_recalculado = 0.0
        for i in range(len(camino) - 1):
            u = camino[i]
            v = camino[i + 1]
            if v not in problema.grafo.get(u, set()):
                return False, f"La transición ({u} -> {v}) no existe como arista legal en el grafo."
            costo_recalculado += problema.costo_arista_fn(u, v)

        # 4. El costo reportado coincide con la suma de costos
        if abs(resultado.costo - costo_recalculado) > 1e-6:
            return False, f"Discrepancia en costo: reportado {resultado.costo} vs recalculado {costo_recalculado}"

        # 5. Aplicar las acciones reproduce exactamente los estados
        if acciones:
            estado_actual = camino[0]
            for idx, acc in enumerate(acciones):
                dr, dc = ACCIONES_DELTA[acc]
                estado_siguiente = (estado_actual[0] + dr, estado_actual[1] + dc)
                if estado_siguiente != camino[idx + 1]:
                    return False, f"Acción '{acc}' en paso {idx} produjo {estado_siguiente} != esperado {camino[idx+1]}"
                estado_actual = estado_siguiente

        return True, "Camino 100% válido y verificado."


class BateriaPruebasAceptacion:
    """Ejecutor de las pruebas de concordancia y casos límite obligatorios."""

    @classmethod
    def probar_concordancia_arbol(cls, grafo, inicio, meta):
        """Verifica que en un árbol todas las soluciones válidas tengan la misma secuencia."""
        prob = ProblemaLaberinto(grafo, inicio, meta)
        res_dfs = dfs_cero(prob)
        res_bfs = bfs_cero(prob)
        res_ucs = ucs_cero(prob)
        res_lee, _, _ = lee_cero(prob)

        assert res_bfs.camino == res_dfs.camino == res_ucs.camino == res_lee.camino, (
            "En un árbol, todas las rutas válidas deben ser idénticas."
        )
        assert len(res_bfs.camino) - 1 == int(res_ucs.costo) == (len(res_lee.camino) - 1), (
            "Discrepancia en longitud/costo unitario."
        )
        return True, "Concordancia en árbol verificada con éxito."

    @classmethod
    def probar_caso_inicio_igual_meta(cls):
        """Caso límite: inicio igual a meta."""
        grafo = {(0, 0): {(0, 1)}, (0, 1): {(0, 0)}}
        prob = ProblemaLaberinto(grafo, (0, 0), (0, 0))
        res_bfs = bfs_cero(prob)
        res_dfs = dfs_cero(prob)
        res_ucs = ucs_cero(prob)

        assert res_bfs.encontrado and res_bfs.camino == [(0, 0)] and res_bfs.costo == 0.0
        assert res_dfs.encontrado and res_dfs.camino == [(0, 0)] and res_dfs.costo == 0.0
        assert res_ucs.encontrado and res_ucs.camino == [(0, 0)] and res_ucs.costo == 0.0
        return True, "Caso inicio == meta verificado correctamente."

    @classmethod
    def probar_caso_laberinto_1x1(cls):
        """Caso límite: laberinto 1x1."""
        grafo = {(0, 0): set()}
        prob = ProblemaLaberinto(grafo, (0, 0), (0, 0))
        res_bfs = bfs_cero(prob)
        assert res_bfs.encontrado and res_bfs.camino == [(0, 0)] and res_bfs.costo == 0.0
        return True, "Caso 1x1 verificado correctamente."

    @classmethod
    def probar_caso_meta_inalcanzable(cls):
        """Caso límite: meta inalcanzable tras eliminar corredores."""
        # Dos componentes disjuntas
        grafo = {(0, 0): {(0, 1)}, (0, 1): {(0, 0)}, (1, 0): set()}
        prob = ProblemaLaberinto(grafo, (0, 0), (1, 0))
        res_bfs = bfs_cero(prob)
        res_dfs = dfs_cero(prob)
        res_ucs = ucs_cero(prob)
        assert not res_bfs.encontrado and res_bfs.camino == []
        assert not res_dfs.encontrado and res_dfs.camino == []
        assert not res_ucs.encontrado and res_ucs.camino == []
        return True, "Caso meta inalcanzable verificado correctamente."

    @classmethod
    def probar_caso_entradas_obsoletas_ucs(cls):
        """Caso límite: múltiples entradas obsoletas en UCS."""
        # Grafo con dos caminos a (1,1):
        # Camino A: (0,0) -> (0,1) -> (1,1), costos 10 + 10 = 20
        # Camino B: (0,0) -> (1,0) -> (1,1), costos 1 + 1 = 2
        grafo = {
            (0, 0): {(0, 1), (1, 0)},
            (0, 1): {(0, 0), (1, 1)},
            (1, 0): {(0, 0), (1, 1)},
            (1, 1): {(0, 1), (1, 0)},
        }
        costos = {
            ((0, 0), (0, 1)): 10.0,
            ((0, 1), (1, 1)): 10.0,
            ((0, 0), (1, 0)): 1.0,
            ((1, 0), (1, 1)): 1.0,
        }
        costo_fn = lambda u, v: costos.get((u, v), costos.get((v, u), 1.0))
        prob = ProblemaLaberinto(grafo, (0, 0), (1, 1), costo_arista_fn=costo_fn)

        res_ucs = ucs_cero(prob)
        assert res_ucs.encontrado and res_ucs.costo == 2.0
        assert res_ucs.camino == [(0, 0), (1, 0), (1, 1)]
        return True, "Caso entradas obsoletas en UCS verificado con optimalidad."

    @classmethod
    def probar_caso_limites_dls(cls, grafo, inicio, meta, profundidad_exacta):
        """Caso límite: DLS con límite insuficiente, exacto y excesivo."""
        prob = ProblemaLaberinto(grafo, inicio, meta)

        # 1. Límite insuficiente (< d)
        res_insuf, _ = dls_cero(prob, limite=profundidad_exacta - 5)
        assert res_insuf == CORTE, f"DLS insuficiente debió retornar CORTE, retornó {res_insuf}"

        # 2. Límite exacto (== d)
        res_exact, _ = dls_cero(prob, limite=profundidad_exacta)
        assert hasattr(res_exact, "estado") and res_exact.estado == meta, "DLS con límite exacto debió hallar la meta."

        # 3. Límite excesivo (> d)
        res_exces, _ = dls_cero(prob, limite=profundidad_exacta + 20)
        assert hasattr(res_exces, "estado") and res_exces.estado == meta, "DLS con límite excesivo debió hallar la meta."

        return True, "Caso de límites en DLS (insuficiente, exacto, excesivo) verificado con éxito."
