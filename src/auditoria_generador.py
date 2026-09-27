"""
Módulo de auditoría y pruebas para el generador Hunt-and-Kill.
Verifica las propiedades matemáticas y estructurales del grafo generado.
"""

from collections import deque
import random


class AuditoriaGenerador:
    @staticmethod
    def verificar_simetria(grafo):
        """Verifica que si v in grafo[u], entonces u in grafo[v]."""
        for u, vecinos in grafo.items():
            for v in vecinos:
                if u not in grafo[v]:
                    return False, f"Asimetría detectada: {v} no contiene a {u}"
        return True, "Simetría verificada correctamente."

    @staticmethod
    def verificar_ausencia_diagonales(grafo):
        """Verifica que toda conexión sea ortogonal (distancia Manhattan = 1)."""
        for u, vecinos in grafo.items():
            ru, cu = u
            for v in vecinos:
                rv, cv = v
                manhattan = abs(ru - rv) + abs(cu - cv)
                if manhattan != 1:
                    return False, f"Conexión no ortogonal entre {u} y {v} (dist={manhattan})"
        return True, "Ausencia de diagonales verificada."

    @staticmethod
    def verificar_validez_coordenadas(grafo, filas, columnas):
        """Verifica que todos los nodos y sus vecinos estén dentro de la grilla."""
        for u, vecinos in grafo.items():
            ru, cu = u
            if not (0 <= ru < filas and 0 <= cu < columnas):
                return False, f"Coordenada de nodo inválida: {u}"
            for v in vecinos:
                rv, cv = v
                if not (0 <= rv < filas and 0 <= cv < columnas):
                    return False, f"Coordenada de vecino inválida: {v}"
        return True, "Coordenadas válidas en todos los nodos y corredores."

    @staticmethod
    def verificar_conectividad(grafo):
        """Verifica que el grafo sea conexo mediante un recorrido BFS."""
        if not grafo:
            return True, "Grafo vacío."
        inicio = next(iter(grafo))
        visitados = set()
        cola = deque([inicio])
        visitados.add(inicio)

        while cola:
            actual = cola.popleft()
            for vecino in grafo[actual]:
                if vecino not in visitados:
                    visitados.add(vecino)
                    cola.append(vecino)

        total_esperado = len(grafo)
        total_alcanzado = len(visitados)
        if total_alcanzado != total_esperado:
            return False, f"Grafo desconexo: se alcanzaron {total_alcanzado}/{total_esperado} nodos"
        return True, f"Conectividad total verificada: {total_alcanzado} nodos alcanzados."

    @staticmethod
    def verificar_ausencia_ciclos(grafo):
        """
        Verifica ausencia de ciclos mediante DFS y detección de aristas de retroceso (back-edges).
        También corrobora que |E| == |V| - 1.
        """
        num_vertices = len(grafo)
        num_aristas_dirigidas = sum(len(vecinos) for vecinos in grafo.values())
        num_aristas = num_aristas_dirigidas // 2

        if num_aristas != num_vertices - 1:
            return False, f"Fallo en fórmula de árbol: |V|={num_vertices}, |E|={num_aristas} (!= |V|-1)"

        # Verificación explícita de ciclos vía DFS
        visitados = set()
        padre = {}
        for nodo_raiz in grafo:
            if nodo_raiz not in visitados:
                pila = [(nodo_raiz, None)]
                while pila:
                    actual, p = pila.pop()
                    if actual in visitados:
                        continue
                    visitados.add(actual)
                    padre[actual] = p

                    for vecino in grafo[actual]:
                        if vecino == p:
                            continue
                        if vecino in visitados:
                            return False, f"Ciclo detectado entre {actual} y {vecino}"
                        pila.append((vecino, actual))

        return True, f"Ausencia de ciclos verificada (|V|={num_vertices}, |E|={num_aristas}, 0 ciclos)."

    @staticmethod
    def verificar_reproducibilidad(generador_cls, filas=15, columnas=15, semilla=2026):
        """Verifica que dos instancias con la misma semilla generen exactamente el mismo grafo."""
        lab1 = generador_cls(filas=filas, columnas=columnas, semilla=semilla)
        g1 = lab1.generar()

        lab2 = generador_cls(filas=filas, columnas=columnas, semilla=semilla)
        g2 = lab2.generar()

        if g1 != g2:
            return False, "Fallo: Mismas semillas produjeron grafos diferentes."

        lab3 = generador_cls(filas=filas, columnas=columnas, semilla=semilla + 1)
        g3 = lab3.generar()
        if g1 == g3:
            return False, "Fallo: Diferentes semillas produjeron grafos idénticos."

        return True, "Reproducibilidad confirmada: sembrado idéntico produce grafos idénticos."

    @classmethod
    def ejecutar_bateria_completa(cls, laberinto, generador_cls):
        """Ejecuta todas las pruebas sobre una instancia generada."""
        grafo = laberinto.grafo
        filas = laberinto.filas
        columnas = laberinto.columnas

        reporte = {}
        reporte["simetria"] = cls.verificar_simetria(grafo)
        reporte["diagonales"] = cls.verificar_ausencia_diagonales(grafo)
        reporte["coordenadas"] = cls.verificar_validez_coordenadas(grafo, filas, columnas)
        reporte["conectividad"] = cls.verificar_conectividad(grafo)
        reporte["ciclos"] = cls.verificar_ausencia_ciclos(grafo)
        reporte["reproducibilidad"] = cls.verificar_reproducibilidad(
            generador_cls, filas=filas, columnas=columnas, semilla=2026
        )

        todo_exitoso = all(estado for estado, _ in reporte.values())
        return todo_exitoso, reporte
