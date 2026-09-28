"""
Protocolo Experimental y Generación de Gráficas Estadísticas.
Sección 10 del Taller de Inteligencia Artificial.
Control de semillas, cómputo de Mediana e IQR y generación de 6 gráficas obligatorias.
"""

import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aima.search as asrch
from simpleai.search import breadth_first as s_bfs
from simpleai.search import depth_first as s_dfs
from simpleai.search import uniform_cost as s_ucs

from src.adaptador_aima import ejecutar_busqueda_aima
from src.adaptador_simpleai import LaberintoSimpleAI, ejecutar_busqueda_simpleai
from src.busqueda_cero import (
    ProblemaLaberinto,
    bfs_cero,
    bidireccional_cero,
    dfs_cero,
    lee_cero,
    ucs_cero,
)
from src.topologias_costos import agregar_ciclos


def ejecutar_protocolo_completo(generador_cls, n_repeticiones: int = 30) -> pd.DataFrame:
    """
    Ejecuta un protocolo factorial con al menos 30 repeticiones por configuración,
    controlando semillas y separando tiempo de generación de tiempo de búsqueda.
    """
    registros = []
    tamanos = [(15, 15), (25, 25), (40, 40)]
    topologias = ["arbol", "ciclos_5%", "ciclos_10%"]

    print(f"Iniciando protocolo experimental ({n_repeticiones} instancias por configuración)...")

    for filas, columnas in tamanos:
        num_estados = filas * columnas
        for topo in topologias:
            for rep in range(n_repeticiones):
                semilla = 10000 + rep * 101 + filas * 7

                # 1. Medir tiempo de generación
                t_gen_0 = time.perf_counter()
                lab = generador_cls(filas=filas, columnas=columnas, semilla=semilla)
                grafo = lab.generar()

                if topo == "ciclos_5%":
                    grafo, _, _ = agregar_ciclos(grafo, filas, columnas, 0.05, semilla=semilla)
                elif topo == "ciclos_10%":
                    grafo, _, _ = agregar_ciclos(grafo, filas, columnas, 0.10, semilla=semilla)

                t_gen_1 = time.perf_counter()
                tiempo_gen_ms = (t_gen_1 - t_gen_0) * 1000.0

                # Seleccionar puntos de prueba: esquinas e interior
                puntos = [
                    ("esquinas", (1, 1), (filas - 2, columnas - 2)),
                    ("interior", (filas // 4, columnas // 4), (3 * filas // 4, 3 * columnas // 4)),
                ]

                for tipo_pos, inicio, meta in puntos:
                    prob = ProblemaLaberinto(grafo, inicio, meta)

                    # Algoritmos evaluados
                    algoritmos = {
                        "BFS": bfs_cero,
                        "DFS": dfs_cero,
                        "UCS": ucs_cero,
                        "Bidireccional": bidireccional_cero,
                        "Lee": lambda p: lee_cero(p)[0],
                    }

                    for nombre_algo, algo_fn in algoritmos.items():
                        res = algo_fn(prob)
                        registros.append({
                            "tamano": f"{filas}x{columnas}",
                            "estados": num_estados,
                            "topologia": topo,
                            "posicion": tipo_pos,
                            "semilla": semilla,
                            "algoritmo": nombre_algo,
                            "tiempo_gen_ms": tiempo_gen_ms,
                            "tiempo_busqueda_ms": res.tiempo_ms,
                            "expandidos": res.expandidos,
                            "generados": res.generados,
                            "frontera_maxima": res.frontera_maxima,
                            "profundidad": res.profundidad,
                            "costo": res.costo,
                            "pasos": len(res.camino) - 1 if res.encontrado else 0,
                        })

    df = pd.DataFrame(registros)
    print(f"Protocolo finalizado. Total registros generados: {len(df)}")
    return df


def calcular_tabla_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula Mediana y Rango Intercuartílico (IQR = Q3 - Q1) para cada factor y algoritmo."""
    def iqr(x):
        return np.percentile(x, 75) - np.percentile(x, 25)

    agrupado = df.groupby(["tamano", "topologia", "algoritmo"]).agg(
        expandidos_mediana=("expandidos", "median"),
        expandidos_iqr=("expandidos", iqr),
        tiempo_mediana=("tiempo_busqueda_ms", "median"),
        tiempo_iqr=("tiempo_busqueda_ms", iqr),
        frontera_mediana=("frontera_maxima", "median"),
        frontera_iqr=("frontera_maxima", iqr),
    ).reset_index()
    return agrupado


def generar_las_seis_graficas(
    df: pd.DataFrame,
    generador_cls,
    filas_ind: int = 27,
    columnas_ind: int = 25,
    semilla_ind: int = 10057,
):
    """Genera las 6 figuras obligatorias especificadas en la Sección 10."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------------------
    # Gráfica 1: Expandidos vs Número de Estados
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    df_tree = df[df["topologia"] == "arbol"]
    for algo in ["BFS", "DFS", "UCS", "Bidireccional", "Lee"]:
        sub = df_tree[df_tree["algoritmo"] == algo]
        resumen = sub.groupby("estados")["expandidos"].agg(["median", lambda x: np.percentile(x, 75) - np.percentile(x, 25)])
        resumen.columns = ["mediana", "iqr"]
        ax.errorbar(
            resumen.index, resumen["mediana"], yerr=resumen["iqr"],
            marker="o", linewidth=2, capsize=4, label=algo
        )
    ax.set_title("Gráfica 1: Nodos Expandidos vs Número de Estados (|V|) [Mediana e IQR]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Número de Estados (|V| = Filas × Columnas)", fontsize=11)
    ax.set_ylabel("Nodos Expandidos", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("grafica1_expandidos_vs_estados.png", dpi=180)
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfica 2: Frontera Máxima vs Profundidad de Solución
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo, color, marker in [("BFS", "blue", "o"), ("DFS", "red", "^"), ("Bidireccional", "green", "s")]:
        sub = df[df["algoritmo"] == algo]
        ax.scatter(sub["profundidad"], sub["frontera_maxima"], alpha=0.35, color=color, marker=marker, label=algo, s=25)
    ax.set_title("Gráfica 2: Frontera Máxima vs Profundidad de la Solución", fontsize=12, fontweight="bold")
    ax.set_xlabel("Profundidad de la Solución (Pasos)", fontsize=11)
    ax.set_ylabel("Tamaño Pico de la Frontera (Nodos simultáneos)", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("grafica2_frontera_vs_profundidad.png", dpi=180)
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfica 3: Tiempo frente a Tamaño
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ["BFS", "DFS", "UCS", "Bidireccional"]:
        sub = df[df["algoritmo"] == algo]
        resumen = sub.groupby("estados")["tiempo_busqueda_ms"].median()
        ax.plot(resumen.index, resumen.values, marker="s", linewidth=2, label=algo)
    ax.set_title("Gráfica 3: Tiempo de Búsqueda (ms) frente al Tamaño del Laberinto", fontsize=12, fontweight="bold")
    ax.set_xlabel("Número de Estados (|V|)", fontsize=11)
    ax.set_ylabel("Tiempo Mediano de Búsqueda (ms)", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("grafica3_tiempo_vs_tamano.png", dpi=180)
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfica 4: Costo y Longitud de las Soluciones
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    topos = ["arbol", "ciclos_5%", "ciclos_10%"]
    x = np.arange(len(topos))
    ancho = 0.35

    long_bfs = [df[(df["topologia"] == t) & (df["algoritmo"] == "BFS")]["pasos"].median() for t in topos]
    long_dfs = [df[(df["topologia"] == t) & (df["algoritmo"] == "DFS")]["pasos"].median() for t in topos]

    ax.bar(x - ancho / 2, long_bfs, ancho, label="BFS (Óptimo en pasos)", color="royalblue")
    ax.bar(x + ancho / 2, long_dfs, ancho, label="DFS (No óptimo)", color="crimson")
    ax.set_title("Gráfica 4: Comparación de Longitud de Solución por Topología", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Árbol (Base)", "5% Ciclos", "10% Ciclos"])
    ax.set_ylabel("Longitud Mediana de la Ruta (Pasos)", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("grafica4_costo_y_longitud.png", dpi=180)
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfica 5: Mapa de Calor de Lee (Asegurar archivo grafica5_mapa_calor_lee.png)
    # -------------------------------------------------------------------------
    lab = generador_cls(filas=filas_ind, columnas=columnas_ind, semilla=semilla_ind)
    grafo_ind = lab.generar()
    prob_ind = ProblemaLaberinto(grafo_ind, (3, 4), (filas_ind - 4, columnas_ind - 5))
    res_lee, etiquetas, _ = lee_cero(prob_ind)

    matriz = np.full((filas_ind, columnas_ind), np.nan)
    for (r, c), dist in etiquetas.items():
        matriz[r, c] = dist

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matriz, cmap="magma", origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tiempo de Llegada / Etiqueta", fontsize=11)

    if res_lee.camino:
        cc = [c for _, c in res_lee.camino]
        cr = [r for r, _ in res_lee.camino]
        ax.plot(cc, cr, color="cyan", linewidth=2.5, label=f"Camino Óptimo ({len(res_lee.camino)-1} pasos)")

    ax.scatter([4], [3], c="lime", s=110, edgecolors="black", zorder=5, label="Inicio (3, 4)")
    ax.scatter([columnas_ind - 5], [filas_ind - 4], c="red", s=140, marker="*", edgecolors="black", zorder=5, label="Meta")
    ax.set_title("Gráfica 5: Mapa de Calor de Propagación de Lee", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("grafica5_mapa_calor_lee.png", dpi=180)
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfica 6: Comparación Cruzada de las Tres Versiones (Desde cero, SimpleAI, AIMA)
    # -------------------------------------------------------------------------
    versiones_data = []
    # Comparar sobre una muestra de 10 laberintos
    for s in range(10):
        sem = 3000 + s
        l = generador_cls(20, 20, sem)
        g = l.generar()
        ini, fin = (1, 1), (18, 18)

        # 1. Desde cero
        p0 = ProblemaLaberinto(g, ini, fin)
        t0 = time.perf_counter()
        r_cero = bfs_cero(p0)
        t_cero = (time.perf_counter() - t0) * 1000

        # 2. SimpleAI
        p_s = LaberintoSimpleAI(g, ini, fin)
        t0 = time.perf_counter()
        r_sai = ejecutar_busqueda_simpleai(s_bfs, p_s)
        t_sai = (time.perf_counter() - t0) * 1000

        # 3. AIMA-Python
        t0 = time.perf_counter()
        r_aima = ejecutar_busqueda_aima(asrch.breadth_first_graph_search, g, ini, fin)
        t_aima = (time.perf_counter() - t0) * 1000

        versiones_data.append({"version": "Desde Cero", "tiempo_ms": t_cero, "expandidos": r_cero.expandidos})
        versiones_data.append({"version": "SimpleAI", "tiempo_ms": t_sai, "expandidos": r_sai.expandidos})
        versiones_data.append({"version": "AIMA-Python", "tiempo_ms": t_aima, "expandidos": r_aima.expandidos})

    df_v = pd.DataFrame(versiones_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    df_v.groupby("version")["tiempo_ms"].mean().plot(kind="bar", ax=ax1, color=["teal", "darkorange", "purple"])
    ax1.set_title("Tiempo Medio de Búsqueda (ms)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Milisegundos")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    df_v.groupby("version")["expandidos"].mean().plot(kind="bar", ax=ax2, color=["teal", "darkorange", "purple"])
    ax2.set_title("Promedio de Nodos Expandidos", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cantidad de Nodos")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

    plt.suptitle("Gráfica 6: Comparación Cruzada de las Tres Versiones (Algoritmo BFS)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("grafica6_comparativa_versiones.png", dpi=180)
    plt.close()

    print("Las 6 gráficas obligatorias fueron generadas y guardadas exitosamente.")
