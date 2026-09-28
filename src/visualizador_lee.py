"""
Módulo de visualización para el Algoritmo de Lee (Frente de Onda).
Genera visualizaciones de los instantes del frente y mapa de calor de llegada.
Optimizado para carga ultra-rápida.
"""

import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def visualizar_frente_lee(
    filas: int,
    columnas: int,
    grafo: Dict[Tuple[int, int], set],
    etiquetas: Dict[Tuple[int, int], int],
    historial_frentes: List[List[Tuple[int, int]]],
    camino: List[Tuple[int, int]],
    inicio: Tuple[int, int],
    meta: Tuple[int, int],
    guardar_como: Optional[str] = "lee_frente_onda.png",
    forzar: bool = False,
):
    """
    Dibuja 8 instantes del frente de onda luminosa y el mapa final de llegada
    con el camino reconstruido.
    """
    if not forzar and guardar_como and os.path.exists(guardar_como):
        return

    total_pasos = len(historial_frentes)
    if total_pasos < 8:
        indices = list(range(total_pasos))
    else:
        indices = [int(i * (total_pasos - 1) / 7) for i in range(8)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    max_etiqueta = max(etiquetas.values()) if etiquetas else 1

    for i, idx in enumerate(indices):
        ax = axes[i]
        matriz = np.full((filas, columnas), np.nan)

        for paso in range(idx + 1):
            for celda in historial_frentes[paso]:
                matriz[celda[0], celda[1]] = etiquetas.get(celda, 0)

        im = ax.imshow(matriz, cmap="plasma", vmin=0, vmax=max_etiqueta, origin="upper")

        if idx < len(historial_frentes):
            frente_actual = historial_frentes[idx]
            cf = [c for _, c in frente_actual]
            rf = [r for r, _ in frente_actual]
            ax.scatter(cf, rf, c="cyan", s=14, edgecolors="white", linewidths=0.3)

        ax.scatter([inicio[1]], [inicio[0]], c="lime", s=50, marker="o", edgecolors="black")
        ax.scatter([meta[1]], [meta[0]], c="red", s=60, marker="*", edgecolors="black")

        ax.set_title(f"Paso {idx+1}/{total_pasos} (t={idx})", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Propagación del Frente Luminoso — Algoritmo de Lee (8 Instantes)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    if guardar_como:
        plt.savefig(guardar_como, dpi=160, bbox_inches="tight")
    plt.close()


def visualizar_mapa_calor_lee(
    filas: int,
    columnas: int,
    etiquetas: Dict[Tuple[int, int], int],
    camino: List[Tuple[int, int]],
    inicio: Tuple[int, int],
    meta: Tuple[int, int],
    guardar_como: Optional[str] = "lee_mapa_calor.png",
    forzar: bool = False,
):
    """Genera el mapa de calor continuo de distancias mínimas y el camino óptimo."""
    if not forzar and guardar_como and os.path.exists(guardar_como):
        return

    matriz = np.full((filas, columnas), np.nan)
    for (r, c), dist in etiquetas.items():
        matriz[r, c] = dist

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matriz, cmap="inferno", origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tiempo de llegada (Etiqueta)", fontsize=10)

    if camino:
        camino_c = [c for _, c in camino]
        camino_r = [r for r, _ in camino]
        ax.plot(camino_c, camino_r, color="cyan", linewidth=2.2, label=f"Camino ({len(camino)-1} pasos)")

    ax.scatter([inicio[1]], [inicio[0]], c="lime", s=100, marker="o", edgecolors="black", zorder=5, label=f"Inicio {inicio}")
    ax.scatter([meta[1]], [meta[0]], c="red", s=130, marker="*", edgecolors="black", zorder=5, label=f"Meta {meta}")

    ax.set_title(f"Mapa de Calor de Lee y Trayectoria Óptima ({filas}x{columnas})", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")

    plt.tight_layout()
    if guardar_como:
        plt.savefig(guardar_como, dpi=160, bbox_inches="tight")
    plt.close()
