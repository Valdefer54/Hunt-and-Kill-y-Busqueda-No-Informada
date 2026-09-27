"""
Módulo de visualización para el Algoritmo de Lee (Frente de Onda).
Genera visualizaciones de los instantes del frente y mapa de calor de llegada.
"""

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
):
    """
    Dibuja 8 instantes del frente de onda luminosa y el mapa final de llegada
    con el camino reconstruido.
    """
    total_pasos = len(historial_frentes)
    if total_pasos < 8:
        indices = list(range(total_pasos))
    else:
        # Seleccionar 8 instantes representativos distribuidos uniformemente
        indices = [int(i * (total_pasos - 1) / 7) for i in range(8)]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    max_etiqueta = max(etiquetas.values()) if etiquetas else 1

    for i, idx in enumerate(indices):
        ax = axes[i]
        # Crear matriz para renderizar: -1 pared/no visitado, valores positivos para tiempo
        matriz = np.full((filas, columnas), np.nan)

        # Cargar etiquetas acumuladas hasta el instante 'idx'
        for paso in range(idx + 1):
            for celda in historial_frentes[paso]:
                matriz[celda[0], celda[1]] = etiquetas.get(celda, 0)

        # Mapa de color secuencial (plasma/viridis)
        im = ax.imshow(matriz, cmap="plasma", vmin=0, vmax=max_etiqueta, origin="upper")

        # Dibujar frente activo actual en cyan brillante
        if idx < len(historial_frentes):
            frente_actual = historial_frentes[idx]
            cf = [c for _, c in frente_actual]
            rf = [r for r, _ in frente_actual]
            ax.scatter(cf, rf, c="cyan", s=18, edgecolors="white", linewidths=0.5, label="Frente activo")

        # Marcar inicio y meta
        ax.scatter([inicio[1]], [inicio[0]], c="lime", s=70, marker="o", edgecolors="black", label="Inicio")
        ax.scatter([meta[1]], [meta[0]], c="red", s=80, marker="*", edgecolors="black", label="Meta")

        ax.set_title(f"Instante {idx+1}/{total_pasos} (t={idx})", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Propagación del Frente Luminoso — Algoritmo de Lee (8 Instantes)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    if guardar_como:
        plt.savefig(guardar_como, dpi=200, bbox_inches="tight")
        print(f"Figura de Lee guardada exitosamente en: {guardar_como}")
    plt.close()


def visualizar_mapa_calor_lee(
    filas: int,
    columnas: int,
    etiquetas: Dict[Tuple[int, int], int],
    camino: List[Tuple[int, int]],
    inicio: Tuple[int, int],
    meta: Tuple[int, int],
    guardar_como: Optional[str] = "lee_mapa_calor.png",
):
    """Genera el mapa de calor continuo de distancias mínimas y el camino óptimo."""
    matriz = np.full((filas, columnas), np.nan)
    for (r, c), dist in etiquetas.items():
        matriz[r, c] = dist

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matriz, cmap="inferno", origin="upper")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tiempo de llegada (Etiqueta / Distancia Manhattan en Grafo)", fontsize=11)

    # Superponer el camino óptimo reconstruido
    if camino:
        camino_c = [c for _, c in camino]
        camino_r = [r for r, _ in camino]
        ax.plot(camino_c, camino_r, color="cyan", linewidth=2.5, label=f"Camino reconstruido ({len(camino)-1} pasos)")

    ax.scatter([inicio[1]], [inicio[0]], c="lime", s=120, marker="o", edgecolors="black", zorder=5, label=f"Inicio {inicio}")
    ax.scatter([meta[1]], [meta[0]], c="red", s=160, marker="*", edgecolors="black", zorder=5, label=f"Meta {meta}")

    ax.set_title(f"Mapa de Calor de Lee y Trayectoria Óptima ({filas}x{columnas})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")

    plt.tight_layout()
    if guardar_como:
        plt.savefig(guardar_como, dpi=200, bbox_inches="tight")
        print(f"Mapa de calor guardado en: {guardar_como}")
    plt.close()
