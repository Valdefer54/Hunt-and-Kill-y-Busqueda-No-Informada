# Hunt-and-Kill y Búsqueda no Informada

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Valdefer54/Hunt-and-Kill-y-Busqueda-No-Informada/blob/main/Taller_Hunt_and_Kill_Busqueda_No_Informada.ipynb)
[![View on nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/Valdefer54/Hunt-and-Kill-y-Busqueda-No-Informada/blob/main/Taller_Hunt_and_Kill_Busqueda_No_Informada.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![AIMA Reference](https://img.shields.io/badge/AIMA-Capítulo%203-brightgreen.svg)](https://aima.cs.berkeley.edu/)

Proyecto integral del curso de **Inteligencia Artificial** centrado en la generación de laberintos perfectos mediante el algoritmo **Hunt-and-Kill** y la implementación, verificación experimental y análisis comparativo de algoritmos de **búsqueda no informada**.

> 💡 **Nota de carga rápida:**
> - **En GitHub:** Si la vista previa de GitHub tarda en renderizar el archivo `.ipynb`, haz clic en el botón superior **"Open in Colab"** o **"Render on nbviewer"** para visualizar y ejecutar de inmediato.
> - **En el Cuaderno:** La Celda 27 incluye carga precalculada de los 900 experimentos (`datos_experimentos.csv`), permitiendo que el cuaderno se ejecute completo en menos de 5 segundos.

---

## 📌 Contenido y Estructura del Proyecto

El proyecto implementa el contrato común de búsqueda en **tres versiones independientes y comparables**:

1. **Versión 1 — Desde Cero (`src/busqueda_cero.py`):**
   - Implementación pura sin bibliotecas de grafos ni de búsqueda.
   - Algoritmos: **DFS (iterativo)**, **BFS**, **UCS (Costo Uniforme)**, **DLS (Profundidad Limitada)**, **IDDFS (Profundización Iterativa)**, **Búsqueda Bidireccional** y **Algoritmo de Lee (Frente de Onda)**.
2. **Versión 2 — SimpleAI (`src/adaptador_simpleai.py`):**
   - Subclase de `simpleai.search.SearchProblem` con estados inmutables.
   - Instrumentación desacoplada mediante un `BaseViewer` personalizado sin modificar la biblioteca instalada.
3. **Versión 3 — AIMA-Python (`src/adaptador_aima.py`):**
   - Subclase de `aima.search.Problem` adaptada a la referencia oficial de Russell & Norvig.
   - Instrumentación pura que captura nodos expandidos y generados.

```text
.
├── Taller_Hunt_and_Kill_Busqueda_No_Informada.ipynb   # Cuaderno principal con las 14 secciones
├── datos_experimentos.csv                            # Base de datos precomputada del benchmark factorial
├── requirements.txt                                  # Dependencias del proyecto
├── src/                                              # Código fuente modular
│   ├── auditoria_generador.py                        # Pruebas automáticas del generador Hunt-and-Kill
│   ├── comun.py                                      # Contrato ResultadoBusqueda y clase Nodo
│   ├── busqueda_cero.py                              # Algoritmos implementados desde cero
│   ├── visualizador_lee.py                           # Visualizaciones del frente y mapa de calor de Lee
│   ├── adaptador_simpleai.py                         # Adaptador e instrumentación para SimpleAI
│   ├── adaptador_aima.py                             # Adaptador e instrumentación para AIMA-Python
│   ├── pruebas_aceptacion.py                         # Suite de pruebas de aceptación y casos límite
│   ├── topologias_costos.py                          # Transformaciones con ciclos y costos ponderados
│   └── protocolo_experimental.py                     # Benchmark factorial (mediana e IQR) y gráficas
├── aima/                                             # Módulo local de referencia de AIMA-Python
├── grafica1_expandidos_vs_estados.png                # Gráfica obligatoria 1
├── grafica2_frontera_vs_profundidad.png              # Gráfica obligatoria 2
├── grafica3_tiempo_vs_tamano.png                     # Gráfica obligatoria 3
├── grafica4_costo_y_longitud.png                     # Gráfica obligatoria 4
├── grafica5_mapa_calor_lee.png                       # Gráfica obligatoria 5
├── grafica6_comparativa_versiones.png                # Gráfica obligatoria 6
├── lee_frente_onda.png                               # 8 instantes del frente de onda de Lee
└── lee_mapa_calor.png                                # Mapa de calor de llegada de Lee
```

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/Valdefer54/Hunt-and-Kill-y-Busqueda-No-Informada.git
cd Hunt-and-Kill-y-Busqueda-No-Informada
```

### 2. Crear y activar entorno virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Ejecutar el Cuaderno
```bash
jupyter notebook Taller_Hunt_and_Kill_Busqueda_No_Informada.ipynb
```

---

## 📊 Parámetros de la Instancia Individual

- **Código Estudiantil:** `202410057`
- **Semilla:** `10057`
- **Dimensiones:** $27 \times 25$ ($675$ celdas)
- **Inicio:** `(3, 4)` (Cuadrante Superior Izquierdo)
- **Meta:** `(23, 20)` (Cuadrante Inferior Derecho)
- **Aristas del Árbol:** $674$ ($|V| - 1$)

---

## 📈 Gráficas y Visualizaciones Obligatorias

El proyecto genera automáticamente las 6 visualizaciones exigidas por la rúbrica:
1. **Nodos Expandidos vs Número de Estados:** Evaluación de escalabilidad con mediana e IQR.
2. **Frontera Máxima vs Profundidad:** Comparación del consumo de memoria (espacio lineal en DFS vs exponencial en BFS).
3. **Tiempo de Búsqueda frente a Tamaño:** Rendimiento en milisegundos a través de diferentes órdenes de cuadrícula.
4. **Costo y Longitud por Topología:** Evidencia de cómo los ciclos diferencian a BFS de DFS, y cómo los costos ponderados diferencian a UCS de BFS.
5. **Mapa de Calor de Lee:** Distribución temporal de frentes isotrópicos con superposición de la ruta geodésica.
6. **Comparativa Cruzada de Versiones:** Rendimiento comparado entre *Desde Cero*, *SimpleAI* y *AIMA-Python*.

---

## 📜 Licencia y Autoría

Desarrollado para el curso de **Inteligencia Artificial** (Referencia: Russell & Norvig, AIMA 4.ª Edición).  
Repositorio oficial: [https://github.com/Valdefer54/Hunt-and-Kill-y-Busqueda-No-Informada](https://github.com/Valdefer54/Hunt-and-Kill-y-Busqueda-No-Informada)
