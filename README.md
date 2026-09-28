# Taller: Hunt-and-Kill y Búsqueda no Informada

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Valdefer54/Hunt-and-Kill-y-Busqueda-No-Informada/blob/main/Taller_Hunt_and_Kill_Busqueda_No_Informada.ipynb)
[![View on nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/Valdefer54/Hunt-and-Kill-y-Busqueda-No-Informada/blob/main/Taller_Hunt_and_Kill_Busqueda_No_Informada.ipynb)

Proyecto del curso de Inteligencia Artificial (basado en el capitulo 3 del libro de Russell y Norvig, AIMA).

En este taller trabajamos sobre un generador de laberintos perfectos (Hunt-and-Kill) y resolvemos la busqueda de caminos entre dos celdas implementando los algoritmos de tres maneras diferentes:
1. Desde cero: sin librerias de busqueda, usando solo estructuras estandar de Python (deque, listas, diccionarios, heapq).
2. SimpleAI: adaptando el laberinto como un SearchProblem.
3. AIMA-Python: adaptandolo como una subclase de Problem.

Tip si GitHub tarda en cargar:
Los cuadernos .ipynb a veces se quedan cargando en la web de GitHub. Si te pasa, puedes abrirlo en 1 segundo usando el boton de Open in Colab o el de nbviewer de arriba.

---

## Estructura del proyecto

- Taller_Hunt_and_Kill_Busqueda_No_Informada.ipynb: Cuaderno Jupyter principal con todo el desarrollo, explicaciones y resultados.
- src/: Modulos en Python para mantener el codigo ordenado y reutilizable:
  - busqueda_cero.py: Implementacion desde cero de DFS, BFS, UCS, DLS, IDDFS, Busqueda Bidireccional y Lee.
  - adaptador_simpleai.py: Adaptador para correr los algoritmos con SimpleAI y extraer las metricas con un viewer propio.
  - adaptador_aima.py: Adaptador para correr con AIMA-Python.
  - pruebas_aceptacion.py: Bateria de pruebas automaticas (caminos validos, casos limite y concordancia).
  - topologias_costos.py: Funciones para abrir paredes (meter ciclos) y asignar costos desiguales a los pasillos.
  - protocolo_experimental.py: Codigo para correr los experimentos, calcular medianas/IQR y sacar las graficas.
  - visualizador_lee.py: Genera los 8 instantes de la onda de Lee y el mapa de calor.
- datos_experimentos.csv: Datos de las pruebas experimentales para que el cuaderno cargue rapido sin tener que recalcular todo desde cero.
- aima/: Archivos base de busqueda de la libreria oficial de AIMA.

---

## Como correrlo localmente

1. Clonar el repo:
```bash
git clone https://github.com/Valdefer54/Hunt-and-Kill-y-Busqueda-No-Informada.git
cd Hunt-and-Kill-y-Busqueda-No-Informada
```

2. Crear un entorno virtual e instalar las librerias:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Abrir el cuaderno:
```bash
jupyter notebook Taller_Hunt_and_Kill_Busqueda_No_Informada.ipynb
```

---

## Datos de mi instancia individual

- Codigo: 202410057
- Semilla: 10057 (ultimos 5 digitos)
- Tamano: 27 filas x 25 columnas
- Inicio: (3, 4) (cuadrante superior izquierdo)
- Meta: (23, 20) (cuadrante inferior derecho)
- Total aristas: 674 (cumple |V| - 1)

---

## Graficas que generamos

El proyecto guarda las 6 figuras solicitadas en la guia:
1. grafica1_expandidos_vs_estados.png: Nodos expandidos segun el tamano del laberinto.
2. grafica2_frontera_vs_profundidad.png: Cuantos nodos se acumulan en memoria en la frontera.
3. grafica3_tiempo_vs_tamano.png: Comparativa de tiempos de busqueda en milisegundos.
4. grafica4_costo_y_longitud.png: Comparacion de longitud de ruta con arbol vs con ciclos.
5. grafica5_mapa_calor_lee.png: Mapa de calor con la onda de propagacion de Lee.
6. grafica6_comparativa_versiones.png: Comparativa cruzada entre nuestra version desde cero, SimpleAI y AIMA.
