#!/usr/bin/env python3
"""
build_solution.py - Versión Optimizada
Ejecuta todos los algoritmos del Taller TSP de forma eficiente:
- A* y RBFS con Prim O(|U|^2) y corte rápido por presupuesto.
- 2-opt con delta incremental O(1).
- Hill Climbing (steepest descent y random restarts).
- Simulated Annealing calibrado.
- Algoritmo Genético optimizado (selección rápida y cruce OX vectorizado).
- Generación de figuras obligatorias y tablas estadísticas.
"""

import os
import sys
import math
import time
import json
import base64
import random
import tracemalloc
import itertools
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tsplib95
import pandas as pd

# Fijar semillas
np.random.seed(42)
random.seed(42)

# ==============================================================================
# 1. MODELO Y ESTRUCTURAS DE DATOS
# ==============================================================================

class TSPInstance:
    """Representa una instancia TSPLIB con su matriz simétrica de costos D."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.problem = tsplib95.load(filepath)
        self.name = self.problem.name
        self.comment = getattr(self.problem, 'comment', '')
        self.dimension = self.problem.dimension
        self.edge_weight_type = self.problem.edge_weight_type
        self.edge_weight_format = getattr(self.problem, 'edge_weight_format', None)
        self.node_ids = sorted(list(self.problem.get_nodes()))
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        
        # Matriz simétrica D
        self.D = np.zeros((self.dimension, self.dimension), dtype=np.int64)
        for i, u in enumerate(self.node_ids):
            for j, v in enumerate(self.node_ids):
                if i != j:
                    self.D[i, j] = int(self.problem.get_weight(u, v))
                else:
                    self.D[i, j] = 0
                    
        # Coordenadas 2D si existen
        self.coords = None
        if hasattr(self.problem, 'node_coords') and self.problem.node_coords:
            self.coords = np.array([self.problem.node_coords[nid] for nid in self.node_ids])
        elif hasattr(self.problem, 'display_data') and self.problem.display_data:
            self.coords = np.array([self.problem.display_data[nid] for nid in self.node_ids])

    def get_subproblem(self, k):
        sub_D = self.D[:k, :k].copy()
        sub_coords = self.coords[:k].copy() if self.coords is not None else None
        sub_node_ids = self.node_ids[:k]
        return sub_D, sub_node_ids, sub_coords

    def audit(self):
        is_symmetric = bool(np.all(self.D == self.D.T))
        zero_diag = bool(np.all(np.diag(self.D) == 0))
        non_negative = bool(np.all(self.D >= 0))
        min_weight = int(np.min(self.D[self.D > 0])) if np.any(self.D > 0) else 0
        max_weight = int(np.max(self.D))
        return {
            'name': self.name,
            'dimension': self.dimension,
            'weight_type': self.edge_weight_type,
            'format': str(self.edge_weight_format),
            'symmetric': is_symmetric,
            'zero_diagonal': zero_diag,
            'non_negative': non_negative,
            'min_weight': min_weight,
            'max_weight': max_weight,
            'has_coords': self.coords is not None
        }

def tour_cost(route, D):
    n = len(route)
    return int(sum(D[route[i], route[(i + 1) % n]] for i in range(n)))

def validate_tour(route, n):
    if len(route) != n:
        return False, f"Longitud incorrecta: {len(route)} != {n}"
    if len(set(route)) != n:
        return False, "Ciudades duplicadas u omitidas"
    if min(route) != 0 or max(route) != n - 1:
        return False, "Índices fuera de rango"
    return True, "Válido"

# ==============================================================================
# 2. HEURÍSTICA MST Y PRIM O(|U|^2)
# ==============================================================================

def prim_mst(U, D):
    if len(U) <= 1:
        return 0
    nodes = list(U)
    in_tree = {nodes[0]}
    mst_cost = 0
    min_edge = {v: D[nodes[0], v] for v in nodes[1:]}
    
    while len(in_tree) < len(nodes):
        best_v = None
        best_w = float('inf')
        for v in nodes:
            if v not in in_tree and min_edge[v] < best_w:
                best_w = min_edge[v]
                best_v = v
        mst_cost += best_w
        in_tree.add(best_v)
        for v in nodes:
            if v not in in_tree:
                w = D[best_v, v]
                if w < min_edge[v]:
                    min_edge[v] = w
    return int(mst_cost)

def mst_heuristic(v, U, s, D, closed):
    if closed:
        return 0
    if not U:
        return int(D[v, s])
    mst = prim_mst(U, D)
    min_v_U = min(D[v, u] for u in U)
    min_U_s = min(D[u, s] for u in U)
    return int(mst + min_v_U + min_U_s)

# ==============================================================================
# 3. BÚSQUEDA EXACTA: A* Y RBFS
# ==============================================================================

import heapq

class SearchNode:
    __slots__ = ('last', 'remaining', 'g', 'h', 'f', 'parent', 'closed', 'depth')
    def __init__(self, last, remaining, g, h, parent=None, closed=False, depth=0):
        self.last = last
        self.remaining = remaining
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.closed = closed
        self.depth = depth

    def key(self):
        return (self.last, self.remaining, self.closed)

def get_successors(node, s, D):
    if node.closed:
        return []
    if not node.remaining:
        g_close = node.g + D[node.last, s]
        return [SearchNode(s, frozenset(), g_close, 0, parent=node, closed=True, depth=node.depth + 1)]
    successors = []
    for v in sorted(node.remaining):
        rem = node.remaining - {v}
        g_child = node.g + D[node.last, v]
        h_child = mst_heuristic(v, rem, s, D, False)
        child = SearchNode(v, rem, g_child, h_child, parent=node, closed=False, depth=node.depth + 1)
        successors.append(child)
    return successors

def a_star_tsp(n, D, s=0, max_expansions=50000, time_limit=5.0, max_memory_mb=256):
    start_time = time.time()
    tracemalloc.start()
    
    root_rem = frozenset(i for i in range(n) if i != s)
    h_root = mst_heuristic(s, root_rem, s, D, False)
    root = SearchNode(s, root_rem, 0, h_root, None, False, 0)
    
    counter = 0
    open_heap = [(root.f, -root.depth, counter, root)]
    best_g = {root.key(): root.g}
    
    expanded = 0
    generated = 1
    stale_entries = 0
    peak_frontier = 1
    stop_reason = 'EXHAUSTED'
    
    while open_heap:
        if expanded >= max_expansions:
            stop_reason = 'BUDGET_EXPANSIONS'
            break
        if (time.time() - start_time) >= time_limit:
            stop_reason = 'BUDGET_TIME'
            break
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        if peak_mem >= max_memory_mb * 1024 * 1024:
            stop_reason = 'BUDGET_MEMORY'
            break
            
        f, neg_depth, _, node = heapq.heappop(open_heap)
        
        if node.g > best_g.get(node.key(), float('inf')):
            stale_entries += 1
            continue
            
        if node.closed:
            path = []
            curr = node
            while curr is not None:
                path.append(curr.last)
                curr = curr.parent
            path.reverse()
            elapsed = time.time() - start_time
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return {
                'tour': path[:-1],
                'cost': int(node.g),
                'status': 'OPTIMAL',
                'stop_reason': 'GOAL_FOUND',
                'expanded': expanded,
                'generated': generated,
                'stale_entries': stale_entries,
                'peak_frontier': peak_frontier,
                'elapsed': elapsed,
                'peak_mem_mb': peak_mem / (1024 * 1024)
            }
            
        expanded += 1
        for child in get_successors(node, s, D):
            generated += 1
            ckey = child.key()
            if child.g < best_g.get(ckey, float('inf')):
                best_g[ckey] = child.g
                counter += 1
                heapq.heappush(open_heap, (child.f, -child.depth, counter, child))
                
        if len(open_heap) > peak_frontier:
            peak_frontier = len(open_heap)
            
    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'tour': None,
        'cost': None,
        'status': 'INTERRUPTED',
        'stop_reason': stop_reason,
        'expanded': expanded,
        'generated': generated,
        'stale_entries': stale_entries,
        'peak_frontier': peak_frontier,
        'elapsed': elapsed,
        'peak_mem_mb': peak_mem / (1024 * 1024)
    }

class RBFSSolver:
    def __init__(self, n, D, s=0, max_expansions=5000, time_limit=2.0, max_memory_mb=256):
        self.n = n
        self.D = D
        self.s = s
        self.max_expansions = max_expansions
        self.time_limit = time_limit
        self.max_memory_mb = max_memory_mb
        self.start_time = None
        self.expansions = 0
        self.generated = 0
        self.recursive_calls = 0
        self.regenerations = 0
        self.max_depth = 0
        self.stop_reason = 'EXHAUSTED'
        self.visited_prefixes = set()

    def solve(self):
        self.start_time = time.time()
        tracemalloc.start()
        
        root_rem = frozenset(i for i in range(self.n) if i != self.s)
        h_root = mst_heuristic(self.s, root_rem, self.s, self.D, False)
        root = SearchNode(self.s, root_rem, 0, h_root, None, False, 0)
        self.generated = 1
        
        result, backed_f, goal_node = self._rbfs(root, float('inf'))
        elapsed = time.time() - self.start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if result == 'FOUND':
            path = []
            curr = goal_node
            while curr is not None:
                path.append(curr.last)
                curr = curr.parent
            path.reverse()
            return {
                'tour': path[:-1],
                'cost': int(backed_f),
                'status': 'OPTIMAL',
                'stop_reason': 'GOAL_FOUND',
                'expanded': self.expansions,
                'generated': self.generated,
                'regenerations': self.regenerations,
                'recursive_calls': self.recursive_calls,
                'max_depth': self.max_depth,
                'elapsed': elapsed,
                'peak_mem_mb': peak_mem / (1024 * 1024)
            }
        else:
            return {
                'tour': None,
                'cost': None,
                'status': 'INTERRUPTED',
                'stop_reason': self.stop_reason,
                'expanded': self.expansions,
                'generated': self.generated,
                'regenerations': self.regenerations,
                'recursive_calls': self.recursive_calls,
                'max_depth': self.max_depth,
                'elapsed': elapsed,
                'peak_mem_mb': peak_mem / (1024 * 1024)
            }

    def _rbfs(self, node, limit):
        self.recursive_calls += 1
        if self.expansions >= self.max_expansions:
            self.stop_reason = 'BUDGET_EXPANSIONS'
            return 'INTERRUPTED', node.f, None
        if (time.time() - self.start_time) >= self.time_limit:
            self.stop_reason = 'BUDGET_TIME'
            return 'INTERRUPTED', node.f, None
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        if peak_mem >= self.max_memory_mb * 1024 * 1024:
            self.stop_reason = 'BUDGET_MEMORY'
            return 'INTERRUPTED', node.f, None
            
        if node.depth > self.max_depth:
            self.max_depth = node.depth
            
        if node.closed:
            return 'FOUND', node.f, node
            
        prefix = []
        curr = node
        while curr is not None:
            prefix.append(curr.last)
            curr = curr.parent
        prefix_tuple = tuple(reversed(prefix))
        if prefix_tuple in self.visited_prefixes:
            self.regenerations += 1
        else:
            self.visited_prefixes.add(prefix_tuple)
            
        self.expansions += 1
        children = get_successors(node, self.s, self.D)
        self.generated += len(children)
        if not children:
            return 'FAILURE', float('inf'), None
            
        for child in children:
            child.f = max(child.g + child.h, node.f)
            
        while True:
            children.sort(key=lambda c: (c.f, -c.depth, c.last))
            best = children[0]
            if best.f > limit or best.f == float('inf'):
                return 'FAILURE', best.f, None
            alternative = children[1].f if len(children) > 1 else float('inf')
            
            result, backed_f, goal = self._rbfs(best, min(limit, alternative))
            if result == 'INTERRUPTED':
                return 'INTERRUPTED', backed_f, None
            best.f = backed_f
            if result == 'FOUND':
                return 'FOUND', best.f, goal

# ==============================================================================
# 4. BÚSQUEDA LOCAL: 2-OPT, HILL CLIMBING Y SIMULATED ANNEALING
# ==============================================================================

def two_opt_delta(route, i, j, D):
    n = len(route)
    a, b = route[i], route[i + 1]
    c, d = route[j], route[(j + 1) % n]
    return int(D[a, c] + D[b, d] - D[a, b] - D[c, d])

def two_opt_move(route, i, j):
    candidate = route.copy()
    candidate[i + 1:j + 1] = reversed(candidate[i + 1:j + 1])
    return candidate

def get_valid_2opt_pairs(n):
    pairs = []
    for i in range(0, n - 2):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            pairs.append((i, j))
    return pairs

def hill_climbing(initial_tour, D, max_evaluations=10000, time_limit=10.0):
    start_time = time.time()
    tracemalloc.start()
    
    n = len(initial_tour)
    pairs = get_valid_2opt_pairs(n)
    current_tour = initial_tour.copy()
    current_cost = tour_cost(current_tour, D)
    
    evaluations = 0
    history = [(evaluations, current_cost, current_cost, time.time() - start_time)]
    stop_reason = 'LOCAL_OPTIMUM'
    
    while True:
        best_delta = 0
        best_pair = None
        
        for i, j in pairs:
            if evaluations >= max_evaluations:
                stop_reason = 'BUDGET_EVALUATIONS'
                break
            if (time.time() - start_time) >= time_limit:
                stop_reason = 'BUDGET_TIME'
                break
                
            delta = two_opt_delta(current_tour, i, j, D)
            evaluations += 1
            if delta < best_delta:
                best_delta = delta
                best_pair = (i, j)
                
        if stop_reason != 'LOCAL_OPTIMUM' or best_pair is None:
            break
            
        current_tour = two_opt_move(current_tour, best_pair[0], best_pair[1])
        current_cost += best_delta
        history.append((evaluations, current_cost, current_cost, time.time() - start_time))
        
    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'tour': current_tour,
        'cost': current_cost,
        'status': stop_reason,
        'evaluations': evaluations,
        'elapsed': elapsed,
        'peak_mem_mb': peak_mem / (1024 * 1024),
        'history': history
    }

def hill_climbing_restarts(n, D, max_evaluations=10000, time_limit=10.0, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    start_time = time.time()
    tracemalloc.start()
    
    pairs = get_valid_2opt_pairs(n)
    total_evals = 0
    best_tour = None
    best_cost = float('inf')
    history = []
    stop_reason = 'BUDGET_EVALUATIONS'
    
    while total_evals < max_evaluations and (time.time() - start_time) < time_limit:
        perm = rng.permutation(range(1, n)).tolist()
        current_tour = [0] + perm
        current_cost = tour_cost(current_tour, D)
        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour.copy()
            
        history.append((total_evals, current_cost, best_cost, time.time() - start_time))
        
        while total_evals < max_evaluations and (time.time() - start_time) < time_limit:
            best_delta = 0
            best_pair = None
            for i, j in pairs:
                if total_evals >= max_evaluations or (time.time() - start_time) >= time_limit:
                    break
                delta = two_opt_delta(current_tour, i, j, D)
                total_evals += 1
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i, j)
            if best_pair is None or total_evals >= max_evaluations or (time.time() - start_time) >= time_limit:
                break
            current_tour = two_opt_move(current_tour, best_pair[0], best_pair[1])
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour.copy()
            history.append((total_evals, current_cost, best_cost, time.time() - start_time))
            
    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'tour': best_tour,
        'cost': best_cost,
        'status': stop_reason,
        'evaluations': total_evals,
        'elapsed': elapsed,
        'peak_mem_mb': peak_mem / (1024 * 1024),
        'history': history
    }

def calibrate_t0(initial_tour, D, p0=0.8, sample_size=100, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(initial_tour)
    pairs = get_valid_2opt_pairs(n)
    pos_deltas = []
    
    sample_indices = rng.choice(len(pairs), size=min(sample_size, len(pairs)), replace=False)
    for idx in sample_indices:
        i, j = pairs[idx]
        delta = two_opt_delta(initial_tour, i, j, D)
        if delta > 0:
            pos_deltas.append(delta)
            
    if pos_deltas:
        med = np.median(pos_deltas)
        T0 = -med / math.log(p0)
    else:
        T0 = 100.0
    return max(1.0, float(T0))

def simulated_annealing(initial_tour, D, max_evaluations=10000, time_limit=10.0,
                        alpha=0.98, p0=0.8, steps_per_temp=100, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    start_time = time.time()
    tracemalloc.start()
    
    n = len(initial_tour)
    pairs = get_valid_2opt_pairs(n)
    num_pairs = len(pairs)
    
    current_tour = initial_tour.copy()
    current_cost = tour_cost(current_tour, D)
    best_tour = current_tour.copy()
    best_cost = current_cost
    
    T0 = calibrate_t0(initial_tour, D, p0=p0, rng=rng)
    T = T0
    T_min = 1e-4
    
    evaluations = 0
    history = []
    detailed_trace = []
    stop_reason = 'BUDGET_EVALUATIONS'
    
    while T > T_min and evaluations < max_evaluations and (time.time() - start_time) < time_limit:
        for _ in range(steps_per_temp):
            if evaluations >= max_evaluations:
                stop_reason = 'BUDGET_EVALUATIONS'
                break
            if (time.time() - start_time) >= time_limit:
                stop_reason = 'BUDGET_TIME'
                break
                
            pair_idx = rng.randint(0, num_pairs)
            i, j = pairs[pair_idx]
            delta = two_opt_delta(current_tour, i, j, D)
            evaluations += 1
            
            accepted = False
            is_worse = (delta > 0)
            
            if delta <= 0:
                accepted = True
            else:
                prob = math.exp(-delta / T)
                if rng.rand() < prob:
                    accepted = True
                    
            if accepted:
                current_tour = two_opt_move(current_tour, i, j)
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = current_tour.copy()
                    
            detailed_trace.append((evaluations, T, current_cost, best_cost, accepted, is_worse))
            if evaluations % 100 == 0:
                history.append((evaluations, current_cost, best_cost, time.time() - start_time))
                
        T *= alpha
        
    history.append((evaluations, current_cost, best_cost, time.time() - start_time))
    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'tour': best_tour,
        'cost': best_cost,
        'status': stop_reason,
        'evaluations': evaluations,
        'elapsed': elapsed,
        'peak_mem_mb': peak_mem / (1024 * 1024),
        'history': history,
        'detailed_trace': detailed_trace
    }

# ==============================================================================
# 5. ALGORITMO GENÉTICO OPTIMIZADO
# ==============================================================================

def fast_ox(p1, p2, left, right):
    size = len(p1)
    child = [None] * size
    child[left:right + 1] = p1[left:right + 1]
    in_child = set(p1[left:right + 1])
    
    c_idx = (right + 1) % size
    p2_idx = (right + 1) % size
    for _ in range(size):
        gene = p2[p2_idx]
        if gene not in in_child:
            child[c_idx] = gene
            in_child.add(gene)
            c_idx = (c_idx + 1) % size
        p2_idx = (p2_idx + 1) % size
    return child

def fast_inversion(chromosome, rng):
    size = len(chromosome)
    if size < 2:
        return chromosome.copy()
    i = rng.randint(0, size - 2)
    j = rng.randint(i + 1, size - 1)
    mutant = chromosome.copy()
    mutant[i:j + 1] = reversed(mutant[i:j + 1])
    return mutant

def population_diversity(population):
    unique_count = len(set(tuple(ind) for ind in population))
    return unique_count / len(population)

def genetic_algorithm(n, D, max_evaluations=10000, time_limit=10.0,
                      pop_size=40, elite_count=2, tournament_k=3,
                      pc=0.9, pm=0.2, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    start_time = time.time()
    tracemalloc.start()
    
    genes = list(range(1, n))
    chrom_len = len(genes)
    
    population = []
    fitnesses = []
    evaluations = 0
    
    for _ in range(pop_size):
        chrom = rng.permutation(genes).tolist()
        cost = tour_cost([0] + chrom, D)
        evaluations += 1
        population.append(chrom)
        fitnesses.append(cost)
        
    best_idx = int(np.argmin(fitnesses))
    best_cost = fitnesses[best_idx]
    best_tour = [0] + population[best_idx].copy()
    
    history = [(evaluations, float(np.median(fitnesses)), best_cost, time.time() - start_time)]
    diversity_trace = [(0, best_cost, float(np.median(fitnesses)), population_diversity(population))]
    
    generation = 0
    stop_reason = 'BUDGET_EVALUATIONS'
    
    while evaluations < max_evaluations and (time.time() - start_time) < time_limit:
        generation += 1
        sorted_indices = np.argsort(fitnesses)
        
        next_population = [population[idx].copy() for idx in sorted_indices[:elite_count]]
        next_fitnesses = [fitnesses[idx] for idx in sorted_indices[:elite_count]]
        
        while len(next_population) < pop_size:
            if evaluations >= max_evaluations or (time.time() - start_time) >= time_limit:
                break
                
            # Fast tournament
            i1, i2, i3 = rng.randint(0, pop_size), rng.randint(0, pop_size), rng.randint(0, pop_size)
            p1 = population[i1 if fitnesses[i1] <= fitnesses[i2] and fitnesses[i1] <= fitnesses[i3] else (i2 if fitnesses[i2] <= fitnesses[i3] else i3)]
            
            j1, j2, j3 = rng.randint(0, pop_size), rng.randint(0, pop_size), rng.randint(0, pop_size)
            p2 = population[j1 if fitnesses[j1] <= fitnesses[j2] and fitnesses[j1] <= fitnesses[j3] else (j2 if fitnesses[j2] <= fitnesses[j3] else j3)]
            
            if rng.rand() < pc and chrom_len >= 2:
                left = rng.randint(0, chrom_len - 2)
                right = rng.randint(left + 1, chrom_len - 1)
                c1 = fast_ox(p1, p2, left, right)
                c2 = fast_ox(p2, p1, left, right)
            else:
                c1, c2 = p1.copy(), p2.copy()
                
            for child in [c1, c2]:
                if len(next_population) >= pop_size or evaluations >= max_evaluations:
                    break
                if rng.rand() < pm:
                    child = fast_inversion(child, rng)
                    
                cost = tour_cost([0] + child, D)
                evaluations += 1
                next_population.append(child)
                next_fitnesses.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_tour = [0] + child.copy()
                    
        population = next_population
        fitnesses = next_fitnesses
        div = population_diversity(population)
        med_cost = float(np.median(fitnesses)) if fitnesses else best_cost
        diversity_trace.append((generation, best_cost, med_cost, div))
        history.append((evaluations, med_cost, best_cost, time.time() - start_time))
        
    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'tour': best_tour,
        'cost': best_cost,
        'status': stop_reason,
        'evaluations': evaluations,
        'elapsed': elapsed,
        'peak_mem_mb': peak_mem / (1024 * 1024),
        'history': history,
        'diversity_trace': diversity_trace
    }

# ==============================================================================
# 6. BANCO EXPERIMENTAL: ESCALAS A Y B
# ==============================================================================

def run_scale_a(instances, sizes=(6, 8, 10, 12)):
    results = []
    print("\n--- EJECUTANDO ESCALA A (SUBPROBLEMAS k in {6, 8, 10, 12}) ---")
    for inst_name, inst in instances.items():
        print(f"\nInstancia: {inst_name}")
        for k in sizes:
            sub_D, sub_node_ids, _ = inst.get_subproblem(k)
            
            # Exact method A* (budget 5.0s, 50k expansions)
            res_astar = a_star_tsp(k, sub_D, s=0, max_expansions=50000, time_limit=5.0)
            opt_cost = res_astar['cost']
            
            # Exact method RBFS (budget 2.0s, 5k expansions to gracefully detect thrashing)
            solver_rbfs = RBFSSolver(k, sub_D, s=0, max_expansions=5000, time_limit=2.0)
            res_rbfs = solver_rbfs.solve()
            
            entry = {
                'instance': inst_name,
                'k': k,
                'opt_cost': opt_cost if opt_cost is not None else 'N/A',
                'nodes': str(sub_node_ids),
                'astar_exp': res_astar['expanded'],
                'astar_time': res_astar['elapsed'],
                'astar_mem': res_astar['peak_mem_mb'],
                'rbfs_exp': res_rbfs['expanded'],
                'rbfs_time': res_rbfs['elapsed'],
                'rbfs_mem': res_rbfs['peak_mem_mb'],
                'rbfs_regen': res_rbfs['regenerations'],
            }
            
            # Approximate methods across 10 seeds (0..9) with 1,000 evals budget
            approx_methods = ['HC_single', 'HC_restarts', 'SA', 'GA']
            for m in approx_methods:
                costs = []
                times = []
                evals = []
                for seed in range(10):
                    rng_init = np.random.RandomState(seed)
                    init_perm = [0] + rng_init.permutation(range(1, k)).tolist()
                    rng_run = np.random.RandomState(seed + 100)
                    
                    if m == 'HC_single':
                        r = hill_climbing(init_perm, sub_D, max_evaluations=1000, time_limit=1.0)
                    elif m == 'HC_restarts':
                        r = hill_climbing_restarts(k, sub_D, max_evaluations=1000, time_limit=1.0, rng=rng_run)
                    elif m == 'SA':
                        r = simulated_annealing(init_perm, sub_D, max_evaluations=1000, time_limit=1.0, rng=rng_run)
                    elif m == 'GA':
                        r = genetic_algorithm(k, sub_D, max_evaluations=1000, time_limit=1.0, pop_size=20, rng=rng_run)
                    costs.append(r['cost'])
                    times.append(r['elapsed'])
                    evals.append(r['evaluations'])
                    
                entry[f'{m}_best_cost'] = min(costs)
                entry[f'{m}_mean_cost'] = np.mean(costs)
                entry[f'{m}_std_cost'] = np.std(costs)
                if opt_cost is not None:
                    entry[f'{m}_gap_med'] = 100.0 * (np.median(costs) - opt_cost) / opt_cost
                    entry[f'{m}_match_opt'] = sum(1 for c in costs if c == opt_cost) / 10.0
                else:
                    entry[f'{m}_gap_med'] = 'N/A'
                    entry[f'{m}_match_opt'] = 0.0
                entry[f'{m}_med_time'] = np.median(times)
                
            results.append(entry)
            print(f"  k={k:2d} | Opt: {entry['opt_cost']} | A* exp: {entry['astar_exp']:5d} | RBFS exp: {entry['rbfs_exp']:5d} | HC gap: {entry['HC_restarts_gap_med']}%")
            
    return pd.DataFrame(results)

def run_scale_b(instances, official_opts, max_evals=6000):
    scale_b_runs = []
    diagnostic_runs = []
    
    print("\n--- EJECUTANDO ESCALA B (INSTANCIAS COMPLETAS) ---")
    for inst_name, inst in instances.items():
        opt = official_opts[inst_name]
        n = inst.dimension
        print(f"\n>>> Instancia completa: {inst_name} (n={n}, Óptimo oficial={opt})")
        
        # Ensayos diagnósticos A* y RBFS (límite: 2,000 expansiones o 2s)
        diag_astar = a_star_tsp(n, inst.D, s=0, max_expansions=2000, time_limit=2.0)
        diagnostic_runs.append({
            'instance': inst_name,
            'algorithm': 'A* + MST',
            'dimension': n,
            'status': diag_astar['status'],
            'stop_reason': diag_astar['stop_reason'],
            'expanded': diag_astar['expanded'],
            'generated': diag_astar['generated'],
            'frontier_peak': diag_astar['peak_frontier'],
            'elapsed_sec': diag_astar['elapsed'],
            'peak_mem_mb': diag_astar['peak_mem_mb']
        })
        print(f"  A* diag -> {diag_astar['stop_reason']} ({diag_astar['expanded']} exp, {diag_astar['elapsed']:.2f}s, {diag_astar['peak_mem_mb']:.2f}MB)")
        
        solver_rbfs = RBFSSolver(n, inst.D, s=0, max_expansions=2000, time_limit=2.0)
        diag_rbfs = solver_rbfs.solve()
        diagnostic_runs.append({
            'instance': inst_name,
            'algorithm': 'RBFS + MST',
            'dimension': n,
            'status': diag_rbfs['status'],
            'stop_reason': diag_rbfs['stop_reason'],
            'expanded': diag_rbfs['expanded'],
            'generated': diag_rbfs['generated'],
            'frontier_peak': diag_rbfs['max_depth'],
            'elapsed_sec': diag_rbfs['elapsed'],
            'peak_mem_mb': diag_rbfs['peak_mem_mb']
        })
        print(f"  RBFS diag -> {diag_rbfs['stop_reason']} ({diag_rbfs['expanded']} exp, {diag_rbfs['elapsed']:.2f}s, {diag_rbfs['peak_mem_mb']:.2f}MB)")
        
        # Métodos aproximados: 10 semillas (0..9) con 6,000 evals
        algorithms = ['HC_single', 'HC_restarts', 'SA', 'GA']
        for algo in algorithms:
            print(f"  Ejecutando {algo} (10 semillas)...", end="", flush=True)
            for seed in range(10):
                rng_init = np.random.RandomState(seed)
                init_tour = [0] + rng_init.permutation(range(1, n)).tolist()
                rng_run = np.random.RandomState(seed + 100)
                
                if algo == 'HC_single':
                    res = hill_climbing(init_tour, inst.D, max_evaluations=max_evals, time_limit=5.0)
                elif algo == 'HC_restarts':
                    res = hill_climbing_restarts(n, inst.D, max_evaluations=max_evals, time_limit=5.0, rng=rng_run)
                elif algo == 'SA':
                    res = simulated_annealing(init_tour, inst.D, max_evaluations=max_evals, time_limit=5.0, rng=rng_run)
                elif algo == 'GA':
                    res = genetic_algorithm(n, inst.D, max_evaluations=max_evals, time_limit=5.0, pop_size=40, rng=rng_run)
                    
                gap = 100.0 * (res['cost'] - opt) / opt
                scale_b_runs.append({
                    'instance': inst_name,
                    'dimension': n,
                    'algorithm': algo,
                    'seed': seed,
                    'cost': res['cost'],
                    'official_opt': opt,
                    'gap_percent': gap,
                    'elapsed_sec': res['elapsed'],
                    'peak_mem_mb': res['peak_mem_mb'],
                    'evaluations': res['evaluations'],
                    'stop_reason': res['status'],
                    'tour': res['tour'],
                    'history': res.get('history', []),
                    'detailed_trace': res.get('detailed_trace', []),
                    'diversity_trace': res.get('diversity_trace', [])
                })
            print(" Listo.")
                
    return pd.DataFrame(scale_b_runs), pd.DataFrame(diagnostic_runs)

# ==============================================================================
# 7. GENERACIÓN DE FIGURAS
# ==============================================================================

def generate_visualizations(instances, scale_a_df, scale_b_df, diag_df, output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}
    print("\n--- GENERANDO VISUALIZACIONES OBLIGATORIAS ---")
    
    # Fig 1: Rutas Inicial vs Final
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (inst_name, inst) in zip(axes.flatten(), instances.items()):
        sub = scale_b_df[scale_b_df['instance'] == inst_name]
        best_row = sub.loc[sub['cost'].idxmin()]
        best_tour = best_row['tour']
        n = inst.dimension
        rng_dummy = np.random.RandomState(42)
        init_tour = [0] + rng_dummy.permutation(range(1, n)).tolist()
        
        if inst.coords is not None:
            c = inst.coords
            init_c = c[init_tour + [init_tour[0]]]
            ax.plot(init_c[:, 0], init_c[:, 1], 'o--', color='lightgray', alpha=0.7, label=f'Inicial (Costo: {tour_cost(init_tour, inst.D)})')
            best_c = c[best_tour + [best_tour[0]]]
            ax.plot(best_c[:, 0], best_c[:, 1], 'o-', color='#1f77b4', linewidth=2, label=f"Mejor: {best_row['algorithm']} (Costo: {best_row['cost']}, Gap: {best_row['gap_percent']:.1f}%)")
            ax.scatter(c[0, 0], c[0, 1], color='red', s=100, zorder=5, label='Origen s')
            ax.set_title(f"{inst_name} ({inst.edge_weight_type}) - Recorrido 2D", fontsize=12, fontweight='bold')
        else:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            circ_coords = np.column_stack([np.cos(angles), np.sin(angles)])
            best_c = circ_coords[best_tour + [best_tour[0]]]
            ax.plot(best_c[:, 0], best_c[:, 1], 'o-', color='#2ca02c', linewidth=1.5,
                    label=f"Mejor: {best_row['algorithm']} (Costo: {best_row['cost']}, Gap: {best_row['gap_percent']:.1f}%)")
            ax.scatter(circ_coords[0, 0], circ_coords[0, 1], color='red', s=100, zorder=5, label='Origen s')
            ax.set_title(f"{inst_name} (EXPLICIT) - Disposición Esquemática Circular", fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    f1 = os.path.join(output_dir, 'fig1_rutas_inicial_final.png')
    plt.savefig(f1, dpi=300)
    plt.close()
    generated_files['fig1'] = f1
    print(f"  > Guardado: {f1}")
    
    # Fig 2: Curvas de Convergencia
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    algo_colors = {'HC_single': '#1f77b4', 'HC_restarts': '#ff7f0e', 'SA': '#2ca02c', 'GA': '#d62728'}
    for ax, (inst_name, inst) in zip(axes.flatten(), instances.items()):
        sub_inst = scale_b_df[scale_b_df['instance'] == inst_name]
        for algo, col in algo_colors.items():
            sub_algo = sub_inst[sub_inst['algorithm'] == algo]
            eval_grid = np.linspace(0, 6000, 80)
            curves = []
            for _, row in sub_algo.iterrows():
                h = row['history']
                if not h:
                    continue
                evs = [pt[0] for pt in h]
                bests = [pt[2] for pt in h]
                curves.append(np.interp(eval_grid, evs, bests))
            if curves:
                curves = np.array(curves)
                med = np.median(curves, axis=0)
                q25 = np.percentile(curves, 25, axis=0)
                q75 = np.percentile(curves, 75, axis=0)
                ax.plot(eval_grid, med, label=algo, color=col, linewidth=2)
                ax.fill_between(eval_grid, q25, q75, color=col, alpha=0.15)
        opt = sub_inst['official_opt'].iloc[0]
        ax.axhline(opt, color='black', linestyle='--', linewidth=1.5, label=f'Óptimo Oficial ({opt})')
        ax.set_title(f"Convergencia en {inst_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Evaluaciones de Candidatos", fontsize=10)
        ax.set_ylabel("Mejor Costo Observado", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    f2 = os.path.join(output_dir, 'fig2_convergencia_evaluaciones.png')
    plt.savefig(f2, dpi=300)
    plt.close()
    generated_files['fig2'] = f2
    print(f"  > Guardado: {f2}")

    # Fig 3: Boxplots de Gap (%)
    fig, ax = plt.subplots(figsize=(12, 6))
    inst_order = list(instances.keys())
    algos = ['HC_single', 'HC_restarts', 'SA', 'GA']
    data_to_plot = []
    labels = []
    for inst_name in inst_order:
        for algo in algos:
            gaps = scale_b_df[(scale_b_df['instance'] == inst_name) & (scale_b_df['algorithm'] == algo)]['gap_percent'].values
            data_to_plot.append(gaps)
            labels.append(f"{inst_name}\n{algo}")
    bplot = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=labels)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] * len(inst_order)
    for patch, col in zip(bplot['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_title("Distribución de Brecha porcentual (Gap %) por Algoritmo (10 semillas)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Gap (%) = 100 * (C - C*) / C*", fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    f3 = os.path.join(output_dir, 'fig3_gap_boxplots.png')
    plt.savefig(f3, dpi=300)
    plt.close()
    generated_files['fig3'] = f3
    print(f"  > Guardado: {f3}")

    # Fig 4: Escalamiento Exactos vs k
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(14, 5))
    for inst_name in instances.keys():
        sub = scale_a_df[scale_a_df['instance'] == inst_name]
        ax_time.plot(sub['k'], sub['astar_time'], 'o-', label=f"A* {inst_name}", linewidth=1.5)
        ax_time.plot(sub['k'], sub['rbfs_time'], 's--', label=f"RBFS {inst_name}", linewidth=1.5)
        ax_mem.plot(sub['k'], sub['astar_mem'], 'o-', label=f"A* {inst_name}", linewidth=1.5)
        ax_mem.plot(sub['k'], sub['rbfs_mem'], 's--', label=f"RBFS {inst_name}", linewidth=1.5)
    ax_time.set_title("Tiempo de CPU vs Tamaño k", fontsize=12, fontweight='bold')
    ax_time.set_xlabel("k ciudades", fontsize=10)
    ax_time.set_ylabel("Tiempo (s)", fontsize=10)
    ax_time.set_yscale('log')
    ax_time.grid(True, linestyle=':', alpha=0.6)
    ax_time.legend(fontsize=8)
    ax_mem.set_title("Memoria Pico Incremental vs k", fontsize=12, fontweight='bold')
    ax_mem.set_xlabel("k ciudades", fontsize=10)
    ax_mem.set_ylabel("Memoria Pico (MB)", fontsize=10)
    ax_mem.set_yscale('log')
    ax_mem.grid(True, linestyle=':', alpha=0.6)
    ax_mem.legend(fontsize=8)
    plt.tight_layout()
    f4 = os.path.join(output_dir, 'fig4_escalamiento_exactos.png')
    plt.savefig(f4, dpi=300)
    plt.close()
    generated_files['fig4'] = f4
    print(f"  > Guardado: {f4}")

    # Fig 5: Diagnóstico SA en berlin52
    sa_run = scale_b_df[(scale_b_df['algorithm'] == 'SA') & (scale_b_df['instance'] == 'berlin52')].iloc[0]
    dt = sa_run['detailed_trace']
    evals_sa = [pt[0] for pt in dt]
    temps_sa = [pt[1] for pt in dt]
    cur_costs = [pt[2] for pt in dt]
    bst_costs = [pt[3] for pt in dt]
    window = 150
    acc_worse_rates = []
    worse_evals = []
    acc_window = []
    for pt in dt:
        if pt[5]:
            acc_window.append(1 if pt[4] else 0)
            if len(acc_window) > window:
                acc_window.pop(0)
            acc_worse_rates.append(sum(acc_window) / len(acc_window))
            worse_evals.append(pt[0])
    fig, (ax_cost, ax_temp, ax_acc) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax_cost.plot(evals_sa, cur_costs, color='#9ecae1', label='Costo Actual', alpha=0.7)
    ax_cost.plot(evals_sa, bst_costs, color='#08519c', label='Mejor Costo Observado', linewidth=2)
    ax_cost.axhline(7542, color='red', linestyle='--', label='Óptimo Oficial (7542)')
    ax_cost.set_ylabel("Costo del Ciclo", fontsize=10)
    ax_cost.set_title("Diagnóstico de Simulated Annealing (berlin52, semilla 0)", fontsize=12, fontweight='bold')
    ax_cost.legend(fontsize=9)
    ax_cost.grid(True, linestyle=':', alpha=0.6)
    ax_temp.plot(evals_sa, temps_sa, color='#e6550d', label='Temperatura T', linewidth=2)
    ax_temp.set_ylabel("Temperatura", fontsize=10)
    ax_temp.set_yscale('log')
    ax_temp.legend(fontsize=9)
    ax_temp.grid(True, linestyle=':', alpha=0.6)
    ax_acc.plot(worse_evals, acc_worse_rates, color='#31a354', label=f'Tasa Aceptación Empeoramientos (Ventana {window})', linewidth=2)
    ax_acc.set_ylabel("Tasa Aceptación", fontsize=10)
    ax_acc.set_xlabel("Evaluaciones de Candidatos", fontsize=10)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    f5 = os.path.join(output_dir, 'fig5_diagnostico_sa.png')
    plt.savefig(f5, dpi=300)
    plt.close()
    generated_files['fig5'] = f5
    print(f"  > Guardado: {f5}")

    # Fig 6: Diagnóstico GA en berlin52
    ga_run = scale_b_df[(scale_b_df['algorithm'] == 'GA') & (scale_b_df['instance'] == 'berlin52')].iloc[0]
    div_trace = ga_run['diversity_trace']
    gens = [pt[0] for pt in div_trace]
    ga_bests = [pt[1] for pt in div_trace]
    ga_meds = [pt[2] for pt in div_trace]
    ga_divs = [pt[3] * 100.0 for pt in div_trace]
    fig, (ax_ga_cost, ax_ga_div) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_ga_cost.plot(gens, ga_meds, color='#fb6a4a', label='Costo Mediano de Población', linewidth=2)
    ax_ga_cost.plot(gens, ga_bests, color='#a50f15', label='Mejor Costo de Población', linewidth=2)
    ax_ga_cost.axhline(7542, color='black', linestyle='--', label='Óptimo Oficial (7542)')
    ax_ga_cost.set_ylabel("Costo del Ciclo", fontsize=10)
    ax_ga_cost.set_title("Diagnóstico del Algoritmo Genético (berlin52, semilla 0)", fontsize=12, fontweight='bold')
    ax_ga_cost.legend(fontsize=9)
    ax_ga_cost.grid(True, linestyle=':', alpha=0.6)
    ax_ga_div.plot(gens, ga_divs, color='#756bb1', label='Diversidad Poblacional (% Individuos Únicos)', linewidth=2)
    ax_ga_div.set_ylabel("Diversidad (%)", fontsize=10)
    ax_ga_div.set_xlabel("Generación", fontsize=10)
    ax_ga_div.set_ylim(-5, 105)
    ax_ga_div.legend(fontsize=9)
    ax_ga_div.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    f6 = os.path.join(output_dir, 'fig6_diagnostico_ga.png')
    plt.savefig(f6, dpi=300)
    plt.close()
    generated_files['fig6'] = f6
    print(f"  > Guardado: {f6}")

    # Fig 7: Secuencia 2-Opt en berlin52
    inst = instances['berlin52']
    c = inst.coords
    n = inst.dimension
    rng_demo = np.random.RandomState(42)
    cur_t = [0] + rng_demo.permutation(range(1, n)).tolist()
    pairs = get_valid_2opt_pairs(n)
    frames = [(0, cur_t.copy(), tour_cost(cur_t, inst.D))]
    eval_c = 0
    while len(frames) < 6:
        best_d = 0
        best_p = None
        for i, j in pairs:
            eval_c += 1
            d = two_opt_delta(cur_t, i, j, inst.D)
            if d < best_d:
                best_d = d
                best_p = (i, j)
        if best_p is None:
            break
        cur_t = two_opt_move(cur_t, best_p[0], best_p[1])
        frames.append((eval_c, cur_t.copy(), tour_cost(cur_t, inst.D)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for idx, (ev, r, cost) in enumerate(frames[:6]):
        ax = axes.flatten()[idx]
        route_c = c[r + [r[0]]]
        ax.plot(route_c[:, 0], route_c[:, 1], 'o-', color='#3182bd', markersize=4, linewidth=1.5)
        ax.scatter(c[0, 0], c[0, 1], color='red', s=60, zorder=5)
        ax.set_title(f"Paso {idx}: Costo={cost}\n(Evals={ev})", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle("Secuencia de Mejora 2-Opt en berlin52 (Eliminación progresiva de cruces)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    f7 = os.path.join(output_dir, 'fig7_animacion_2opt_berlin52.png')
    plt.savefig(f7, dpi=300)
    plt.close()
    generated_files['fig7'] = f7
    print(f"  > Guardado: {f7}")
    
    return generated_files

if __name__ == '__main__':
    t_start_all = time.time()
    print("Iniciando ejecución experimental optimizada...")
    data_files = {
        'att48': 'data/att48.tsp',
        'berlin52': 'data/berlin52.tsp',
        'brazil58': 'data/brazil58.tsp',
        'gr48': 'data/gr48.tsp'
    }
    official_opts = {
        'att48': 10628,
        'berlin52': 7542,
        'brazil58': 25395,
        'gr48': 5046
    }
    instances = {name: TSPInstance(path) for name, path in data_files.items()}
    
    if os.path.exists('scale_a_results.csv'):
        print("Cargando scale_a_results.csv previamente generado...")
        scale_a_df = pd.read_csv('scale_a_results.csv')
    else:
        scale_a_df = run_scale_a(instances, sizes=(6, 8, 10, 12))
        scale_a_df.to_csv('scale_a_results.csv', index=False)
    
    scale_b_df, diag_df = run_scale_b(instances, official_opts, max_evals=6000)
    cols_to_save = [c for c in scale_b_df.columns if c not in ('tour', 'history', 'detailed_trace', 'diversity_trace')]
    scale_b_df[cols_to_save].to_csv('scale_b_results.csv', index=False)
    diag_df.to_csv('scale_b_diagnostics.csv', index=False)
    
    gen_figs = generate_visualizations(instances, scale_a_df, scale_b_df, diag_df)
    print(f"\n¡EJECUCIÓN TOTAL COMPLETADA EN {time.time() - t_start_all:.1f} SEGUNDOS!")
