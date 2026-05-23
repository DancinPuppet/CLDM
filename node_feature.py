import os
import ast
import json
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

def _to_int(x):
    try:
        return int(float(x))
    except Exception:
        return 0

def compute_node_features(G: nx.Graph) -> nx.Graph:
    G_graph = G

    degree_centrality = nx.degree_centrality(G_graph)
    clustering = nx.clustering(G_graph)
    pagerank = nx.pagerank(G_graph)
    core_number = nx.core_number(G_graph)
    max_degree = max(dict(G_graph.degree()).values()) if G_graph.number_of_nodes() > 0 else 1
    max_core_number = max(core_number.values()) if core_number else 1
    pagerank_values = list(pagerank.values())
    max_pagerank = max(pagerank_values) if pagerank_values else 1
    for node in G_graph.nodes():
        node_features = {
            'degree_centrality': degree_centrality[node],
            'clustering': clustering[node],
            'pagerank': pagerank[node] / max_pagerank,
            'avg_neighbor_degree': (np.mean([G_graph.degree(n) for n in G_graph.neighbors(node)])) / max_degree if G_graph.degree(node) > 0 else 0,  # H2 平均邻居度
            'core_number':  core_number[node] / max_core_number if max_core_number > 0 else 0
        }
        G_graph.nodes[node]['degree_centrality'] = node_features['degree_centrality']
        G_graph.nodes[node]['avg_neighbor_degree'] = node_features['avg_neighbor_degree']
        G_graph.nodes[node]['clustering'] = node_features['clustering']
        G_graph.nodes[node]['pagerank'] = node_features['pagerank']
        G_graph.nodes[node]['core_number'] = node_features['core_number']

    return G_graph