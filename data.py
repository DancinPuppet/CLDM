import networkx as nx
import torch
import random
from torch.utils.data import Dataset
from args import Args
from torch_geometric.data import Data
import numpy as np



def nx_to_pyg(graph: nx.Graph, args: Args, return_node_mapping: bool = False):
    if len(graph.nodes()) == 0:
        data = Data(
            x=torch.empty((0, args.node_features_dim), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.empty((0,), dtype=torch.long),
        )
        return (data, {}) if return_node_mapping else data

    nodes = list(graph.nodes())
    seed = hash(getattr(graph, 'id', 0)) % (2**32)
    # local_random = random.Random(seed)
    local_random = random.Random()
    local_random.shuffle(nodes)
    node_mapping = {node: i for i, node in enumerate(nodes)}

    num_nodes = graph.number_of_nodes()#
    edge_index = []
    for src, dst, _attr in graph.edges(data=True):
        edge_index.append([node_mapping[src], node_mapping[dst]])
        edge_index.append([node_mapping[dst], node_mapping[src]])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    feature_names = ['degree_centrality', 'avg_neighbor_degree', 'clustering',
                     'core_number', 'pagerank']
    label_name = 'source'

    node_features = torch.zeros((num_nodes, args.node_features_dim), dtype=torch.float)
    node_labels   = torch.zeros(num_nodes, dtype=torch.long)

    for node, mapped_idx in node_mapping.items():
        nd = graph.nodes[node]
        for feat_idx, feature_name in enumerate(feature_names):
            if feature_name in nd:
                node_features[mapped_idx, feat_idx] = float(nd[feature_name])
            else:
                node_features[mapped_idx, feat_idx] = 0.0
        if label_name in nd:
            node_labels[mapped_idx] = int(nd[label_name])
        else:
            node_labels[mapped_idx] = 0

    data = Data(x=node_features, edge_index=edge_index, y=node_labels)
    data.graph_id = getattr(graph, 'id', None)

    return (data, node_mapping) if return_node_mapping else data


class GraphGroupDataset(Dataset):
    def __init__(self, graph_groups: list, args: Args):
        self.graph_groups = graph_groups
        self.args = args

    def __len__(self) -> int:
        return len(self.graph_groups)

    def __getitem__(self, idx: int) -> Data:
        return nx_to_pyg(self.graph_groups[idx], self.args)