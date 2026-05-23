import os
from pathlib import Path
import ast
import networkx as nx
import args
import random
import node_feature
import pickle
import ndlib.models.ModelConfig as mc
from scipy.io import mmread


def twitter25(args_para: args.Args) -> list:
    graphs = []
    root_dir = Path(args_para.data_path)
    for subfolder in root_dir.iterdir():
        if subfolder.is_dir():
            print(f'loading {subfolder.name}\n')
            graph_file_path = args_para.data_path + subfolder.name + '/' + subfolder.name + '_single_graph_pro.json'
            review_file_path = args_para.data_path + subfolder.name + '/' + subfolder.name + '_review.json'
            quote_file_path = args_para.data_path + subfolder.name + '/' + subfolder.name + '_quote.json'
            G_graph = nx.Graph()
            with open(graph_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_strip = line.strip()
                    left_str, right_str = line_strip.split("->")
                    try:
                        left_list = ast.literal_eval(left_str)
                        right_list = ast.literal_eval(right_str)
                    except Exception as e:
                        print(f"error：{line}\n{e}")
                        continue

                    left_type, left_user, left_event, left_time = left_list
                    right_type, right_user, right_event, right_time = right_list

                    left_user = str(left_user)
                    right_user = str(right_user)

                    if left_user not in G_graph:
                        if left_type == 0:
                            G_graph.add_node(left_user, source=1)
                        else:
                            G_graph.add_node(left_user, source=0)
                    if right_user not in G_graph:
                        G_graph.add_node(right_user, source=0)

                    G_graph.add_edge(left_user, right_user)

            G_new_graph = node_feature.compute_node_features(G_graph)
            G_new_graph.id = subfolder.name
            graphs.append(G_new_graph)

    max_num_nodes = 0
    for graph in graphs:
        if max_num_nodes < graph.number_of_nodes():
            max_num_nodes = graph.number_of_nodes()
    print(f'Max_nodes:{max_num_nodes}')
    args_para.max_num_node = max_num_nodes
    print('All connected, features computed !')
    return graphs


def twitter15(args_para: args.Args) -> list:
    graphs = []
    root_dir = Path(args_para.data_path)
    for file in root_dir.glob('*.txt'):
        G_graph = nx.Graph()
        print(f'processing {file}')
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line_strip = line.strip()
                left_str, right_str = line_strip.split("->")
                try:
                    left_list = ast.literal_eval(left_str)
                    right_list = ast.literal_eval(right_str)
                except Exception as e:
                    print(f"error：{line}\n{e}")
                    continue

                left_user, left_event, left_time = left_list
                right_user, right_event, right_time = right_list

                if 'ROOT' in left_user:
                    continue

                if left_user == right_user:
                    continue

                if left_user not in G_graph:
                    if left_time == '0.0':
                        G_graph.add_node(int(left_user), source=1)
                    else:
                        G_graph.add_node(int(left_user), source=0)
                if right_user not in G_graph:
                    G_graph.add_node(int(right_user), source=0)

                G_graph.add_edge(int(left_user), int(right_user))
            num_components = nx.number_connected_components(G_graph)
            assert num_components == 1, 'error,not connected'
        G_new_graph = node_feature.compute_node_features(G_graph)
        graphs.append(G_new_graph)

    max_num_nodes = 0
    for graph in graphs:
        if max_num_nodes < graph.number_of_nodes():
            max_num_nodes = graph.number_of_nodes()
    print(f'Max_nodes:{max_num_nodes}')
    args_para.max_num_node = max_num_nodes
    print('All connected, features computed !')
    return graphs


def twitter16(args_para: args.Args) -> list:
    graphs = []
    root_dir = Path(args_para.data_path)
    for file in root_dir.glob('*.txt'):
        G_graph = nx.Graph()
        print(f'processing {file}')
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line_strip = line.strip()
                left_str, right_str = line_strip.split("->")
                try:
                    left_list = ast.literal_eval(left_str)
                    right_list = ast.literal_eval(right_str)
                except Exception as e:
                    print(f"error：{line}\n{e}")
                    continue

                left_user, left_event, left_time = left_list
                right_user, right_event, right_time = right_list

                if 'ROOT' in left_user:
                    continue

                if left_user == right_user:
                    continue

                if left_user not in G_graph:
                    if left_time == '0.0':
                        G_graph.add_node(int(left_user), source=1)
                    else:
                        G_graph.add_node(int(left_user), source=0)
                if right_user not in G_graph:
                    G_graph.add_node(int(right_user), source=0)

                G_graph.add_edge(int(left_user), int(right_user))
            num_components = nx.number_connected_components(G_graph)
            assert num_components == 1, 'error,not connected'
        G_new_graph = node_feature.compute_node_features(G_graph)
        graphs.append(G_new_graph)

    max_num_nodes = 0
    for graph in graphs:
        if max_num_nodes < graph.number_of_nodes():
            max_num_nodes = graph.number_of_nodes()
    print(f'Max_nodes:{max_num_nodes}')
    args_para.max_num_node = max_num_nodes
    print('All connected, features computed !')
    return graphs

def weibo(args_para: args.Args) -> list:
    graphs = []
    graphs_path = args_para.data_path + 'weibo_graph.pkl'
    with open(graphs_path, 'rb') as f:
        datas = pickle.load(f)
    raw_graphs = datas.data.get('propagation', None)
    source_mark = datas.data.get('source', None)
    for idx, (graph, source) in enumerate(zip(raw_graphs, source_mark)):
        graph = graph.to_undirected()
        nodes = graph.nodes()
        source_features = {node: 1 if node == source else 0 for node in nodes}
        nx.set_node_attributes(graph, source_features, 'source')
        G_new_graph = node_feature.compute_node_features(graph)
        graphs.append(G_new_graph)
    return graphs


def data(args_para: args.Args, load_exited_graphs=False):
    graphs = []
    print(f"{args_para.graph_type} data loading......")
    if args_para.graph_type == 'twitter25':
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            graphs = twitter25(args_para)
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if args_para.graph_type == 'twitter15':
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            graphs = twitter15(args_para)
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if args_para.graph_type == 'twitter16':
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            graphs = twitter16(args_para)
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")

    if args_para.graph_type == 'weibo':
        graphs_path = args_para.graphs_saved_path
        if load_exited_graphs:
            if os.path.exists(graphs_path):
                print(f"Loading exited graphs path is {graphs_path}")
                with open(graphs_path, 'rb') as f:
                    graphs = pickle.load(f)
                max_num_nodes = 0
                for graph in graphs:
                    if max_num_nodes < graph.number_of_nodes():
                        max_num_nodes = graph.number_of_nodes()
                args_para.max_num_node = max_num_nodes
            else:
                print(f"Can't find exited graphs path :{graphs_path}")
        else:
            graphs = weibo(args_para)
            with open(graphs_path, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"Finish creating graphs in {graphs_path}")
    for i, graph in enumerate(graphs):
        graph.id = i
    return graphs
