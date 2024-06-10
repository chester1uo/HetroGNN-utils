import os
import re
import json
import dgl
import numpy as np
import torch as th
import logging
import pandas as pd
from multiprocessing import Pool, cpu_count

def get_logger(name):
    """
    Configures and returns a logger object.
    
    Parameters:
    - name (str): Name for the logger.
    
    Returns:
    - logger (Logger): Configured logger object.
    """
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger

logging = get_logger(__name__)

def get_features(id_to_node, node_features):
    """
    Reads node features from a file and returns the feature matrix and new nodes.
    
    Parameters:
    - id_to_node (dict): Dictionary mapping node names (id) to dgl node indices.
    - node_features (str): Path to the file containing node features.
    
    Returns:
    - features (np.ndarray): Node feature matrix in order.
    - new_nodes (list): List of new nodes not yet in the graph.
    """
    indices, features, new_nodes = [], [], []
    max_node = max(id_to_node.values())

    with open(node_features, "r") as fh:
        for line in fh:
            node_feats = line.strip().split(",")
            node_id = node_feats[0]
            feats = np.array(list(map(float, node_feats[1:])))
            features.append(feats)
            if node_id not in id_to_node:
                max_node += 1
                id_to_node[node_id] = max_node
                new_nodes.append(max_node)

            indices.append(id_to_node[node_id])

    features = np.array(features).astype('float32')
    features = features[np.argsort(indices), :]
    return features, new_nodes

def _get_node_idx(args):
    """
    Helper function to get or create a node index.
    
    Parameters:
    - args (tuple): Tuple containing id_to_node, node_type, node_id, ptr.
    
    Returns:
    - node_idx (int): Index of the node.
    - id_to_node (dict): Updated dictionary mapping node names (id) to dgl node indices.
    - ptr (int): Updated pointer for the next node index.
    """
    id_to_node, node_type, node_id, ptr = args
    if node_type in id_to_node:
        if node_id in id_to_node[node_type]:
            node_idx = id_to_node[node_type][node_id]
        else:
            id_to_node[node_type][node_id] = ptr
            node_idx = ptr
            ptr += 1
    else:
        id_to_node[node_type] = {}
        id_to_node[node_type][node_id] = ptr
        node_idx = ptr
        ptr += 1

    return node_idx, id_to_node, ptr

def parse_edgelist(args):
    """
    Parses an edgelist file and returns edge lists and updated node dictionary.
    
    Parameters:
    - args (tuple): Tuple containing edges, id_to_node, header, source_type, sink_type.
    
    Returns:
    - edge_list (list): List of edges as tuples.
    - rev_edge_list (list): List of reverse edges as tuples.
    - id_to_node (dict): Updated dictionary mapping node names (id) to dgl node indices.
    - source_type (str): Type of the source node in the edge.
    - sink_type (str): Type of the sink node in the edge.
    """
    edges, id_to_node, header, source_type, sink_type = args
    edge_list = []
    rev_edge_list = []
    source_pointer, sink_pointer = 0, 0
    with open(edges, "r") as fh:
        for i, line in enumerate(fh):
            source, sink = line.strip().split(",")
            if i == 0:
                if header:
                    source_type, sink_type = source, sink
                if source_type in id_to_node:
                    source_pointer = max(id_to_node[source_type].values()) + 1
                if sink_type in id_to_node:
                    sink_pointer = max(id_to_node[sink_type].values()) + 1
                continue

            source_node, id_to_node, source_pointer = _get_node_idx((id_to_node, source_type, source, source_pointer))
            if source_type == sink_type:
                sink_node, id_to_node, source_pointer = _get_node_idx((id_to_node, sink_type, sink, source_pointer))
            else:
                sink_node, id_to_node, sink_pointer = _get_node_idx((id_to_node, sink_type, sink, sink_pointer))

            edge_list.append((source_node, sink_node))
            rev_edge_list.append((sink_node, source_node))

    return edge_list, rev_edge_list, id_to_node, source_type, sink_type

def get_edgelists(edgelist_expression, directory):
    """
    Gets a list of edge list files matching a given expression in the specified directory.
    
    Parameters:
    - edgelist_expression (str): Regular expression to match edgelist files.
    - directory (str): Directory to search for edgelist files.
    
    Returns:
    - list: List of matched edgelist filenames.
    """
    if "," in edgelist_expression:
        return edgelist_expression.split(",")
    files = os.listdir(directory)
    compiled_expression = re.compile(edgelist_expression)
    return [filename for filename in files if compiled_expression.match(filename)]

def construct_graph(output_dir, edges, nodes, target_node_type, num_workers=4):
    """
    Constructs a graph from the given edge lists and node features.
    
    Parameters:
    - output_dir (str): Directory containing the edge list and node feature files.
    - edges (list): List of edge list filenames.
    - nodes (str): Path to the file containing node features.
    - target_node_type (str): Type of the target node.
    - num_workers (int): Number of worker processes for parallel processing.
    
    Returns:
    - g (DGLGraph): Constructed heterogeneous graph.
    - features (np.ndarray): Node feature matrix.
    - target_id_to_node (dict): Dictionary mapping target node names (id) to dgl node indices.
    - id_to_node (dict): Updated dictionary mapping node names (id) to dgl node indices.
    """
    logging.info("Getting relation graphs from the following edge lists: %s", edges)
    edgelists, id_to_node = {}, {}

    # Use multiprocessing to parse edgelist files
    pool = Pool(num_workers)
    parse_args = [(os.path.join(output_dir, edge), id_to_node, True, 'user', 'user') for edge in edges]
    results = pool.map(parse_edgelist, parse_args)

    for edge, (edge_list, rev_edge_list, id_to_node, src, dst) in zip(edges, results):
        if src == target_node_type:
            src = 'target'
        if dst == target_node_type:
            dst = 'target'

        if src == 'target' and dst == 'target':
            logging.info("Will add self loop for target later.")
        else:
            edgelists[(src, src + '<>' + dst, dst)] = edge_list
            edgelists[(dst, dst + '<>' + src, src)] = rev_edge_list
            logging.info("Read edges for %s from edgelist: %s", src + '<' + dst + '>', os.path.join(output_dir, edge))

    # Get features for target nodes
    features, new_nodes = get_features(id_to_node[target_node_type], os.path.join(output_dir, "feature.csv"))
    logging.info("Read in features for target nodes")

    # Add self relation
    edgelists[('target', 'self_relation', 'target')] = [(t, t) for t in id_to_node[target_node_type].values()]

    # Construct the heterogeneous graph
    g = dgl.heterograph(edgelists)
    logging.info("Constructed heterograph with the following metagraph structure: Node types %s, Edge types %s", g.ntypes, g.canonical_etypes)
    logging.info("Number of nodes of type target: %d", g.number_of_nodes('target'))

    g.nodes['target'].data['features'] = th.from_numpy(features)

    target_id_to_node = id_to_node[target_node_type]
    id_to_node['target'] = target_id_to_node

    del id_to_node[target_node_type]

    return g, features, target_id_to_node, id_to_node

def save_edge_file(edge_file, df):
    """
    Saves the dataframe to a CSV file.
    
    Parameters:
    - edge_file (str): Path to the output edge file.
    - df (DataFrame): DataFrame containing the edge data.
    """
    df.to_csv(edge_file, index=False)

def construct_graph_main(config_path):
    """
    Main function to construct the graph from the configuration file.
    
    Parameters:
    - config_path (str): Path to the JSON configuration file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    input_data = config["input_data"]
    edgelist_expression = config["edgelist_expression"]
    output_dir = config["output_dir"]
    node_cols = config["node_cols"]
    key_col = config["key_col"]
    feature_cols = config["feature_cols"]
    num_workers = config["num_workers"]
    num_workers = min(num_workers, cpu_count())

    os.makedirs(output_dir, exist_ok=True)

    print("Generating Features...")
    # Load and save features
    data = pd.read_csv(input_data)
    submask_data = data[feature_cols]
    submask_data.to_csv(os.path.join(output_dir, "feature.csv"), index=False, header=False)

    print("Generating Nodes...")
    # Use multiprocessing to save edge files
    pool = Pool(num_workers)
    edge_files = [(os.path.join(output_dir, f"relation_{col}_edgelist.csv"), data[[key_col, col]]) for col in node_cols]
    pool.starmap(save_edge_file, edge_files)

    logging.info("Saved edge lists for all node columns.")

    # Get edge lists
    edges = get_edgelists(edgelist_expression, output_dir)
    print("Generating Graph...")
    # Construct the graph
    graph, features, target_id_to_node, id_to_node = construct_graph(output_dir, edges, input_data, key_col, num_workers)

    result = {
        'graph': graph,
        'features': features,
        'target_id_to_node': target_id_to_node,
        'id_to_node': id_to_node
    }

    # Save the graph data
    output_path = os.path.join(output_dir, 'graph_data.pth')
    th.save(result, output_path)
    logging.info("Saved graph data to %s", output_path)
    logging.info("Graph construction completed. Graph summary: %s", graph)


if __name__ == "__main__":
    construct_graph_main("config.json")
