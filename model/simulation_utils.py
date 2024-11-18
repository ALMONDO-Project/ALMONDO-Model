import os
import numpy as np
import networkx as nx

def generate_graph(graph_type='erdos_renyi', **kwargs):
    """
    Generate or load a graph based on the specified type.

    Args:
        graph_type (str): Type of graph to generate. Supported types:
                          - 'erdos_renyi': Erdős-Rényi graph
                          - 'barabasi_albert': Barabási-Albert graph
                          - 'watts_strogatz': Watts-Strogatz graph
                          - 'custom_edgelist': Load graph from an edgelist file
                          - 'custom_adjacency': Load graph from an adjacency matrix
        **kwargs: Additional keyword arguments specific to the graph type.

    Returns:
        networkx.Graph: Generated or loaded graph.
    """
    if graph_type == 'erdos_renyi':
        N = kwargs.get('N', 1000)
        p_er = kwargs.get('p_er', 0.01)
        seed = kwargs.get('seed', None)
        G = nx.erdos_renyi_graph(N, p_er, seed=seed)
    
    elif graph_type == 'complete_graph':
        N = kwargs.get('N', 1000)
        G = nx.complete_graph(N)
    
    elif graph_type == 'barabasi_albert':
        N = kwargs.get('N', 1000)
        m = kwargs.get('m', 5)  # Number of edges to attach from a new node to existing nodes
        seed = kwargs.get('seed', None)
        G = nx.barabasi_albert_graph(N, m, seed=seed)
    
    elif graph_type == 'watts_strogatz':
        N = kwargs.get('N', 1000)
        k = kwargs.get('k', 10)  # Each node is joined with its k nearest neighbors in a ring topology
        p_ws = kwargs.get('p_ws', 0.1)  # The probability of rewiring each edge
        seed = kwargs.get('seed', None)
        G = nx.watts_strogatz_graph(N, k, p_ws, seed=seed)
    
    elif graph_type == 'custom_edgelist':
        edgelist_path = kwargs.get('edgelist_path', None)
        if edgelist_path is None or not os.path.isfile(edgelist_path):
            raise ValueError(f"Invalid or missing 'edgelist_path' for graph_type '{graph_type}'.")
        G = nx.read_edgelist(edgelist_path, nodetype=int)
    
    elif graph_type == 'custom_adjacency':
        adjacency_matrix = kwargs.get('adjacency_matrix', None)
        if adjacency_matrix is None:
            raise ValueError(f"Missing 'adjacency_matrix' for graph_type '{graph_type}'.")
        if isinstance(adjacency_matrix, str) and os.path.isfile(adjacency_matrix):
            # Assume it's a file path to a saved adjacency matrix (e.g., .npy, .csv)
            _, ext = os.path.splitext(adjacency_matrix)
            if ext == '.npy':
                adjacency_matrix = np.load(adjacency_matrix)
            elif ext == '.csv':
                adjacency_matrix = np.loadtxt(adjacency_matrix, delimiter=',')
            else:
                raise ValueError(f"Unsupported adjacency matrix file format: {ext}")
        elif isinstance(adjacency_matrix, np.ndarray):
            pass  # Already a NumPy array
        else:
            raise ValueError(f"'adjacency_matrix' must be a NumPy array or a valid file path.")
        G = nx.from_numpy_array(adjacency_matrix)
    else:
        raise ValueError(f"Unsupported graph_type '{graph_type}'. Supported types are: 'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'custom_edgelist', 'custom_adjacency'.")
    return G




################ Saving files #######################################################################################################################
def create_save_directory(params, base_dir='results'):
    """
    Create a directory named based on simulation parameters.

    Args:
        params (dict): Dictionary of simulation parameters.
        base_dir (str): Base directory to store results.

    Returns:
        str: Path to the created directory.
    """
    # Exclude 'save_dir', 'seed', 'graph_type', and custom graph paths from folder name to keep it concise
    folder_elements = [
        f"{key}{value}"
        for key, value in params.items()
        if key not in ['save_dir', 'seed', 'edgelist_path', 'adjacency_matrix']
    ]
    folder_name = '_'.join(folder_elements)
    save_dir = os.path.join(base_dir, folder_name)

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
