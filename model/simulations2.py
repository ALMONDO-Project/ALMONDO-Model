import os
import future.utils
import argparse
import logging
import json
import networkx as nx
from ndlib.models.DiffusionModel import DiffusionModel
import ndlib.models.ModelConfig as mc
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import time
import pickle
from almondoModel import AlmondoModel  # Ensure this import path is correct
from lobbyistAgent import LobbyistAgent  # Ensure lobbyistAgent.py is in the same directory
from plottingutils import *

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AlmondoModel Simulation")

    # Configuration file argument
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file (optional)')

    # Graph parameters
    parser.add_argument('--graph_type', type=str, default='erdos_renyi',
                        help="Type of graph to generate. Supported types: 'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'custom_edgelist', 'custom_adjacency'.")
    parser.add_argument('--edgelist_path', type=str, default=None, help='Path to edgelist file (required if graph_type is "custom_edgelist").')
    parser.add_argument('--adjacency_matrix', type=str, default=None, help='Path to adjacency matrix file or provide as a NumPy array (required if graph_type is "custom_adjacency").')
    # Specific graph parameters
    parser.add_argument('--N', type=int, help='Number of nodes in the graph (default: 1000)')
    parser.add_argument('--p', type=float, help='Probability of edge creation (default: 0.01) (erdos_renyi)')
    parser.add_argument('--m', type=int, help='Number of edges to attach from a new node to existing nodes (Barabási-Albert).')
    parser.add_argument('--k', type=int, help='Each node is joined with its k nearest neighbors in a ring topology (Watts-Strogatz).')
    parser.add_argument('--p_ws', type=float, help='The probability of rewiring each edge (Watts-Strogatz).')

    # Define simulation parameters with default values
    parser.add_argument('--p_o', type=float, help='Probability of optimistic event model (default: 0.01)')
    parser.add_argument('--p_p', type=float, help='Probability of pessimistic event model (default: 0.99)')
    parser.add_argument('--l', type=float, help='Underreaction parameter (default: 0.8)')
    parser.add_argument('--n_lobbyists', type=int, help='Number of lobbyists (default: 5)')
    parser.add_argument('--max_iterations', type=int, help='Maximum number of iterations (default: 10000)')
    parser.add_argument('--save_dir', type=str, help='Directory to save logs and results (default: ./results)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility (default: None)')

    # Add any lobbyist-specific parameters if needed
    # For example, signalmatrix paths or influence strengths can be added here

    args = parser.parse_args()

    # Initialize parameters dictionary with defaults
    params = {
        'p_o': 0.01,
        'p_p': 0.99,
        'l': 0.8,
        'N': 1000,
        'n_lobbyists': 2,
        'T': 100,
        'save_dir': 'results',
        'seed': None,
        'graph_type': 'erdos_renyi',
        'p': 0.01,
        'm': 5,
        'k': 10,
        'p_ws': 0.1,
        'edgelist_path': None,
        'adjacency_matrix': None
    }

    # If a config file is provided, load parameters from it
    if args.config:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Configuration file '{args.config}' does not exist.")
        with open(args.config, 'r') as f:
            config_params = json.load(f)
        # Update the default params with those from the config file
        params.update(config_params)

    # Override with command-line arguments if provided
    cli_args = vars(args)
    for key in params.keys():
        if cli_args.get(key) is not None:
            params[key] = cli_args[key]

    return params

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
        if key not in ['save_dir', 'seed', 'graph_type', 'edgelist_path', 'adjacency_matrix']
    ]
    folder_name = '_'.join(folder_elements)
    save_dir = os.path.join(base_dir, folder_name)

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def setup_logging(save_dir):
    """
    Set up logging configuration.

    Args:
        save_dir (str): Directory where the log file will be saved.

    Returns:
        Logger object.
    """
    log_file = os.path.join(save_dir, 'simulation.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    return logger

def validate_parameters(params):
    """
    Validate simulation parameters to ensure they are within expected ranges.

    Args:
        params (dict): Dictionary of simulation parameters.

    Raises:
        AssertionError: If any parameter is out of its valid range.
    """
    assert 0 <= params['p_o'] <= 1, "p_o must be between 0 and 1."
    assert 0 <= params['p_p'] <= 1, "p_p must be between 0 and 1."
    assert 0 <= params['l'] <= 1, "l must be between 0 and 1."
    assert params['N'] > 0, "N must be a positive integer."
    assert 0 <= params['p'] <= 1, "p_er must be between 0 and 1."
    assert params['n_lobbyists'] >= 0, "n_lobbyists must be non-negative."
    assert params['T'] > 0, "max_iterations must be a positive integer."
    # Additional validations for graph types
    if params['graph_type'] == 'barabasi_albert':
        assert 'm' in params and params['m'] > 0, "m must be a positive integer for Barabási-Albert graph."
    if params['graph_type'] == 'watts_strogatz':
        assert 'k' in params and params['k'] > 0, "k must be a positive integer for Watts-Strogatz graph."
        assert 'p_ws' in params and 0 <= params['p_ws'] <= 1, "p_ws must be between 0 and 1 for Watts-Strogatz graph."
    if params['graph_type'] == 'custom_edgelist':
        assert params['edgelist_path'] is not None, "edgelist_path must be provided for custom_edgelist graph type."
    if params['graph_type'] == 'custom_adjacency':
        assert params['adjacency_matrix'] is not None, "adjacency_matrix must be provided for custom_adjacency graph type."
        
import numpy as np
import os

def generate_signalmatrix(T, N, budget=None, p=0.5):
    if budget is None:
        budget = T*N
    signal_matrix = np.zeros((T, N), dtype=int)
    total_signals = min(budget, T * N)
    # Randomly choose indices to set to 1
    indices = np.random.choice(T * N, size=total_signals, replace=False)
    signal_matrix.flat[indices] = 1
    return signal_matrix    

def read_signalmatrix(filename):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Signal matrix file '{filepath}' does not exist.")
    _, ext = os.path.splitext(filepath)
    if ext == '.npy':
        signal_matrix = np.load(filepath)
    elif ext == '.csv':
        signal_matrix = np.loadtxt(filepath, delimiter=',').astype(int)
    else:
        raise ValueError(f"Unsupported signal matrix file format: {ext}. Supported formats are .npy and .csv.")
    if signal_matrix.shape != (T, N):
        raise ValueError(f"Signal matrix shape {signalmatrix.shape} does not match expected shape ({T}, {N}).")
    return signal_matrix

def pass_signalmatrix(matrix):
    signal_matrix = matrix
    return signal_matrix

def create_signal_matrix(T, N, budget=None, p=0.5, filename=None, matrix = None):
    if filename is not None:
        signal_matrix = read_signalmatrix(filename)
    elif matrix is not None:
        signal_matrix = matrix
    else:
        signal_matrix = generate_signalmatrix(T, N, budget, p)
    return signal_matrix

def save_output(save_dir, iterations, logger):
    output_file = os.path.join(save_dir, 'results.pickle')
    logger.info(f"Saving results to {output_file}.")
    with open(output_file, 'wb') as ofile:
        pickle.dump(iterations, ofile)
    
    # Save results
    output_file = os.path.join(save_dir, 'senders_signals.csv')
    logger.info(f"Saving results to {output_file}.")
    # Convert system_status to DataFrame
    df = pd.DataFrame([{
        'iteration': entry['iteration'],
        'sender': entry['sender'],
        'signal': entry['signal']
    } for entry in iterations])
    df.to_csv(output_file, index=False)
    
    # Optionally, save node statuses over time
    output_file = os.path.join(save_dir, 'node_status.npy')
    np.save(output_file, [entry['status'] for entry in iterations])
    logger.info(f"Node statuses saved to {output_file}.")

def main():
    # Parse command-line arguments and load parameters
    params = parse_arguments()
    
    # Validate parameters
    try:
        validate_parameters(params)
    except AssertionError as e:
        print(f"Parameter validation error: {e}")
        exit(1)
    
    # Set random seed for reproducibility
    if params['seed'] is not None:
        random.seed(params['seed'])
        np.random.seed(params['seed'])

    # Create save directory based on parameters
    save_dir = create_save_directory(params, base_dir=params['save_dir'])
    
    # Set up logging
    logger = setup_logging(save_dir)
    logger.info("Starting AlmondoModel Simulation with parameters:")
    logger.info(json.dumps(params, indent=4))
    
    # Generate the graph
    logger.info(f"Generating graph of type '{params['graph_type']}'.")
    try:
        G = generate_graph(**params)
        logger.info(f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    except ValueError as ve:
        logger.error(f"Graph generation failed: {ve}")
        exit(1)
      
    # Initialize lobbyists
    logger.info(f"Adding {params['n_lobbyists']} lobbyists.")
    # Optionally, you can load signal matrices from files or define them here
    # For simplicity, we'll assume no pre-defined signal matrices
    lobbyists = []
    for i in range(params['n_lobbyists']):
        model = input(f'Is the lobbyist {id} optimist or pessimist?')
        T = params['T']
        N = params['N']
        signals = create_signal_matrix(T=T, N=N, p=0.5)
        lobbyists.append(LobbyistAgent(model=model, graph=G, signalmatrix = signals, seed=4))
    
    # Initialize the model
    logger.info("Initializing AlmondoModel.")
    # Model selection
    model = AlmondoModel(graph=G, seed=4)

    # Model configuration
    config = mc.Configuration()
    config.add_model_parameter("p_o", params['p_o'])
    config.add_model_parameter("p_p", params['p_p'])
    config.add_model_parameter("l", params['l'])
    model.set_initial_status(configuration=config)
    logger.info("Initial node statuses set.")
    
    model.add_lobbyists(lobbyists)
    logger.info("Lobbyists added to the model.")
    
    # Run the simulation
    logger.info(f"Running simulation for max_iterations={params['T']}.")
    start_time = time.time()
    
    iterations = model.iteration_bunch(T=T)
    
    end_time = time.time()
    logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    
    save_output(save_dir, iterations, logger)
    logger.info(f"Output saved in {save_dir}")
    
    plot_output(params, iterations, save_dir, logger)
    
    logger.info(f"Process ended")

if __name__ == "__main__":
    main()
