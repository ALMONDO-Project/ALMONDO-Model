from almondo_model.classes.almondoModel import AlmondoModel # This will be imported from ndlib once the model is loaded there
import ndlib.models.ModelConfig as mc
from almondo_model.functions.utils import transform
import networkx as nx
import numpy as np
from tqdm import tqdm
import random
import json
import os

class ALMONDOSimulator(object):
    """
    A simulator for running agent-based models in a network with lobbyists and different strategies.

    Arguments:
    - N (int): The number of agents in the network (nodes).
    - initial_distribution (str): The type of initial distribution of agent states (e.g., random, uniform).
    - T (int): The maximum number of iterations for the simulation.
    - p_o (float): Parameter for the opinion dynamics model.
    - p_p (float): Parameter for the opinion dynamics model.
    - lambda_values (float | list): The lambda values used to configure agents' susceptibility to influence.
    - phi_values (float | list): The phi values used to configure agents' resistance to influence.
    - base (str, default='results/'): The base directory for storing results.
    - scenario (str, optional): The name of the scenario for organizing results.
    - nruns (int, default=100): The number of simulation runs.
    - n_lobbyists (int, default=0): The number of lobbyists in the simulation.
    - lobbyists_data (dict, default={}): A dictionary containing data for the lobbyists (e.g., strategies, parameters).
    - verbose (bool, default=True): Whether to print debug information during the simulation.
    """

    def __init__(
        self, 
        N: int, 
        initial_distribution: str,
        T: int,
        p_o: float,
        p_p: float,
        lambda_values: float | list, 
        phi_values: float | list,
        base: str = 'results/',
        scenario: str = None,
        nruns: int = 100,
        initial_status: list = None,
        n_lobbyists: int = 0,
        lobbyists_data: dict = {},
        verbose: bool = True
    ):
        """
        Initialize the simulator with necessary configurations, directories, and data.

        Arguments:
        - N: Number of agents in the network.
        - initial_distribution: Type of initial agent states.
        - T: Maximum number of iterations.
        - p_o: Parameter for opinion dynamics.
        - p_p: Parameter for opinion dynamics.
        - lambda_values: Lambda configuration for agent influence susceptibility.
        - phi_values: Phi configuration for agent resistance to influence.
        - base: Base directory to store simulation results.
        - scenario: Specific scenario name for organizing the results.
        - nruns: The number of runs for the Monte Carlo simulation.
        - n_lobbyists: The number of lobbyists to be added to the model.
        - lobbyists_data: Data related to lobbyists' strategies and characteristics.
        """
        
        self.verbose = verbose
        self.base_path = base
        self.scenario_path = os.path.join(base, scenario)
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.scenario_path, exist_ok=True)
        self.strategies_path = os.path.join(self.scenario_path, "strategies")
        os.makedirs(self.strategies_path, exist_ok=True)        
        
        self.N = N
        self.graph = nx.complete_graph(N)  # Create a complete graph of N nodes
        self.p_o = p_o
        self.p_p = p_p
        self.lambdas = lambda_values
        if self.verbose:
            print(f'explored lambda values are {lambda_values}')
        self.phis = phi_values
        if self.verbose:
            print(f'explored phi values are {phi_values}')
        self.T = T
        self.initial_distribution = initial_distribution
        self.initial_status = initial_status
        self.nruns = nruns
        self.lobbyists_data = lobbyists_data
        self.n_lobbyists = n_lobbyists
        
        if self.n_lobbyists > 0:
            self.create_strategies()  # Generate strategies for lobbyists if any exist

        if self.verbose:
            print('Simulator created')

    def _print(self, *args, **kwargs):
        """Custom print method that respects verbose setting"""
        if self.verbose:
            print(*args, **kwargs)

    def config_model(self, lambda_v: float | list, phi_v: float | list):
        """
        Configures the model with agent parameters, initial distribution, and graph settings.

        Arguments:
        - lambda_v: Lambda values for the agents (either a single value or a list for each agent).
        - phi_v: Phi values for the agents (either a single value or a list for each agent).
        """
        self._print('Creating configuration object')
        config = mc.Configuration()

        self._print('Assigning p_o and p_p parameters')
        config.add_model_parameter("p_o", self.p_o)
        config.add_model_parameter("p_p", self.p_p)
        self._print(f'p_o={self.p_o}, p_p={self.p_p}')

        # Configure lambda values for each agent
        if isinstance(lambda_v, list):
            for i in self.graph.nodes():
                config.add_node_configuration("lambda", i, lambda_v[i])
        elif isinstance(lambda_v, float):
            self._print('Assigning homogeneous lambda')
            for i in self.graph.nodes():
                config.add_node_configuration("lambda", i, lambda_v)
        else:
            raise ValueError("lambda_v must be a float or a list")

        # Configure phi values for each agent
        if isinstance(phi_v, list):
            for i in self.graph.nodes():
                config.add_node_configuration("phi", i, phi_v[i])
        elif isinstance(phi_v, float):
            self._print('Assigning homogeneous phi')
            for i in self.graph.nodes():
                config.add_node_configuration("phi", i, phi_v)
        else:
            raise ValueError("phi_v must be a float or a list")

        # Initialize the model with the graph and configuration
        self._print('Configuring model: assigning graph, parameters, and initial distribution of weights')
        self.model = AlmondoModel(self.graph)
        self.model.set_initial_status(config, kind=self.initial_distribution, status=self.initial_status)

        self._print('Assign strategies to lobbyists if any')
        if self.n_lobbyists > 0:
            for id in tqdm(self.lobbyists_data):
                data = self.lobbyists_data[id]
                B = data['B']
                m = data['m']
                matrix, name = self.read_random_strategy(B)
                # self._print(f'Assigning strategy {name} to lobbyist {id}')
                # Add lobbyist with strategy to the model
                self.model.add_lobbyist(m, matrix)
                self.lobbyists_data[id]['strategies'].append(name)
        
        self._print('Configuration ended')

    def single_run(self, lambda_v: float | list, phi_v: float | list, drop_ev: bool = False) -> tuple:
        """
        Run a single simulation with given lambda and phi values.

        Arguments:
        - lambda_v: Lambda value(s) for the agents.
        - phi_v: Phi value(s) for the agents.
        - drop_ev (optional, bool): keep in memory iterations dictionary (keep true if you want evolution plots in the end)

        Returns:
        - tuple: A tuple containing system status and final distribution data.
        """
                
        self.config_model(lambda_v, phi_v)

        # Execute the system until steady state is reached
        self.system_status = self.model.steady_state(max_iterations=self.T,drop_evolution = drop_ev)

        # Save system status to a file
        self.save_system_status(self.runpath)

        # Calculate the final weights and probabilities
        fws = [el for el in self.system_status[-1]['status'].values()]
        fps = transform(fws, self.p_o, self.p_p)

        fd = {
            'final_weights': fws,
            'final_probabilities': fps,
            'final_iterations': int(self.system_status[-1]['iteration'])
        }

        return self.system_status, fd

    def runs(self, lambda_v: float | list, phi_v: float | list, overwrite: bool = False, drop_ev: bool = False):
        """
        Perform multiple simulation runs (Monte Carlo simulations).

        Arguments:
        - lambda_v: Lambda values for the agents.
        - phi_v: Phi values for the agents.
        - overwrite: Whether to overwrite existing runs (default is False).
        - drop_evolution (optional, bool): keep in memory iterations dictionary (keep true if you want evolution plots in the end)
        """
        self._print('Starting Monte Carlo runs')

        runs_data = []

        # Clear previous lobbyist strategies for new runs
        for id in range(self.n_lobbyists):
            self.lobbyists_data[id]['strategies'] = []

        for run in range(self.nruns):
            self._print(f"Running simulation with lambda={lambda_v}, phi={phi_v}, nl={self.n_lobbyists}")
            self._print(f'Starting run {run+1}/{self.nruns}')
            self.runpath = os.path.join(self.config_path, f'{run}')

            if not overwrite:
                if os.path.exists(f'{self.runpath}/status.json'):
                    self._print(f'Run {run+1}/{self.nruns} already performed, moving to next')
                    continue
                else:
                    os.makedirs(self.runpath, exist_ok=True)
            else:
                os.makedirs(self.runpath, exist_ok=True)
            
            if run<4: #in any case keeps first 4 runs for evolution plots
                _, final_distributions = self.single_run(lambda_v, phi_v,drop_ev=False)
            else:
                _, final_distributions = self.single_run(lambda_v, phi_v, drop_ev=drop_ev)

            runs_data.append(final_distributions)

        self._print('Saving configuration to file')
        self.save_config()

        self._print('Saving final distributions to file')
        filename = os.path.join(self.config_path, 'runs_data.json')
        with open(filename, 'w') as f:
            json.dump(runs_data, f)

    def execute_experiments(self, overwrite_runs: bool = False, drop_evolution: bool = False):
        """
        Execute experiments for all lambda and phi configurations.

        Arguments:
        - overwrite_runs: Whether to overwrite previous runs (default is False).
        - drop_evolution (optional, bool): keep in memory iterations dictionary (keep true if you want evolution plots in the end)
        """
        self._print('Starting experiments')

        for _, (lambda_v, phi_v) in enumerate([(l, p) for l in self.lambdas for p in self.phis]):

            self._print(f'Starting configuration lambda={lambda_v}, phi={phi_v}')

            self.config_path = os.path.join(self.scenario_path, f'{lambda_v}_{phi_v}')
            os.makedirs(self.config_path, exist_ok=True)

            self.runs(lambda_v, phi_v, overwrite=overwrite_runs, drop_ev=drop_evolution)

    def save_config(self, filename: str = None):
        """
        Save the current simulation configuration to a file.

        Arguments:
        - filename: The filename to save the configuration. If None, it saves to a default path.
        """
        
        d = {
            'p_o': self.p_o,
            'p_p': self.p_p,
            'lambda_values': self.lambdas,
            'phi_values': self.phis,
            'T': self.T,
            'n_lobbyists': self.n_lobbyists,
            'lobbyists_data': self.lobbyists_data
        }

        d['nruns'] = self.nruns

        # Save graph as an edge list
        nx.write_edgelist(self.graph, os.path.join(self.scenario_path, 'graph.csv'), delimiter=",", data=False)

        if filename is None:
            with open(os.path.join(self.scenario_path, 'config.json'), 'w') as f:
                json.dump(d, f, indent=4)
        else:
            with open(filename, 'w') as f:
                json.dump(d, f, indent=4)

    def save_system_status(self, path: str):
        """
        Save the system status to a JSON file.

        Arguments:
        - path: The directory path where the status will be saved.
        """
        if not hasattr(self, "system_status"):
            raise RuntimeError("You must run single_run() before calling save()")

        filename = os.path.join(path, 'status.json')
        with open(filename, 'w') as f:
            json.dump(self.system_status, f)

    def create_single_random_strategy(self, B: int, T: int, c: int =1) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Create the strategy matrix TxN and randomly selects B/c signals in the TxN matrix to set equals to 1.

        Args:
            B (int): The total budget of lobbyist
            T (int): The number of active time steps of lobbyist
            c (int): The cost to send a signal

        Returns:
            numpy.ndarray: A matrix TxN of 0s with B/c randomly selected elements set to 1.
            list: A list of the (row, column) indices that were set to 1, i.e. the list of (time_step, agent) of sent signals
        """
        matrix = np.zeros((T, self.N), dtype=int)
        total_elements = T * self.N
        num_signals = B//c  # number of signals
        if num_signals > total_elements:
            self._print("Number of signals is greater than the total number of elements in the matrix."
                "Lobbyist will always send signals to all agents at each iteration.")
            num_signals = total_elements
            

        # Generate k unique random linear indices
        linear_indices = np.random.choice(total_elements, size=num_signals, replace=False)

        # Convert linear indices to row and column indices
        row_indices, col_indices = np.unravel_index(linear_indices, (T, self.N))

        # Create a list of (row, column) index pairs
        selected_indices = list(zip(row_indices, col_indices))

        # Set the corresponding elements in the matrix to 1
        matrix[row_indices, col_indices] = 1

        return matrix

    def create_single_random_strategy_per_time(self, B: int, T: int, c: int =1) -> np.ndarray:
        """
        Create the strategy matrix TxN, randomly selects fixed number of signals at each time step in the TxN matrix
        and sets them equals to 1. Per time step, the number of signals is fixed B/(c*T).

        Args:
            B (int): The total budget of lobbyist
            T (int): The number of active time steps of lobbyist
            c (int): The cost to send a signal

        Returns:
            numpy.ndarray: A matrix TxN of 0s with randomly selected elements set to 1. Per time step, the number of signals is fixed B/(c*T).
            list: A list of the (row, column) indices that were set to 1, i.e. the list of (time_step, agent) of sent signals
        """
        inter_per_time = B // (c * T)
        matrix = np.zeros((T, self.N), dtype=int)
        for t in range(T):
            indices = np.random.choice(self.N, inter_per_time, replace=False)
            matrix[t, indices] = 1
        return matrix

    def create_strategies(self):
        """
        Generate and save strategies for the lobbyists.
        """
        self._print('Creating strategies')

        for id in range(self.n_lobbyists):
            data = self.lobbyists_data[id]
            B = data['B']
            c = data['c']
            T = data['T']
            folder = os.path.join(self.strategies_path, str(B))
            os.makedirs(folder, exist_ok=True)
            for run in range(self.nruns):
                filename = f'strategy_{run}.txt'
                path = os.path.join(folder, filename)
                if not os.path.exists(path):
                    matrix = self.create_single_random_strategy(B, T, c)
                    self._print('Saving strategy to file')
                    np.savetxt(path, matrix, fmt="%i")
                else:
                    continue
        self._print('Strategies created')


    def read_random_strategy(self, B: int) -> tuple:
        """
        Read a random strategy for a given B value.

        Arguments:
        - B: The B value associated with the lobbyist's strategy.

        Returns:
        - tuple: A tuple containing the strategy matrix and its filename.
        """
        path = os.path.join(self.strategies_path, str(B))
        strategy_name = random.choice(os.listdir(path))
        filepath = os.path.join(path, strategy_name)
        return np.loadtxt(filepath).astype(int), filepath

    def get_results(self) -> tuple:
        """
        Get the results of the simulation.

        Returns:
        - tuple: A tuple containing the model, system status, and lobbyists' data.
        """
        if not hasattr(self, "system_status"):
            raise RuntimeError("No results available. Did you call run()?")

        return self.model, self.system_status, self.model.lobbyists
    
    def get_model(self):
        return self.model
