from ndlib.models.DiffusionModel import DiffusionModel
from tqdm import tqdm
import numpy as np
import random
import os
import json
import pickle

# Author information
__author__ = ["Alina Sirbu", "Giulio Rossetti", "Valentina Pansanella"]
__email__ = [
    "alina.sirbu@unipi.it",
    "giulio.rossetti@isti.cnr.it",
    "valentina.pansanella@isti.cnr.it",
]


class AlmondoModel(DiffusionModel):
    """
    A class to model diffusion in a network, with additional functionality 
    for lobbying agents influencing node states. Extends DiffusionModel from ndlib.
    """

    class LobbyistAgent:
        """
        A class to represent a lobbyist agent influencing nodes in the network.

        Attributes:
            m: Model of the lobbyist (0 for pessimistic, 1 for optimistic).
            strategy: A matrix representing the strategy for node influence over time.
        """

        def __init__(self, m: int, strategy: np.ndarray):
            """
            Initialize a LobbyistAgent.
            
            Arguments:
                m (int): Model type, 0 for pessimistic, 1 for optimistic.
                strategy (np.ndarray): The strategy matrix (T x N) for the agent's influence.
            """
            self.m = m  # 0 (pessimistic) or 1 (optimistic)
            self.strategy = strategy  # Strategy matrix

        def get_model(self) -> int:
            """Returns the model of the lobbyist (0 or 1)."""
            return self.m
        
        def get_strategy(self) -> np.ndarray:
            """Returns the full strategy matrix of the lobbyist."""
            return self.strategy
        
        def get_current_strategy(self, t: int) -> float:
            """
            Get the strategy of the lobbyist at time t.

            Arguments:
                t (int): The time step for which the strategy is required.

            Returns:
                float: The strategy value for time step t.
            """
            return self.strategy[t]

    def __init__(self, graph, seed: int = None):
        """
        Initialize the Almondo diffusion model.

        Arguments:
            graph (networkx.Graph): The graph representing the network of nodes.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(graph, seed)
        self.discrete_state = False  # Use continuous state model
        self.parameters = {
            "model": {
                "p_o": {  # Probability of optimistic events
                    "descr": "Probability of event optimist model",
                    "range": [0, 1],
                    "optional": False
                },
                "p_p": {  # Probability of pessimistic events
                    "descr": "Probability of event pessimist model",
                    "range": [0, 1],
                    "optional": False
                }
            },
            "nodes": {
                "lambda": {  # Node-specific parameter lambda
                    "descr": "...",
                    "range": [0, 1],
                    "optional": False
                },
                "phi": {  # Node-specific parameter phi
                    "descr": "...",
                    "range": [0, 1],
                    "optional": False
                }
            },
            "edges": {}  # Empty, no edge-specific parameters
        }
        self.name = "Almondo"
        self.n = self.graph.number_of_nodes()  # Number of nodes in the graph
        self.seed = seed
        self.T = None
        self.status = None  # Initial status, will be assigned later

    def set_initial_status(
        self, 
        configuration=None, 
        kind: str = 'uniform'
    ) -> None:
        
        """
        Sets the initial status of all nodes in the network.

        Arguments:
            configuration (optional): Configuration for setting initial status.
            kind (str, optional): Type of distribution for status ('uniform' by default).
            uniform_range (list, optional): Range for the uniform distribution.
            unbiased_value (float, optional): Value to initialize nodes to if no status is provided.
            status (optional): Predefined status values for nodes.
            gaussian_params (optional): Parameters for Gaussian initialization (not implemented).
            initial_lambda (float, optional): Initial value for the lambda parameter of nodes.

        Returns:
            None
        """
        super().set_initial_status(configuration)
        
        if kind == 'uniform':
            self.status = np.random.uniform(low=0, high=1, size=self.n)  # Random uniform status for each node
        else:
            raise ValueError("Other types of initial distributions are not implemented yet.")
        
        # Set node-specific parameters (lambda and phi)
        self.lambdas = np.array([el for el in self.params['nodes']['lambda'].values()])
        self.phis = np.array([el for el in self.params['nodes']['phi'].values()])

        self.lobbyists = []  # Empty list of lobbyists initially
        self.system_status = []  # Empty list to store system status over time

    def add_lobbyist(self, m: int, strategy: np.ndarray) -> None:
        """
        Adds a lobbyist agent to the model.

        Arguments:
            m (int): Model type of the lobbyist (0 for pessimistic, 1 for optimistic).
            strategy (np.ndarray): Strategy matrix for the lobbyist.

        Returns:
            None
        """
        new_lobbyist = self.LobbyistAgent(m, strategy)
        self.lobbyists.append(new_lobbyist)

    def get_lobbyists(self, strategy: bool = False) -> list:
        """
        Retrieves the list of lobbyists.

        Arguments:
            strategy (bool, optional): Whether to include the strategy matrix in the return data.

        Returns:
            list: A list of dictionaries with lobbyist details.
        """
        ls = []
        if len(self.lobbyists) > 0:
            if strategy:
                for i, l in enumerate(self.lobbyists):
                    ls.append({
                        'id': i,
                        'model': l.get_model(),
                        'strategy': l.get_strategy()
                    })
            else:
                for i, l in enumerate(self.lobbyists):
                    ls.append({
                        'id': i,
                        'model': l.get_model()
                    })
            return ls
        else:
            return None

    def generate_lambda(self, w: np.ndarray, s: float, phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """
        Generates a lambda value for each node based on the current status and parameters.

        Arguments:
            w (np.ndarray): The current status of nodes.
            s (float): Strategy value for influence.
            phi (np.ndarray): The phi parameter for nodes.
            lam (np.ndarray): The lambda parameter for nodes.

        Returns:
            np.ndarray: The calculated lambda values for the nodes.
        """
        f = np.abs((1 - s) - w)  # Difference between current state and strategy
        return phi * f + (1 - phi) * lam  # Weighted influence

    def update(self, receivers: np.ndarray, s: float) -> np.ndarray:
        """
        Updates the status of receiver nodes based on a signal and their current status.

        Arguments:
            receivers (np.ndarray): List of nodes to receive the signal.
            s (float): The signal to propagate to receivers.

        Returns:
            np.ndarray: Updated statuses for the receiver nodes.
        """
        w = self.actual_status[receivers]  # Current status of the receiver nodes
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']
        p = w * p_o + (1 - w) * p_p  # Combined probability based on current node's status
        phi = self.phis[receivers]
        lam = self.lambdas[receivers]
        l = self.generate_lambda(self.actual_status[receivers], s, phi, lam)
        return l * w + (1 - l) * w * (s * (p_o / p) + (1 - s) * ((1 - p_o) / (1 - p)))

    def lupdate(self, w: np.ndarray, lobbyist: LobbyistAgent, t: int) -> np.ndarray:
        """
        Updates the status of nodes with the influence of a given lobbyist.

        Arguments:
            w (np.ndarray): The current status of nodes.
            lobbyist (LobbyistAgent): The lobbyist influencing the nodes.
            t (int): Current time step for the lobbyist's strategy.

        Returns:
            np.ndarray: Updated node statuses considering lobbyist influence.
        """
        m = lobbyist.get_model()
        s = lobbyist.get_current_strategy(t)
        if s is not None:
            c = m * s  # Create a signal from the lobbyist at time t
            p_o = self.params['model']['p_o']
            p_p = self.params['model']['p_p']
            p = w * p_o + (1 - w) * p_p
            phi = self.phis
            lam = self.lambdas
            l = self.generate_lambda(w, s, phi, lam)
            return (1 - s) * w + s * w * (l + (1 - l) * (c * (p_o / p) + (1 - c) * ((1 - p_o) / (1 - p))))
        else:
            return w

    def apply_lobbyist_influence(self, w: np.ndarray, t: int) -> np.ndarray:
        """
        Applies the influence of all lobbyists to the current node statuses.

        Arguments:
            w (np.ndarray): The current status of nodes.
            t (int): The current time step.

        Returns:
            np.ndarray: The updated statuses after lobbyist influence.
        """
        for lobbyist in self.lobbyists:
            w = self.lupdate(w, lobbyist, t)
        return w

    def iteration(self, node_status: bool = True) -> dict:
        """
        Performs one iteration of the diffusion process, updating node statuses.

        Arguments:
            node_status (bool, optional): Whether to return the node status after the iteration.

        Returns:
            dict: Information about the current iteration, including updated node statuses.
        """
        self.actual_status = self.status.copy()

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            if node_status:
                return {"iteration": 0, "status": {i: value for i, value in enumerate(self.actual_status)}}
            else:
                return {"iteration": 0, "status": {}}

        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']

        try:
            self.actual_status = self.apply_lobbyist_influence(self.actual_status, self.actual_iteration)
        except IndexError:
            pass
        
        sender = random.randint(0, self.n - 1)
        p = self.actual_status[sender] * p_o + (1 - self.actual_status[sender]) * p_p
        signal = np.random.binomial(1, p)
        receivers = np.array(list(self.graph.neighbors(sender)))
        if len(receivers) > 0:
            self.actual_status[receivers] = self.update(receivers, signal)

        self.actual_iteration += 1
        self.status = self.actual_status

        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": {i: value for i, value in enumerate(self.actual_status)}}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {}}

    def iteration_bunch(self, T: int = 100) -> list:
        """
        Runs the model for a specified number of iterations.

        Arguments:
            T (int, optional): The number of iterations to run.

        Returns:
            list: A list of the system status (dictionaries) at each iteration.
        """
        self.T = T
        random.seed(self.seed)
        for _ in tqdm(range(T)):
            its = self.iteration()
            self.system_status.append(its)
        return self.system_status

    # Run the model until a steady state is reached or a maximum number of iterations
    def steady_state(self, 
                     max_iterations: int=1000000, 
                     nsteady:int=1000, 
                     sensibility:float=0.00001, 
                     node_status:bool=True, 
                     progress_bar:bool=True, 
                     drop_evolution:bool=False) -> list:
        
        """ 
        Runs the model untill convergence or stopping condition is met.add()
        
        Arguments: 
            max_iterations (optional, int): stopping condition
            nsteady (optional, int): number of iterations with minimum number of opinion changes to declare convergence
            sensibility (optional, float): maximum opinion change tollerated to compute convergence
            node_status (optional, bool): keep node statues
            progress_bar (optional, bool): show progress bar
            drop_evolution (optional, bool): keep in memory iterations dictionary (keep true if you want evolution plots in the end)
            
        Returns:
            list: A list of the system status at each iteration (dictionary)
        
        """
        
        self.T = max_iterations
        steady_it = 0  # Counter for consecutive steady iterations
        # Iterate until reaching a steady state or max_iterations
        for it in tqdm(range(max_iterations), disable=not progress_bar):
            its = self.iteration(node_status=True)
            # Check if the difference between consecutive states is below the threshold
            if it > 0:
                old = np.array([el for el in self.system_status[-1]['status'].values()]) # Previous status
                actual = np.array([el for el in its['status'].values()]) # Current status
                res = np.abs(old - actual)  # Calculate the difference
                if np.all((res < sensibility)):  # Check if the difference is small enough
                    steady_it += 1
                else:
                    steady_it = 0  # Reset if not steady

            self.system_status.append(its)  # Append current iteration status

            # If steady state is achieved for nsteady consecutive iterations, stop
            if steady_it == nsteady:
                print(f'Convergence reached after {it} iterations')
                return self.system_status[:-nsteady]
            
            if drop_evolution:
              self.system_status = [its]

        # Return the status of the system at each iteration (if no steady state is reached)
        return self.system_status

