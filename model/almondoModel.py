# Import necessary libraries
from ndlib.models.DiffusionModel import DiffusionModel
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import time

# Author information
__author__ = ["Alina Sirbu", "Giulio Rossetti", "Valentina Pansanella"]
__email__ = [
    "alina.sirbu@unipi.it",
    "giulio.rossetti@isti.cnr.it",
    "valentina.pansanella@isti.cnr.it",
]

# Define the AlmondoModel class, which extends ndlib's DiffusionModel
class AlmondoModel(DiffusionModel):

    # Initialization of the model, setting up the graph, and model parameters
    def __init__(self, graph, seed=None):
        super(self.__class__, self).__init__(graph, seed)

        # Define whether the state is continuous or discrete (here it's continuous)
        self.discrete_state = False

        # Define model parameters with descriptions, ranges, and whether they are optional
        self.parameters = {
            "model": {
                "p_o": {  # Probability of an optimistic event
                    "descr": "Probability of event optimist model",
                    "range": [0, 1],
                    "optional": False
                },
                "p_p": {  # Probability of a pessimistic event
                    "descr": "Probability of event pessimist model",
                    "range": [0, 1],
                    "optional": False
                },
                "l": {  # Underreaction parameter
                    "descr": "Underreaction parameter",
                    "range": [0, 1],
                    "optional": False
                }
            },
            "nodes": {},  # Node-specific parameters (empty for now)
            "edges": {}   # Edge-specific parameters (empty for now)
        }

        self.name = "Almondo"  # Name of the model
        self.n = self.graph.number_of_nodes()  # Number of nodes in the graph
        self.seed = seed  # Random seed (for reproducibility)
        self.T = None
        self.status = None  # Status of each node (initially not set)
        self.lobbyists = []  # List to store lobbyist agents influencing the system

    # Function to set the initial status of nodes (w values for nodes)
    def set_initial_status(self, configuration=None, status=None):
        super(AlmondoModel, self).set_initial_status(configuration)
        # If no status is provided, assign random values to the nodes' status
        if status is None:
            self.status = np.random.rand(self.n)  # Random initial status for each node
        else:
            self.status = status  # Use the provided status

    # Add lobbyist agents to the model
    def add_lobbyists(self, lobbyists_list):
        self.lobbyists.extend(lobbyists_list)

    # Function to remove a lobbyist by their ID (not implemented yet)
    def remove_lobbyist(self, lobbyist_id):
        pass      

    # Function to update node status based on a signal and current status (without lobbyist influence)
    def update(self, w, s):
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']
        l = self.params['model']['l']
        p = w * p_o + (1 - w) * p_p  # Combined probability based on current node's status
        return l * w + (1 - l) * w * (s * (p_o / p) + (1 - s) * ((1 - p_o) / (1 - p)))

    # Function to update a node's status taking lobbyist influence into account
    def lupdate(self, w, lobbyist, t):
        m = lobbyist.get_model()  # Get lobbyist parameters
        # Translate lobbyist model type to numeric (optimistic = 1, pessimistic = 0)
        if m == 'optimistic': 
            m = 1 
        elif m == 'pessimistic': 
            m = 0

        # Create a signal from the lobbyist at time t
        s = lobbyist.create_signal(t)
        c = m * s

        # Recalculate probabilities and update status based on lobbyist influence
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']
        l = self.params['model']['l']
        p = w * p_o + (1 - w) * p_p
        return (1 - s) * w + s * w * (l + (1 - l) * (c * (p_o / p) + (1 - c) * ((1 - p_o) / (1 - p))))

    # Function to apply the influence of all lobbyists to the system's current status
    def apply_lobbyist_influence(self, actual_status, t):
        for lobbyist in self.lobbyists:
            actual_status = self.lupdate(actual_status, lobbyist, t)  # Apply influence of each lobbyist
        return actual_status

    # Perform one iteration of the model
    def iteration(self, node_status=True):
        actual_status = self.status.copy()  # Copy current status of the nodes
        
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']

        # First iteration, no updates yet, just return the initial status
        if self.actual_iteration == 0:
            self.actual_iteration += 1
            if node_status:
                return {"iteration": 0, "status": actual_status, "sender": None, "signal": None}
            else:
                return {"iteration": 0, "status": {}, "sender": None, "signal": None}

        # Apply lobbyist influence to current status
        actual_status = self.apply_lobbyist_influence(actual_status, self.actual_iteration)

        # Randomly select a node to send a signal
        sender = random.randint(0, self.n - 1)

        # Generate a signal for the sender node (based on the node's current status)
        p = actual_status[sender] * p_o + (1 - actual_status[sender]) * p_p
        signal = np.random.binomial(1, p)  # Signal is binary (1 or 0)

        # Get the sender's neighbors and update their status based on the signal
        receivers = np.array(list(self.graph.neighbors(sender)))
        if len(receivers) > 0:
            actual_status[receivers] = self.update(actual_status[receivers], signal)

        # Increment the iteration count and update the status of the nodes
        self.actual_iteration += 1
        self.status = actual_status

        # Return the current iteration's information, including the sender and signal
        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": actual_status, "sender": sender, "signal": signal}
        else:
            return {"iteration": self.actual_iteration - 1, "status": actual_status, "sender": sender, "signal": signal}

    # Run the model for T iterations
    def iteration_bunch(self, T=100):
        self.T = T
        random.seed(self.seed)  # Set random seed for reproducibility
        system_status = []  # List to store the status after each iteration
        for it in tqdm(range(T)):  # Use tqdm for a progress bar
            its = self.iteration()  # Run one iteration
            system_status.append(its)  # Append the result to system status
        return system_status

    # Run the model until a steady state is reached or a maximum number of iterations
    def steady_state(self, max_iterations=1000000, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True, drop_evolution=True):
        self.T = max_iterations
        system_status = []  # List to store the system status at each iteration
        steady_it = 0  # Counter for consecutive steady iterations

        # Iterate until reaching a steady state or max_iterations
        for it in tqdm(range(max_iterations)):
            its = self.iteration(node_status=True)

            # Check if the difference between consecutive states is below the threshold
            if it > 0:
                old = system_status[-1]['status']  # Previous status
                actual = its['status']  # Current status
                res = np.abs(old - actual)  # Calculate the difference
                if np.all((res < sensibility)):  # Check if the difference is small enough
                    steady_it += 1
                else:
                    steady_it = 0  # Reset if not steady

            system_status.append(its)  # Append current iteration status

            # If steady state is achieved for nsteady consecutive iterations, stop
            if steady_it == nsteady:
                print(f'Convergence reached after {it} iterations')
                return system_status

        # Return the status of the system at each iteration (if no steady state is reached)
        return system_status
    
    def get_T(self):
        return self.T
