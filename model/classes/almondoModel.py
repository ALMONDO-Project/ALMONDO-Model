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

class LobbyistAgent:
        
    def __init__(self, m, strategy):        
        self.m = m #è uno 0 o un 1
        self.strategy = strategy #è una matrice T x N
    
    def get_model(self):
        return self.m
    
    def get_current_strategy(self, t):
        return self.strategy[t]


# Define the AlmondoModel class, which extends ndlib's DiffusionModel
class AlmondoModel(DiffusionModel):

#################################### MODEL INITIALIZATION FUNCTIONS ##################################################################        
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
                }
            },
            "nodes": {
                "lambda":{
                    "descr": "...",
                    "range": [0,1],
                    "optional": False
                    },
                "phi":{
                    "descr":"...",
                    "range":[0,1],
                    "optional":False}
                },  # Node-specific parameters (empty for now)
            "edges": {}   # Edge-specific parameters (empty for now)
        }
            
        self.name = "Almondo"  # Name of the model
        self.n = self.graph.number_of_nodes()  # Number of nodes in the graph
        self.seed = seed  # Random seed (for reproducibility)
        self.T = None
        self.status = None  # Status of each node to pass to subsequent iteration (initially not set)
        
    def set_initial_status(self, configuration=None, kind='uniform', uniform_range=[0, 1], unbiased_value=0.5, status=None, gaussian_params=None, initial_lambda = 0.5):
        super(AlmondoModel, self).set_initial_status(configuration)
        
        self.status = np.random.uniform(low=0, high=1, size=self.n)
        
        self.lambdas = np.array([el for el in self.params['nodes']['lambda'].values()])
        self.phis = np.array([el for el in self.params['nodes']['phi'].values()])

        self.lobbyists = []
        self.system_status = []
    
    def add_lobbyist(self, m, strategy):
        newl = LobbyistAgent(m, strategy)
        self.lobbyists.append(newl)
          
######################################################################################################################        
        
    def generate_lambda(self, w, s, phi, lam):
        f = np.abs((1 - s) - w)
        return phi * f + (1 - phi) * lam
        
    # Function to update node status based on a signal and current status (without lobbyist influence)
    def update(self, receivers, s):
        w = self.actual_status[receivers]
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']
        p = w * p_o + (1 - w) * p_p  # Combined probability based on current node's status
        phi = self.phis[receivers]
        lam = self.lambdas[receivers]
        l = self.generate_lambda(self.actual_status[receivers], s, phi, lam) 
        return l * w + (1 - l) * w * (s * (p_o / p) + (1 - s) * ((1 - p_o) / (1 - p)))

    # Function to update a node's status taking lobbyist influence into account
    def lupdate(self, w, lobbyist, t):
        m = lobbyist.get_model()  # Get lobbyist parameters remember that m = 1 is optimistic and m = 0 is pessimistic
        s = lobbyist.get_current_strategy(t)
        if s is not None:
          c = m * s # Create a signal from the lobbyist at time t
          p_o = self.params['model']['p_o']
          p_p = self.params['model']['p_p']
          p = w * p_o + (1 - w) * p_p
          phi = self.phis
          lam = self.lambdas
          l = self.generate_lambda(w, s, phi, lam)
          return (1 - s) * w + s * w * (l + (1 - l) * (c * (p_o / p) + (1 - c) * ((1 - p_o) / (1 - p))))
        else:
          return w

    # Function to apply the influence of all lobbyists to the system's current status
    def apply_lobbyist_influence(self, w, t):
        for lobbyist in self.lobbyists:
            w = self.lupdate(w, lobbyist, t)  # Apply influence of each lobbyist
        return w

    # Perform one iteration of the model
    def iteration(self, node_status=True):
        self.actual_status = self.status.copy()  # Copy current status of the nodes

        # First iteration, no updates yet, just return the initial status
        if self.actual_iteration == 0:
            self.actual_iteration += 1
            if node_status:
                return {"iteration": 0, "status": {i: value for i, value in enumerate(self.actual_status)}, "sender": None, "signal": None}
            else:
                return {"iteration": 0, "status": {}, "sender": None, "signal": None}

        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']

        # Apply lobbyist influence to current status
        try:
            self.actual_status = self.apply_lobbyist_influence(self.actual_status, self.actual_iteration)
        except IndexError:
            pass
        
        # Apply sender signal to current status (after lobbyist intervention)
        sender = random.randint(0, self.n - 1)
        p = self.actual_status[sender] * p_o + (1 - self.actual_status[sender]) * p_p
        signal = np.random.binomial(1, p)  # Signal is binary (1 or 0)
        receivers = np.array(list(self.graph.neighbors(sender)))
        if len(receivers) > 0:
            self.actual_status[receivers] = self.update(receivers, signal)

        # Increment the iteration count and update the status of the nodes
        self.actual_iteration += 1
        self.status = self.actual_status

        # Return the current iteration's information, including the sender and signal
        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": {i: value for i, value in enumerate(self.actual_status)}}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {i: value for i, value in enumerate(self.actual_status)}}

    # Run the model for T iterations
    def iteration_bunch(self, T=100):
        self.T = T
        random.seed(self.seed)  # Set random seed for reproducibility
        for _ in tqdm(range(T)):  # Use tqdm for a progress bar
            its = self.iteration()  # Run one iteration
            self.system_status.append(its)  # Append the result to system status
        return self.system_status

    # Run the model until a steady state is reached or a maximum number of iterations
    def steady_state(self, max_iterations=1000000, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True, drop_evolution=False):
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
