# Import necessary libraries
from ndlib.models.DiffusionModel import DiffusionModel
import numpy as np
import random
import os
import json
import pickle
from tqdm import tqdm


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
                    "descr": "Underreaction parameter update is dynamic",
                    "range": [True, False],
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
        self.lambdas = None
        self.lobbyists = []  # List to store lobbyist agents influencing the system
        self.system_status = []

    # # Function to set the initial status of nodes (w values for nodes)
    # def set_initial_status(self, configuration=None):
    #     super(AlmondoModel, self).set_initial_status(configuration)
    #     self.status = np.random.uniform(self.n)  # Random uniform initial status for each node in the interval uniform_range 
    
    # Function to set the initial status of nodes (w values for nodes)
    def set_initial_status(self, configuration=None, kind='uniform', uniform_range=[0, 1], unbiased_value=0.5, status=None, gaussian_params=None, initial_lambda = 0.5):
        super(AlmondoModel, self).set_initial_status(configuration)
        # If no status is provided, assign random values to the nodes' status
        if kind == 'uniform':
            # Check that uniform_range[0] >= 0 and uniform_range[1] <= 1
            if uniform_range[0] < 0 or uniform_range[1] > 1:
                raise ValueError("uniform_range must be within [0, 1]")
            self.status = np.random.uniform(low=uniform_range[0], high=uniform_range[1], size=self.n)  # Random uniform initial status for each node in the interval uniform_range 
        
        elif kind == 'unbiased':
            # Check that unbiased_value is in [0,1]
            if unbiased_value < 0 or unbiased_value > 1:
                raise ValueError("unbiased_value must be between 0 and 1")
            self.status = np.full(self.n, unbiased_value)  # Unbiased initial status: each node has the same initial status set by the user
        
        elif kind == 'gaussian_mixture':
            # The initial distribution is generated from a Gaussian mixture
            # The Gaussian mixture should be tuned by parameters passed via gaussian_params
            if gaussian_params is None or 'means' not in gaussian_params or 'stds' not in gaussian_params or 'weights' not in gaussian_params:
                raise ValueError("gaussian_params must contain 'means', 'stds', and 'weights'")
            
            means = gaussian_params['means']
            stds = gaussian_params['stds']
            weights = gaussian_params['weights']
            
            # Check that means, stds, and weights have the same length and sum of weights is 1
            if len(means) != len(stds) or len(means) != len(weights):
                raise ValueError("means, stds, and weights must have the same length")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("The sum of weights must be 1")
            
            # Generate Gaussian mixture
            gaussians = [np.random.normal(loc=means[i], scale=stds[i], size=self.n) for i in range(len(means))]
            mixture = np.zeros(self.n)
            
            # Sample according to the weights
            for i in range(len(weights)):
                indices = np.random.choice(self.n, size=int(self.n * weights[i]), replace=False)
                mixture[indices] = gaussians[i][indices]
            
            # Ensure values are between 0 and 1
            self.status = np.clip(mixture, 0, 1)
        
        elif kind == 'user_defined' and status is not None:
            # Check that status is an array of self.n entries with each entry being a float in [0,1]
            if len(status) != self.n:
                raise ValueError(f"Status array must have {self.n} entries")
            if not all(0 <= val <= 1 for val in status):
                raise ValueError("Each status value must be a float in [0, 1]")
            
            self.status = status  # Use the status provided by the user
        
        else:
            raise ValueError("Invalid kind or missing status for 'user_defined'")
        
        #initialize lambdas
        self.lambdas = np.full(self.n, initial_lambda)

    # Add lobbyist agents to the model
    def add_lobbyists(self, lobbyists_list):
        self.lobbyists.extend(lobbyists_list)
    
    def get_lobbyists_info(self):
        for lobbyist in self.lobbyists:
            print(lobbyist)    
    
    def get_lobbyists(self):
        return self.lobbyists
    
    def get_lobbyist_by_id(self, lobbyist_id):
        return self.lobbyists[lobbyist_id]
    
    def remove_lobbyist_by_id(self, lobbyist_id):
        return self.lobbyists.pop(lobbyist_id)
    
    def change_strategy_by_id(self, lobbyist_id, new_matrix):
        self.lobbyists[lobbyist_id].set_strategy(new_matrix)
    
    def generate_lambda(self, w, s):
        c = 0.1 #deve essere un parametro del modello? della simulazione? lo settiamo noi a 0.1
        return np.abs((1 - s) - w)
             
    # Function to update node status based on a signal and current status (without lobbyist influence)
    def update(self, receivers, s):
        w = self.actual_status[receivers]
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']
        p = w * p_o + (1 - w) * p_p  # Combined probability based on current node's status
        if self.params['model']['l']:
            l = self.generate_lambda(self.actual_status[receivers], s) #ma quindi lambda non dipende in alcun modo dal lambda precedente?
        else:
            l = self.lambdas
        return l * w + (1 - l) * w * (s * (p_o / p) + (1 - s) * ((1 - p_o) / (1 - p)))
        
    # Function to update a node's status taking lobbyist influence into account
    def lupdate(self, w, lobbyist, t):
        m = lobbyist.get_model()  # Get lobbyist parameters remember that m = 1 is optimistic and m = 0 is pessimistic
        s = lobbyist.get_strategy(t)
        c = m * s # Create a signal from the lobbyist at time t
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']
        p = w * p_o + (1 - w) * p_p
        if self.params['model']['l']:
            l = self.generate_lambda(self.actual_status, c)
        else:
            l = self.lambdas
        return (1 - s) * w + s * w * (l + (1 - l) * (c * (p_o / p) + (1 - c) * ((1 - p_o) / (1 - p))))

    # Function to apply the influence of all lobbyists to the system's current status
    def apply_lobbyist_influence(self, w, t):
        for lobbyist in self.lobbyists:
            w = self.lupdate(w, lobbyist, t)  # Apply influence of each lobbyist
        return w

    # Perform one iteration of the model
    def iteration(self, node_status=True):
        actual_status = self.status.copy()  # Copy current status of the nodes
        
        # First iteration, no updates yet, just return the initial status
        if self.actual_iteration == 0:
            self.actual_iteration += 1
            if node_status:
                return {"iteration": 0, "status": {i: value for i, value in enumerate(actual_status)}, "sender": None, "signal": None}
            else:
                return {"iteration": 0, "status": {}, "sender": None, "signal": None}
                
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']
        
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
            actual_status[receivers] = self.update(receivers, signal)

        # Increment the iteration count and update the status of the nodes
        self.actual_iteration += 1
        self.status = actual_status

        # Return the current iteration's information, including the sender and signal
        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": {i: value for i, value in enumerate(actual_status)}, "sender": sender, "signal": signal}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {i: value for i, value in enumerate(actual_status)}, "sender": sender, "signal": signal}

    # Run the model for T iterations
    def iteration_bunch(self, T=100):
        self.T = T
        random.seed(self.seed)  # Set random seed for reproducibility
        for _ in tqdm(range(T)):  # Use tqdm for a progress bar
            its = self.iteration()  # Run one iteration
            self.system_status.append(its)  # Append the result to system status
        return self.system_status

    # Run the model until a steady state is reached or a maximum number of iterations
    def steady_state(self, max_iterations=1000000, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True, drop_evolution=True):
        self.T = max_iterations
        steady_it = 0  # Counter for consecutive steady iterations

        # Iterate until reaching a steady state or max_iterations
        for it in tqdm(range(max_iterations)):
            its = self.iteration(node_status=True)

            # Check if the difference between consecutive states is below the threshold
            if it > 0:
                old = self.system_status[-1]['status']  # Previous status
                actual = its['status']  # Current status
                res = np.abs(old - actual)  # Calculate the difference
                if np.all((res < sensibility)):  # Check if the difference is small enough
                    steady_it += 1
                else:
                    steady_it = 0  # Reset if not steady

            self.system_status.append(its)  # Append current iteration status

            # If steady state is achieved for nsteady consecutive iterations, stop
            if steady_it == nsteady:
                print(f'Convergence reached after {it} iterations')
                return self.system_status

        # Return the status of the system at each iteration (if no steady state is reached)
        return self.system_status
       
    def save_status(self, save_dir, filename='results'):
        output_file = os.path.join(save_dir, f'{filename}.{format}')
        if format == 'json':
            with open(output_file, 'w') as ofile:
                json.dump(self.system_status, ofile)
        elif format == 'pickle':
            with open(output_file, 'wb') as ofile:
                pickle.dump(self.system_status, ofile)
            
            
