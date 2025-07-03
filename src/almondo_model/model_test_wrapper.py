# -*- coding: utf-8 -*-
"""
Test to Almondo model and its wrapper for the simulator. 

# Author information
__author__ = ["Fabrizio Fornari", "Verdiana Del Rosso"]
__email__ = [
    "fabrizio.fornari@unicam.it"
    "verdiana.delrosso@unicam.it",
]
"""
# Import necessary libraries
import ndlib.models.ModelConfig as mc
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import json
import os
from classes.almondoModel import AlmondoModel
from classes.simulator import ALMONDOSimulator
from classes.almondoModel import AlmondoModel # This will be imported from ndlib once the model is loaded there
from functions.metrics_functions import nclusters

class SimulationWrapper(object):

    # constructor for Python
    def __init__(self, model):
        self.model = model

    def _print(self, *args, **kwargs):
        """Custom print method that respects verbose setting"""
        if self.model.verbose:
            print(*args, **kwargs)

    # code to let initialize the simulator for a new simulation
    # that is, re-initialize the model to its initial state, and set the
    # new random seed
    def setSimulatorForNewSimulation(self, random_seed): 
        """Initialize the model with a new random seed and set the initial configuration."""
        # Set the random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        # self.model.config_model(self.model.lambdas[0], self.model.phis[0])
        self._print('Creating configuration object')
        config = mc.Configuration()

        self._print('Assigning p_o and p_p parameters')
        config.add_model_parameter("p_o", self.model.p_o)
        config.add_model_parameter("p_p", self.model.p_p)
        self._print(f'p_o={self.model.p_o}, p_p={self.model.p_p}')

        # Configure lambda values for each agent
        if isinstance(self.model.lambdas[0], list):
            for i in self.model.graph.nodes():
                config.add_node_configuration("lambda", i, self.model.lambdas[0][i])
        elif isinstance(self.model.lambdas[0], float):
            self._print('Assigning homogeneous lambda')
            for i in self.model.graph.nodes():
                config.add_node_configuration("lambda", i, self.model.lambdas[0])
        else:
            raise ValueError("lambda_v must be a float or a list")

        # Configure phi values for each agent
        if isinstance(self.model.phis[0], list):
            for i in self.model.graph.nodes():
                config.add_node_configuration("phi", i, self.model.phis[0][i])
        elif isinstance(self.model.phis[0], float):
            self._print('Assigning homogeneous phi')
            for i in self.model.graph.nodes():
                config.add_node_configuration("phi", i, self.model.phis[0])
        else:
            raise ValueError("phi_v must be a float or a list")

        # Initialize the model with the graph and configuration
        self._print('Configuring model: assigning graph, parameters, and initial distributions of weights')
        self.model.model = AlmondoModel(self.model.graph, seed=random_seed, verbose=self.model.verbose)
        self.model.model.set_initial_status(config, kind=self.model.initial_distribution, status=self.model.initial_status)

        self._print('Assign strategies to lobbyists if any')
        if self.model.n_lobbyists > 0:
            for id in tqdm(self.model.lobbyists_data):
                data = self.model.lobbyists_data[id]
                B = data['B']
                m = data['m']
                matrix, name = self.model.read_random_strategy(B)
                # Add lobbyist with strategy to the model
                self.model.model.add_lobbyist(m, matrix)
                self.model.lobbyists_data[id]['strategies'].append(name)

        self._print('Configuration ended')

    # code to let ask the simulator to perform a step of simulation
    def performOneStepOfSimulation(self):
        """Perform one step of simulation."""
        its = self.model.model.iteration() 
        self.model.model.system_status.append(its)
        self.model.system_status = self.model.model.system_status
        # print('config_path: ', self.model.config_path)
        # self.model.save_system_status(self.model.config_path)
        """
        # Create directory if it doesn't exist
        os.makedirs(self.model.config_path, exist_ok=True)

        filename = os.path.join(self.model.config_path, 'status.json')
        try:
            with open(filename, 'w') as f:
                json.dump(self.model.model.system_status, f, indent=2)
                print(f"File saved successfully to: {filename}")
        except PermissionError:
            print(f"Permission denied: Cannot write to {filename}")
        except FileNotFoundError:
            print(f"Path not found: {os.path.dirname(filename)}")
        except Exception as e:
            print(f"Error saving file: {e}")
        """            


    # code to let ask the simulator to perform a
    # "whole simulation"
    # (i.e., until a stopping condition is found by the simulator)
    def performWholeSimulation(self):
        """Perform the whole simulation until a stopping condition is met."""
        # self.model.system_status = self.model.model.steady_state(max_iterations=self.model.T,drop_evolution = True)
        self.model.system_status = self.model.model.iteration_bunch(T = 100, progress_bar=False)
        # Here you should replace 'performWholeSimulation()' with
        # your method to perform a whole simulation, i.e. iteration_bunch() to perform
        # multiple steps of simulation

    # code to let multivesta ask the simulator the current simulated time (or number of simulation step)
    def getTime(self):
        return float(self.model.model.actual_iteration -1)


    # code to let ask the simulator to return the value of the
    # specified observation in the current state of the simulation
    def rval(self, observation):
        # Model configuration
        if observation == 'nclusters':
            weights = self.model.model.actual_status
            probabilities = self.model.model.params['model']['p_o']*weights + self.model.model.params['model']['p_p']*(1-weights)
            nclusters = nclusters(probabilities, 0.01)
            return float(nclusters)
        elif observation.startswith('p_'):
            agent = int(observation[2:])
            weight = self.model.model.actual_status[agent]
            probability = self.model.model.params['model']['p_o']*weight + self.model.model.params['model']['p_p']*(1-weight)
            return float(probability)
        #Here you should replace with
        # your method to evaluate an observation in the 
        # current simulation step 
        # observation = specific agent
        # eval = subjective probability of the specific agent


if __name__ == '__main__':
    #Here you should put any initialization code you need to create an instance of
    #your model_file_name class
    
    nl = 0 # number of lobbyists in the simulations
    n = 2 # number of agents in the simulations
    # nruns = 1 # number of runs for the simulations
    int_rate = 0.2 # interaction rate of the lobbyists per time-step
    T = 3000 # max number of active steps of lobbyists
    params = {
        'p_o': 0.01,
        'p_p': 0.99,
        'initial_distribution': 'uniform',
        'T': 10000,
        'lambda_values': [0.29],
        'phi_values': [0.0], 
        'base': 'results',
    }    
    

    params['N'] = n
    params['scenario'] = f'{n}_agents/{nl}_lobbyists/'
    params['n_lobbyists'] = nl
    b = int(int_rate*params['N']*T) # budget of lobbyists in the simulation 
    if nl > 0:
        params['lobbyists_data'] = dict()
        for id in range(nl):
            params['lobbyists_data'][id] = {'m': id%2, 'B': b, 'c': 1, 'strategies': [], 'T': T}    

    os.makedirs(params['base'], exist_ok=True)
    path = os.path.join(params['base'], params['scenario'])
    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, 'initial_config.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f'performing simulations for {params["scenario"]}')
                
    simulator = ALMONDOSimulator(**params, verbose=False)
    print(f'Starting configuration lambda={params['lambda_values'][0]}, phi={params['phi_values'][0]}')
    simulator.config_path = os.path.join(simulator.scenario_path, f'{params['lambda_values'][0]}_{params['phi_values'][0]}')
    os.makedirs(simulator.config_path, exist_ok=True)

    wrapper = SimulationWrapper(simulator)
    wrapper.setSimulatorForNewSimulation(random_seed=42)  # Initialize with a random seed

    for i in range(5):
        wrapper.performOneStepOfSimulation()
        # wrapper.performWholeSimulation()
        for a in range(wrapper.model.N):
            print(f'Agent {a}: Probability = {wrapper.rval(f"p_{a}")}')

    wrapper.setSimulatorForNewSimulation(random_seed=42)  # Initialize with same random seed to check consistency

    for i in range(5):
        wrapper.performOneStepOfSimulation()
        # wrapper.performWholeSimulation()
        for a in range(wrapper.model.N):
            print(f'Agent {a}: Probability = {wrapper.rval(f"p_{a}")}')