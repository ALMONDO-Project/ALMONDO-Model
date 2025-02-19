from classes.simulator import ALMONDOSimulator
from functions.strategies import generate_ms, generate_strategies
from functions.strategies import read_random_strategy
import networkx as nx
import json
import os

def main(n_lobbyists, nruns):
            
    params = {
        
        'p_o': 0.01,
        'p_p': 0.99,
        'initial_distribution': 'uniform',
        'T': 10000,
        'lambda_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'phi_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'base_path': 'results',
        'N': 500
    }
                
    simulator = ALMONDOSimulator(**params, n_lobbyists = n_lobbyists, nruns=nruns)
         
    simulator.execute_experiments().save_config()      
    
    with open(f'{params["base_path"]}/config.json', 'w') as f:
        json.dump(params, f)
                    
if __name__ == "__main__":
    main(0, 3)
    main(1, 3)
    main(2, 3)
    main(3, 3)