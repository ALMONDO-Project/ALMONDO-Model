from classes.simulator import ALMONDOSimulator
from functions.strategies import generate_ms, generate_strategies
import networkx as nx
import json
import os

def main(n_lobbyists, nruns):
        
    params = {
        'p_o': 0.01,
        'p_p': 0.99,
        'initial_distribution': 'uniform',
        'T': 10000,
        'n_lobbyists': n_lobbyists,   
        'lambda_values': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'phi_values': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'nruns': nruns
    }
    
    N = 500
    graph = nx.complete_graph(N)        
    
    params['ms'] = generate_ms(params['n_lobbyists'])
    
    simulator = ALMONDOSimulator(**params, graph=graph)     
    simulator.execute_experiments().save_config()      
    
                    
if __name__ == "__main__":
    main(0, 3)