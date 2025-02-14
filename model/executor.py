from classes.simulator import ALMONDOSimulator
from functions.strategies import generate_ms, generate_strategies
import networkx as nx
import json
import os

def main(n_lobbyists, nruns):
    
    SCENARIO = f"{n_lobbyists}_lobbyists"
    # These settings are the same across all scenarios and runs
    MAIN_PATH = ''
    scenario_path = os.path.join(MAIN_PATH, "simulations", SCENARIO)
    os.makedirs(scenario_path, exist_ok=True)  
    
    ########### CONSTANT SETTINGS FOR EACH EXPERIMENT ###########
    params = {
        'p_o': 0.01,
        'p_p': 0.99,
        'initial_distribution': 'uniform',
        'T': 10000,
        'n_lobbyists': n_lobbyists,    # SINGLE LOBBYIST!
        'lambda_values': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'phi_values': [0.0, 0.5, 1.0]
    }
    
    params['ms'] = generate_ms(params['n_lobbyists'])
    
    # Define the graph used for the simulation
    graphparams = {
        'graph_N': 500,
        'graph_type': 'complete'
    }

    N = graphparams['graph_N']
    graph = nx.complete_graph(N)
    
    # Create and save the possible strategies
    strategies_path = os.path.join(scenario_path, "strategies")
    os.makedirs(strategies_path, exist_ok=True)
    generate_strategies(strategies_path, nruns, N)
    
    simulator = ALMONDOSimulator(**params, graph=graph)     
    final_data = simulator.execute_experiments(nruns=nruns)            
                
if __name__ == "__main__":
    main(1, 10)