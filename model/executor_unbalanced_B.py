from classes.simulator import ALMONDOSimulator
import json
import os


    

def main(nruns):
    
    """
    Set up an experiment here. 
    With the 'params' dictionaries you can specify immutable parameters across experiments.
    Such parameters are:
    p_o (float): probability of optimist model
    p_p (float): probability of pessimist model
    initial_distribution (string): kind of initial distribution (customizable distributions are not implemented at the present time, only uniform initial distirbution can be used)
    lambda_values (list): list of lambda_v to test in the experiments, a lamdba value can be a float or a list of length N
    phi_values (list): list of phi_v to test in the experiment, a phi_v can be a float or a list of length N
    base (str): folder where results are stored
    scenario (str): folder where you want this set of experiments to be stored
    N (int): number of agents in the population
    lobbyists_data (dict): each key is the id of a lobbyist, each value is the information for that lobbyist
        a single lobbyist is identified by:
         - m (int): model, where 1 = optimist and 0 = pessimist
         - B (int): budget
         - c (int): cost of a signal
         - strategies (list): list of strategies to use (filenames to retrieve the strategies form)
         - T (int): number of active time steps 
         
    Pass these parameters to ALMONDOSimulator, specifying the number of runs nruns you want to perform. 
    
    Use the method ALMONDOSimulator.execute_experiments() to run the simulations. Use the attribute overwrite_runs to overwrite existing runs. 
    
    In these experiments we are going to create a population of 2 lobbyists where 1/2 of lobbyists are optimist and 1/2 of lobbyists are pessimists.
    One lobbyist has a budget of 400000 and the other of 200000. The cost of a single signal is 1. 
    
    """      
      
    params = {
        
        'p_o': 0.01,
        'p_p': 0.99,
        'initial_distribution': 'uniform',
        'T': 10000,
        'lambda_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'phi_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'base': 'results',
        'scenario': 'unbalanced_budgets',
        'N': 500,
        'lobbyists_data': {
            0: {'m': 1, 'B': 200000, 'c': 1, 'strategies': [], 'T': 3000},
            1: {'m': 0, 'B': 400000, 'c': 1, 'strategies': [], 'T': 3000}
        },
        'n_lobbyists': 2
    }    
    
    os.makedirs(params['base'], exist_ok=True)
    path = os.path.join(params['base'], params['scenario'])
    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, 'initial_config.json'), 'w') as f:
        json.dump(params, f, indent=4)
                
    simulator = ALMONDOSimulator(**params, nruns=nruns)
    simulator.execute_experiments(overwrite_runs=False,drop_evolution=False)     

                    
if __name__ == "__main__":
    main(100)