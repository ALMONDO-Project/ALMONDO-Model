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
    
    In these experiments we are going to create a population of nl lobbyists where 1/2 of lobbyists are optimist and 1/2 of lobbyists are pessimists. Each lobbyist
    has a budget of 300000 and is active for 3000 iterations. The cost of a single signal is 1. 
    
    """      
    
    NLs = [0, 1, 2, 3, 4, 20] # number of lobbyists in the simulations
    Ns = [2, 3, 4, 5] # number of agents in the simulations
    int_rate = 0.2 # interaction rate of the lobbyists per time-step
    T = 3000 # max number of active steps of lobbyists
    params = {
        'N': 500,
        'p_o': 0.01,
        'p_p': 0.99,
        'initial_distribution': 'uniform',
        'T': 10000,
        'lambda_values': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        'phi_values': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        'base': 'results'
    }    
    
    
    for nl in NLs:
        for n in Ns:
            params['N'] = n
            params['scenario'] = f'small_networks/{nl}_lobbyists/{n}_agents/'
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
                        
            simulator = ALMONDOSimulator(**params, nruns=nruns)
            simulator.execute_experiments(overwrite_runs=False,drop_evolution=True)     

                    
if __name__ == "__main__":
    main(nruns=150)