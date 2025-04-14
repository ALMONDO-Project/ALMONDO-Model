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
    initial_distribution (string): kind of initial distribution (uniform distribution ('uniform') or customizable distributions ('custom'), only entering a list of values is implemented at the present time)
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
         
    Pass these parameters to ALMONDOSimulator, specifying the number of runs nruns you want to perform 
    
    Use the method ALMONDOSimulator.execute_experiments() to run the simulations. Use the attribute overwrite_runs to overwrite existing runs,
     drop_evolution = True to save only last iteration in the status.json file and the initial status to pass the initial configuration of prior w.
    
    In these experiments we are going to simulate how the initial status influences the final belief of agents.
    For each experiment each lobbyist has a predefined budget and is active for a specific number of iterations. The cost of a single signal is 1. 

    """      
    
    NLs = [2] #[1, 2, 3, 4, 20] #number of lobbyists in the simulations
    Ws = [0.2, 0.4, 0.6, 0.8]  # initial prior w of all agents in the simulation
    b = 10 # budget of lobbyists in the simulation (in k)
    params = {
        'N': 10,
        'p_o': 0.01,
        'p_p': 0.99,
        'initial_distribution': 'custom',
        'T': 20000, # max number of iterations
        'lambda_values': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        'phi_values': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        'base': 'results'
    }    
    
    
    for nl in NLs:
        for w in Ws: 
            params['scenario'] = f'sensitivity_prior/{nl}_lobbyists/{w}_priors/'
            params['n_lobbyists'] = nl
            if nl > 0:
                params['lobbyists_data'] = dict()
                for id in range(nl):
                    params['lobbyists_data'][id] = {'m': id%2, 'B': b*1000, 'c': 1, 'strategies': [], 'T': 3000}    
    
            os.makedirs(params['base'], exist_ok=True)
            path = os.path.join(params['base'], params['scenario'])
            os.makedirs(path, exist_ok=True)
            
            with open(os.path.join(path, 'initial_config.json'), 'w') as f:
                json.dump(params, f, indent=4)
            
            print(f'performing simulations for {params["scenario"]}')

            initial_status = [w]*params['N']
            params['initial_status'] = initial_status
            simulator = ALMONDOSimulator(**params, nruns=nruns)
            simulator.execute_experiments(overwrite_runs=False,drop_evolution=True,)     

                    
if __name__ == "__main__":
    main(nruns=150)
