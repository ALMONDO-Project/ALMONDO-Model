from classes.simulator import ALMONDOSimulator
from viz.OpinionEvolution import OpinionEvolution
from viz.OpinionDistribution import OpinionDistribution
from classes.metrics import AverageMetrics
import networkx as nx
import numpy as np
import json
import pickle
import os
import random


def generate_ms(n_lobbyists: int) -> list[int]:
    return [nl % 2 for nl in range(n_lobbyists)] #numero "ugale" di lobbisti ottimisti e pessimisti (se il totale è pari, altrimenti c'è un 1 in più)
    
def generate_strategies(folder_path: str,
                        n_possibilites: int, 
                        N: int,         # numero nodi 
                        T: int = 3000,  # timestep totali 
                        B: int = 30000, # budget
                        c: int = 1      # costo unitario
                        ) -> None:
    """
    Creare una lista di n_possibilities (uguale a NRUNS?) matrici e salvare su file.
    Le matrici sono binarie e di forma timestep x n_lobbisti.
    Ogni entrata (t, i) indica se il lobbista interagisce
    con il nodo i al tempo t o no. Il numero totale di interazioni
    moltiplicato per il costo unitario sia uguale il budget.
    Temporaneamente assumiamo che i timestep abbiano un numero
    costante di interazioni
    """
    inter_per_time = B // (c * T)
    for i in range(n_possibilites):
        m = np.zeros((T, N), dtype=int)
        for t in range(T):
            indices = np.random.choice(N, inter_per_time, replace=False)
            m[t, indices] = 1
        
        np.savetxt(os.path.join(folder_path, f"strategy_{i}.txt"), m, fmt="%i")
    return

def read_random_strategies(strategies_path: str, n_lobbyists: int) -> list[np.ndarray]:
    strategy_names = random.sample(os.listdir(strategies_path), n_lobbyists)
    strategies = []
    for strategy_name in strategy_names:
        filepath = os.path.join(strategies_path, strategy_name)
        print(f"Reading f{filepath}")
        strategies.append(np.loadtxt(filepath).astype(int))
    return strategies

def transform(w: list, settings: dict):
    w = np.array(w)
    p = w * settings['p_o'] + (1 - w) * settings['p_p']
    p = p.tolist()
    return p


def main(scenario, n_lobbyists):
    SCENARIO = scenario
    # These settings are the same across all scenarios and runs
    MAIN_PATH = ''
    scenario_path = os.path.join(MAIN_PATH, "simulations", SCENARIO)
    os.makedirs(scenario_path, exist_ok=True)  
    
    ########### CONSTANT SETTINGS FOR EACH EXPERIMENT ###########
    # Define the main settings for the simulation
    settings = {
        'p_o': 0.01,
        'p_p': 0.99,
        'initial_distribution': 'uniform',
        'path': scenario_path,
        'T': 10000,
        'n_lobbyists': n_lobbyists    # SINGLE LOBBYIST!
    }
    
    NRUNS = 100

    # Define the graph used for the simulation
    graphparams = {
        'N': 500,
        'type': 'complete'
    }

    N = graphparams['N']
    graph = nx.complete_graph(N)
    nx.write_edgelist(graph, settings['path'] + 'graph.csv', delimiter=',')
    
    # Create and save the possible strategies
    strategies_path = os.path.join(scenario_path, "strategies")
    os.makedirs(strategies_path, exist_ok=True)
    generate_strategies(strategies_path, NRUNS, N)
    #############################################################

    # Define the possible values of lambdas and phis to test 
    lambda_values = [0.0, 0.5, 1.0]
    phi_values = [0.0, 0.5, 1.0]
    
    for _, (lambda_v, phi_v) in enumerate([(l, p) for l in lambda_values for p in phi_values]):
            # Parameters specific to the execution of a simulation
            simparams = {}
            
            configparams = {
                'lambdas': lambda_v,
                'phis': phi_v
            }
            
            configpath = os.path.join(scenario_path, f'{lambda_v}_{phi_v}/')
            os.makedirs(configpath, exist_ok=True)

            paramsdict = {**settings, **configparams, **simparams}
            with open(configpath + 'params.pkl', 'wb') as f:
                pickle.dump(paramsdict, f)        
            
            final_data = []
            
            for run in range(NRUNS):
                runpath = os.path.join(configpath, f'{run}')
                os.makedirs(runpath, exist_ok=True)
                print(runpath)

                # choose the strategies for each lobbyist
                settings['ms'] = generate_ms(settings['n_lobbyists'])
                settings['strategies'] = read_random_strategies(strategies_path, settings['n_lobbyists'])
                params = {**settings, **configparams, 'graph': graph}        
                
                simulator = ALMONDOSimulator(**params).run().save(runpath)
                model, status, _ = simulator.get_results()
                
                fws = [el for el in status[-1]['status'].values()]
                fps = transform(fws, settings)
                
                final_data.append(
                    
                    {
                        'final_weights': fws,
                        'final_probabilities': fps,
                        'final_iterations': int(status[-1]['iteration'])
                    }
                )
                
                oe = OpinionEvolution(model, status)
                oe.plot(filename = runpath + '/probability_evolution.png')
                od = OpinionDistribution(model, status)
                od.plot(filename = runpath + '/probability_distribution.png')
                
            with open(configpath + '/final_data.json', 'w') as f:
                json.dump(final_data, f)
            
            for kind in ['weights', 'probabilities']:    
                
                pars = {
                    'nruns': NRUNS, 
                    'kind': kind,
                    'path': configpath,
                    'graph': graph,
                    'initial_distribution': settings['initial_distribution'],
                    'T': settings['T'],
                    'p_o': settings['p_o'],
                    'p_p': settings['p_p'],
                    'lambdas': configparams['lambdas'],
                    'phis': configparams['phis'],
                    'n_lobbyists': settings['n_lobbyists'],
                    'ms': settings['ms'],
                    'strategies': settings['strategies']
                }    
                
                am = AverageMetrics(**pars).compute()
                metrics = am.get_results()
                with open(configpath+f'/{kind}_metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=4)
            
            
            
            
            
            
            
            
            
            
            
            #  with open(BASE + SCENARIO + CONFIG + RUN + 'final_weights.csv', 'w') as f:
            #         f.write(','.join([str(el) for el in final_weights_run]))
                    

if __name__ == "__main__":
    main("2_lobbyists", 2)