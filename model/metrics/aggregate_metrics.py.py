import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from metrics import nclusters, pwdist, lobbyist_performance

nl = 0
basepath = f'../results/unbalanced_budgets/'
filename = 'config.json'
with open(os.path.join(basepath, filename), 'r') as f:
    params = json.load(f)

print(params)

p_o = params['p_o']
p_p = params['p_p']
lambda_values = params['lambda_values']
phi_values = params['phi_values']
n_lobbyists = params['n_lobbyists']
nruns = params['nruns']
lobbyists_data = params['lobbyists_data']


def get_data(trends: dict, 
            p_o: float,
            p_p: float,
            iteration: Union[int, str] = -1, 
            kind: str = "probabilities"):
       
        """
        Args:
            trends (dict): The computed simulation trends.
            p_o (float): Probability of the optimistic model.
            p_p (float): Probability of the pessimistic model.
            iteration (int | str): The iteration number or "last" for final state (default: -1).
            values (str): The type of values to extract ("probabilities" or "weights").
        """
        
        if isinstance(iteration, int) and -1 <= iteration < len(trends):
            it = trends[iteration]['iteration']
            ops = np.array(list(trends[iteration]['status'].values()), dtype=float)
        else:
            raise ValueError(f"Invalid iteration index: {iteration}")

        # Compute values based on type
        if kind == 'probabilities':
            ops = p_o * ops + p_p * (1 - ops)
            ops = np.array(ops, dtype=float)
        elif kind == 'weights':
            np.array(ops, dtype=float)
        else:
            raise ValueError("`values` must be either 'probabilities' or 'weights'.")
        
        return ops, it

import json
import os
import numpy as np
from tqdm.notebook import tqdm  # Use tqdm for Jupyter Notebook

kinds = ['weights', 'probabilities']

# Total iterations for tqdm
total_iterations = len(kinds) * len(params['lambda_values']) * len(params['phi_values']) * params['nruns']

with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
    for kind in kinds:
        for _, (lambda_v, phi_v) in enumerate([(l, p) for l in params['lambda_values'] for p in params['phi_values']]):    
            path = os.path.join(basepath, f'{lambda_v}_{phi_v}/')  
            metrics = {
                'effective_number_clusters': [],
                'number_iterations': [],
                'average_pairwise_distance': [],
                'average_opinions': [],
                'std_opinions': [],
                'lobbyists_performance': {k: [] for k in range(n_lobbyists)}
            }
            
            avg_metrics = {
                'effective_number_clusters': {'avg': -1, 'std': -1},
                'number_iterations': {'avg': -1, 'std': -1},
                'average_pairwise_distance': {'avg': -1, 'std': -1},
                'average_opinions': {'avg': -1, 'std': -1},
                'std_opinions': {'avg': -1, 'std': -1},
                'lobbyists_performance': {k: {'avg': -1, 'std': -1} for k in range(n_lobbyists)}
            }

            for run in range(params['nruns']):
                runpath = os.path.join(path, str(run))
                
                with open(runpath+'/status.json', 'r') as f:
                    trends = json.load(f)
                
                ops, it = get_data(trends, p_o, p_p, kind=kind)
                
                metrics['effective_number_clusters'].append(nclusters(ops, 0.0001))
                metrics['number_iterations'].append(it)
                metrics['average_pairwise_distance'].append(pwdist(ops))
                metrics['average_opinions'].append(np.array(ops).mean())
                metrics['std_opinions'].append(np.array(ops).std())

                for id, lob in lobbyists_data.items():
                    metrics['lobbyists_performance'][int(id)].append(lobbyist_performance(ops, lob['m'], p_o, p_p))
                    
                for k, v in metrics.items():
                    if k != 'lobbyists_performance':
                        avg = np.array(v).mean()
                        std = np.array(v).std()
                        avg_metrics[k]['avg'] = avg
                        avg_metrics[k]['std'] = std
                    else:
                        for id in range(n_lobbyists):
                            avg = np.array(v[id]).mean()
                            std = np.array(v[id]).std()
                            avg_metrics[k][id]['avg'] = avg
                            avg_metrics[k][id]['std'] = std
                pbar.update(1)  

            with open(path+f'{kind}_metrics_distributions.json', 'w') as f:
                json.dump(metrics, f)
            
            with open(path+f'{kind}_average_metrics.json', 'w') as f:
                json.dump(avg_metrics, f)