import os
import json
import numpy as np
from typing import Union
from functions.metrics_functions import nclusters, pwdist, lobbyist_performance
import json
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import norm  # Import norm for the Z-test p-value calculation

class Metrics(object):
    def __init__(
        self,
        nl: int = 0, 
        basepath: str = f'../results/balanced_budgets/', 
        filename: str = 'config.json',
    ):
        self.nl = nl
        self.basepath = basepath
        self.filename = filename
        
        with open(os.path.join(basepath, filename), 'r') as f:
            params = json.load(f)
        
        self.p_o = params['p_o']
        self.p_p = params['p_p']
        self.lambda_values = params['lambda_values']
        self.phi_values = params['phi_values']
        self.n_lobbyists = params['n_lobbyists']
        self.nruns = params['nruns']
        self.lobbyists_data = params['lobbyists_data']
        self.params = params
        
    def get_data(
            self,
            trends: dict, 
            kind: str,
            iteration: Union[int, str] = -1):
       
        """
        Args:
            trends (dict): The computed simulation trends.
            p_o (float): Probability of the optimistic model.
            p_p (float): Probability of the pessimistic model.
            iteration (int | str): The iteration number or "last" for final state (default: -1).
            kind (str): The type of values to extract ("probabilities" or "weights").
        """
        
        if isinstance(iteration, int) and -1 <= iteration < len(trends):
            it = trends[iteration]['iteration']
            ops = np.array(list(trends[iteration]['status'].values()), dtype=float)
        else:
            raise ValueError(f"Invalid iteration index: {iteration}")

        # Compute values based on type
        if kind == 'probabilities':
            ops = self.p_o * ops + self.p_p * (1 - ops)
            ops = np.array(ops, dtype=float)
        elif kind == 'weights':
            np.array(ops, dtype=float)
        else:
            raise ValueError("`values` must be either 'probabilities' or 'weights'.")
        
        return ops, it

    def compute_metrics(self, 
                        kind:str):
        # Total iterations for tqdm
        total_iterations = len(self.lambda_values) * len(self.phi_values) * self.nruns
        nruns = self.nruns

        with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
            for _, (lambda_v, phi_v) in enumerate([(l, p) for l in self.lambda_values for p in self.phi_values]):    
                path = os.path.join(self.basepath, f'{lambda_v}_{phi_v}/')  
                
                metrics = {
                    'effective_number_clusters': [],
                    'number_iterations': [],
                    'average_pairwise_distance': [],
                    'average_opinions': [],
                    'std_opinions': [],
                    'lobbyists_performance': {k: [] for k in range(self.n_lobbyists)}
                }
                
                avg_metrics = {
                    'effective_number_clusters': {'avg': -1, 'std': -1},
                    'number_iterations': {'avg': -1, 'std': -1},
                    'average_pairwise_distance': {'avg': -1, 'std': -1},
                    'average_opinions': {'avg': -1, 'std': -1,'z_statistic': -1, 'p_value': -1},
                    'std_opinions': {'avg': -1, 'std': -1},
                    'lobbyists_performance': {k: {'avg': -1, 'std': -1} for k in range(self.n_lobbyists)}
                }
                
                if not os.path.exists(path+f'{kind}_metrics_distributions.json') and not os.path.exists(path+f'{kind}_average_metrics.json'):
                    for run in range(self.nruns):
                        runpath = os.path.join(path, str(run))
                        
                        try:
                            with open(runpath+'/status.json', 'r') as f:
                                trends = json.load(f)
                        except json.decoder.JSONDecodeError as e:
                            print(f'Error reading {runpath}/status.json: {e}')
                            continue
                        
                        ops, it = self.get_data(trends, kind=kind)
                        ops_array = np.array(ops)
                        
                        metrics['effective_number_clusters'].append(nclusters(ops, 0.0001))
                        metrics['number_iterations'].append(it)
                        metrics['average_pairwise_distance'].append(pwdist(ops))
                        metrics['average_opinions'].append(ops_array.mean())
                        metrics['std_opinions'].append(ops_array.std())

                        for id, lob in self.lobbyists_data.items():
                            metrics['lobbyists_performance'][int(id)].append(lobbyist_performance(ops, lob['m'], self.p_o, self.p_p))
                            
                        pbar.update(1)

                    # Compute average metrics
                    for k, v in metrics.items():
                        if k != 'lobbyists_performance':
                            avg = np.array(v).mean()
                            std = np.array(v).std()
                            avg_metrics[k]['avg'] = avg
                            avg_metrics[k]['std'] = std
                        else:
                            for id in range(self.n_lobbyists):
                                avg = np.array(v[id]).mean()
                                std = np.array(v[id]).std()
                                avg_metrics[k][id]['avg'] = avg
                                avg_metrics[k][id]['std'] = std

                    # Z-test for proportions calculation
                    p_hat = avg_metrics['average_opinions']['avg']  #average opinion across runs
                    n =self.nruns  # Sample size
                    p_0 = 0.5  # Hypothesized population proportion

                    # Z-test for proportion
                    z = (p_hat - p_0) / np.sqrt((p_0 * (1 - p_0)) / n)
                    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed p-value

                    # Add Z-statistic and p-value to avg_metrics
                    avg_metrics['average_opinions']['z_statistic'] = z
                    avg_metrics['average_opinions']['p_value'] = p_value
                    
                    # Write results to file
                    with open(path+f'{kind}_metrics_distributions.json', 'w') as f:
                       json.dump(metrics, f)

                    with open(path+f'{kind}_average_metrics.json', 'w') as f:
                        json.dump(avg_metrics, f)