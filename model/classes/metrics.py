from functions.metrics import nclusters, pwdist, lobbyist_performance
from typing import Literal
import numpy as np
import json
import os

class AverageMetrics(object):
    
    def __init__(self,
                nruns: int, 
                kind: Literal["weights", "probabilities"], 
                
                p_o: float,
                p_p: float,
                
                n_lobbyists: int = 0,
                ms: list = None,
                
                data: dict = None,
                #path: str = None,
                ):
        
        self.seed = 1
        #self.path = path
        
        self.nruns = nruns
        self.kind = kind
        
        self.p_o = p_o
        self.p_p = p_p

        self.n_lobbyists = n_lobbyists
        self.ms = ms

        
        def _convert(data, kind):
            
            """
            :Args:data (list): List of dictionary of final statuses and final iteration for the nruns
            :Args:kind (str): "Probabilities" or "Weights" indicating on which value to compute the metrics
            """
            
            newdata = []
            newits = []
            if data is not None:
                for run, d in enumerate(data):
                    print(run)
                    if kind == 'weights':
                        ops = d.get('final_weights', None)
                    elif kind == 'probabilities':
                        ops = d.get('final_probabilities', None)
                    else:
                        raise ValueError(f"Invalid kind: {kind}. Expected 'weights' or 'probabilities'.")
                    if ops is None:
                        raise KeyError(f"Missing key '{kind}' in run {run}.")
                    if not isinstance(ops, list):
                        ops = list(ops) if hasattr(ops, '__iter__') else None  # Convert iterables to lists
                    if ops is None:
                        raise TypeError(f"Could not convert '{kind}' data into a list for run {run}.")
                    try:
                        ops = [float(x) for x in ops]
                    except (ValueError, TypeError):
                        raise TypeError(f"All elements in ops must be convertible to float, but got {ops} for run {run}.")
                    
                    try:
                        its = int(d.get('final_iterations', None))
                    except(ValueError, TypeError):
                        raise TypeError(f"Iterations must be convertible to integers but got {its} for run {run}")
                    
                    newits.append(its)
                    newdata.append(ops)
                    
                return newdata, newits
            
            else:
                
                return None, None
        
        
        self.data, self.iterations = _convert(data, kind) #list of list and list of integers
        
        
        self.metrics = {
            'effective_number_clusters': {'avg': 0, 'std': 0},
            'number_iterations': {'avg': 0, 'std': 0},
            'average_pairwise_distance': {'avg': 0, 'std': 0},
            'mean_opinion_distrib': {'avg': 0, 'std': 0},
            'variance_opinion_distrib': {'avg': 0, 'std': 0},
            'lobbyists_performance': {k: dict() for k in range(n_lobbyists)}
        }
        
        for l_id in range(n_lobbyists):
            self.metrics['lobbyists_performance'][l_id] = {'avg': 0, 'std': 0}
        
        
    def compute(self, threshold: float = 0.0001):
        print('computing average metrics')
        self.avg_enc()
        self.avg_nit()
        self.avg_pwdist()
        self.avg_meanprob()
        self.avg_varprob()
        self.avg_lobbyist_performance()
        print('done')
        return self  
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.metrics, f)
       
    def avg_enc(self, threshold: float = 0.2): #threshold scelta più o meno a caso?
        ncs = []
        for _, opinions in enumerate(self.data):
            nc = nclusters(opinions=opinions, threshold=threshold) #effective number of clusters in run
            ncs.append(nc)
        ncs = np.array(ncs)
        avg = np.mean(ncs)
        std = np.std(ncs)
        
        self.metrics['effective_number_clusters']['avg'] = avg
        self.metrics['effective_number_clusters']['std'] = std

    def avg_nit(self):
        its = np.array(self.iterations)
        avg = np.mean(its)
        std = np.std(its)
        
        self.metrics['number_iterations']['avg'] = avg
        self.metrics['number_iterations']['std'] = std
    
    def avg_pwdist(self):
        pwds = []
        for _, opinions in enumerate(self.data):
            pwd = pwdist(opinions=opinions) #effective number of clusters in run
            pwds.append(pwd)
        pwds = np.array(pwds)
        avg = np.mean(pwds)
        std = np.std(pwds)
        
        self.metrics['average_pairwise_distance']['avg'] = avg
        self.metrics['average_pairwise_distance']['std'] = std
    
    def avg_meanprob(self):
        means = []
        for opinions in self.data:
            mean_prob = np.mean(opinions)  # Mean opinion for each run
            means.append(mean_prob)
        means = np.array(means)
        avg = np.mean(means)
        std = np.std(means)
        
        self.metrics['mean_opinion_distrib']['avg'] = avg
        self.metrics['mean_opinion_distrib']['std'] = std
    
    def avg_varprob(self):
        variances = []
        for opinions in self.data:
            var_prob = np.var(opinions)  # Variance of opinion for each run
            variances.append(var_prob)
        variances = np.array(variances)
        avg = np.mean(variances)
        std = np.std(variances)
        
        self.metrics['variance_opinion_distrib']['avg'] = avg
        self.metrics['variance_opinion_distrib']['std'] = std
    
    def avg_lobbyist_performance(self):
        
        if self.n_lobbyists > 0:
        
            for l_id in range(self.n_lobbyists):
                
                entropies = []
                
                for opinions in self.data:
                    
                    entropy = lobbyist_performance(opinions, self.ms[l_id], self.p_o, self.p_p)
                    entropies.append(entropy)
                    
                entropies = np.array(entropies)
                
                avg = np.mean(entropies)
                std = np.std(entropies)
                
                self.metrics['lobbyists_performance'][l_id]['avg'] = avg
                self.metrics['lobbyists_performance'][l_id]['std'] = std
        
        else:
            
            self.metrics['lobbyists_performance'] = None
            self.metrics['lobbyists_performance'] = None

            
        
    
    
    
    
    
    
    
    
    
    
    
    def load(self, file_path: str):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.metrics = json.load(f)
        else:
            raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    def get_results(self):
        return self.metrics
    
             
    
    
    
    
    
    