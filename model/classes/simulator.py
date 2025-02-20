from .almondoModel import AlmondoModel #questo poi sarà un import da ndlib una volta che il modello sarà caricato lì
import ndlib.models.ModelConfig as mc
from functions.utils import transform
import networkx as nx 
import numpy as np
from tqdm import tqdm
import string
import random
import json
import os


class ALMONDOSimulator(object):
    def __init__(
        self, 
        N: int, 
        initial_distribution: str,
        T: int,
        p_o: float,
        p_p: float,
        lambda_values: float | list, 
        phi_values: float | list,
        base: str = 'results/',
        scenario: str = None,
        nruns: int = 100,
        n_lobbyists: int = 0,
        lobbyists_data: dict = {}
    ):
        
        self.base_path = base
        self.scenario_path = os.path.join(base, scenario)
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.scenario_path, exist_ok=True)
        self.strategies_path = os.path.join(self.scenario_path, "strategies")
        os.makedirs(self.strategies_path, exist_ok=True)        
        
        self.N = N
        self.graph = nx.complete_graph(N)
        self.p_o = p_o
        self.p_p = p_p
        self.lambdas = lambda_values
        self.phis = phi_values
        self.T = T
        self.initial_distribution = initial_distribution
        self.nruns = nruns     
        self.lobbyists_data = lobbyists_data
        self.n_lobbyists =  n_lobbyists    
        
        if self.n_lobbyists > 0:
            self.create_strategies()
             
        print('simulator created')
    
        
    def config_model(self, lambda_v, phi_v):
        
        config = mc.Configuration()
        
        config.add_model_parameter("p_o", self.p_o)
        config.add_model_parameter("p_p", self.p_p)
        
        if isinstance(lambda_v, list):
            for i in self.graph.nodes():
                config.add_node_configuration("lambda", i, lambda_v[i])
        elif isinstance(lambda_v, float):
            for i in self.graph.nodes():
                config.add_node_configuration("lambda", i, lambda_v)
        else:
            raise ValueError

        print(phi_v, type(phi_v))
        if isinstance(phi_v, list):
            for i in self.graph.nodes():
                config.add_node_configuration("phi", i, phi_v[i])
        elif isinstance(phi_v, float):
            for i in self.graph.nodes():
                config.add_node_configuration("phi", i, phi_v)
        else:
            raise ValueError

        print('configuring model: assigning graph, parameters and initial distribution of weights')
        self.model = AlmondoModel(self.graph, seed=1)
        self.model.set_initial_status(config, kind=self.initial_distribution)
        
        if self.n_lobbyists > 0:
            print('assigning random strategies to lobbyists')
            for id in self.lobbyists_data:
                data = self.lobbyists_data[id]
                B = data['B']
                m = data['m']
                matrix, name = self.read_random_strategy(B)          
                print(f'assigning strategy {name} to lobbyist {id}')  
                #add lobbyist with model and strategy
                self.model.add_lobbyist(m, matrix)
                self.lobbyists_data[id]['strategies'].append(name)
            
    def single_run(self, lambda_v, phi_v):
        #Creating a new configuration at each run: changing initial distr, and lobbyists strategies which are random
        self.config_model(lambda_v, phi_v)
        
        #Execution 
        self.system_status = self.model.steady_state(max_iterations=self.T)
        
        #Dump status to json file
        self.save_system_status(self.runpath)
        
        fws = [el for el in self.system_status[-1]['status'].values()]
        fps = transform(fws, self.p_o, self.p_p)
            
        fd = {
                'final_weights': fws,
                'final_probabilities': fps,
                'final_iterations': int(self.system_status[-1]['iteration'])
        }
        
        return self.system_status, fd
    
    def runs(self, lambda_v, phi_v, overwrite=False):
        
        print('Starting montecarlo runs')
        
        runs_data = []
        
        #erase strategies from previous runs which is the only thing that changes about lobbyists across simulations
        for id in range(self.n_lobbyists):
            self.lobbyists_data[id]['strategies'] = []
            
        for run in range(self.nruns):
            
            self.runpath = os.path.join(self.config_path, f'{run}')
            
            if not overwrite:
                if os.path.exists(f'{self.runpath}/status.json'):
                    #run is completely executed, move on
                    print(f'run {run}/{self.nruns} already performed and saved, going to the next one')
                    continue
                else:
                    #run is partially executed or not at all but do not erase existing files 
                    os.makedirs(self.runpath, exist_ok=True)
            else:
                #overwrite existing files if any
                os.makedirs(self.runpath, exist_ok=True)

            _, final_distributions = self.single_run(lambda_v, phi_v)
            
            runs_data.append(final_distributions)
            
        self.save_config()
        
        filename = os.path.join(self.config_path, 'runs_data.json')
        with open(filename, 'w') as f:
            json.dump(runs_data, f)
    
    def execute_experiments(self, overwrite_runs=False):
        
        print('Starting experiments')
        
        for _, (lambda_v, phi_v) in enumerate([(l, p) for l in self.lambdas for p in self.phis]):
            print(f'Starting configuration lambda={lambda_v}, phi={phi_v}')
            
            self.config_path = os.path.join(self.scenario_path, f'{lambda_v}_{phi_v}')            
            os.makedirs(self.config_path, exist_ok=True)
            
            self.runs(lambda_v, phi_v, overwrite=overwrite_runs)

    
    
    
    
    
    
    def save_config(self, filename=None):
        
        import networkx as nx
        
        d = {
            'p_o': self.p_o,
            'p_p': self.p_p,
            'lambda_values': self.lambdas,
            'phi_values': self.phis,
            'T': self.T,
            'n_lobbyists': self.n_lobbyists,
            'lobbyists_data': self.lobbyists_data
        }
        
        d['nruns'] = self.nruns
        
        nx.write_edgelist(self.graph, os.path.join(self.scenario_path, 'graph.csv'), delimiter=",", data=False)

        if filename is None:
            with open(os.path.join(self.scenario_path, 'config.json'), 'w') as f:
                json.dump(d, f, indent=4)      
        else:
            with open(filename, 'w') as f:
                json.dump(d, f, indent=4)

    
    
    
    
    
    def save_system_status(self, path):
        if not hasattr(self, "system_status"):
            raise RuntimeError("You must single_run() before calling save()")
        
        filename = os.path.join(path, 'status.json')
        with open(filename, 'w') as f:
            json.dump(self.system_status, f)    
            
    def create_strategies(self):
        print('creating strategies')
        for id in range(self.n_lobbyists):
            data = self.lobbyists_data[id]
            B = data['B']
            c = data['c']
            folder = os.path.join(self.strategies_path, str(B))
            print(folder)
            os.makedirs(folder, exist_ok=True)
            for run in range(self.nruns):
                inter_per_time = B // (c * 3000)
                matrix = np.zeros((3000, self.N), dtype=int)
                for t in range(3000):
                    indices = np.random.choice(self.N, inter_per_time, replace=False)
                    matrix[t, indices] = 1
                print('saving strategy to file')
                filename = f'strategy_{run}.txt'
                path = os.path.join(folder, filename)
                np.savetxt(path, matrix, fmt="%i")
        print('strategies created')             
        
    def read_random_strategy(self, B):
        path = os.path.join(self.strategies_path, str(B))
        strategy_name = random.choice(os.listdir(path))
        filepath = os.path.join(path, strategy_name)
        print(f"Reading {filepath}")
        return np.loadtxt(filepath).astype(int), filepath    

    def get_results(self):
        if not hasattr(self, "system_status"):
            raise RuntimeError("No results available. Did you call run()?")
        return self.model, self.system_status, self.model.lobbyists





















    # def saveConfig(self, lambda_v, phi_v, filename="config.json"):
    #     path = os.path.join(self.config_path, "config_data/")
    #     os.makedirs(path, exist_ok=True)
                
    #     confdict = {
    #         "initial_distribution": self.initial_distribution,
    #         "T": self.T,
    #         "p_o": self.p_o,
    #         "p_p": self.p_p,
    #         "lambda": lambda_v,
    #         "phi": phi_v,
    #         "base_path": self.base_path,
    #         "scenario_path": self.scenario_path,
    #         "config_path": self.config_path,
    #     }
        
    #     lobbyists = self.model.get_lobbyists(strategy=False) #se non ci sono lobbyists è None

    #     confdict['lobbyists_data'] = lobbyists
        
    #     filename = os.path.join(path, filename)
    #     with open(filename, "w") as f:
    #         json.dump(confdict, f, indent=4)
        
    #     graph_file = os.path.join(path, "graph.csv")
    #     nx.write_edgelist(self.graph, graph_file, delimiter=",", data=False)
        
        
    # @classmethod
    # def load_config(cls, filename, graph_filename):
    #     """Load a configuration and graph from files and return an instance of ALMONDOSimulator."""
    #     with open(filename, "r") as f:
    #         config = json.load(f)
    #     # Load the graph
    #     graph = nx.read_edgelist(graph_filename, delimiter=",", data=False)
        
    #     return cls(graph=graph, **config)
                    
        
        
        