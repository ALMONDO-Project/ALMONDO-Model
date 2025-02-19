from .almondoModel import AlmondoModel #questo poi sarà un import da ndlib una volta che il modello sarà caricato lì
import ndlib.models.ModelConfig as mc
from functions.strategies import read_random_strategy, generate_strategies
from functions.utils import transform
import networkx as nx 
import json
import os


class ALMONDOSimulator(object):
    def __init__(
        self, 
        N, 
        initial_distribution: str,
        T: int,
        p_o: float,
        p_p: float,
        lambda_values: float | list, 
        phi_values: float | list,
        n_lobbyists: int,
        base_path: str = 'simulations/',
        scenario_path: str = None,
        nruns: int = 100
    ):
        
        self.base_path = base_path
        if scenario_path is None:
            self.scenario_path = os.path.join(base_path, f'{n_lobbyists}_lobbyists')
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
        self.n_lobbyists = n_lobbyists      
        self.nruns = nruns     
        
        generate_strategies(self.strategies_path, self.nruns, self.graph.number_of_nodes())
                
        print('simulator created')

    def createConfig(self, lambda_v, phi_v):
        
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

        return config    
    
    def add_lobbyists(self):  
        
        self.lobbyists_data = {}      
        
        for id in range(self.n_lobbyists):
            
            m = id%2
            strategy, name = read_random_strategy(self.strategies_path)            
            self.model.add_lobbyist(m, strategy)

            self.lobbyists_data[id] = {}
            self.lobbyists_data[id]['m'] = m
            self.lobbyists_data[id]['strategy'] = name
        
    def single_run(self, lambda_v, phi_v):
        
        #config part
        config = self.createConfig(lambda_v, phi_v)
        self.model = AlmondoModel(self.graph, seed=1)
        self.model.set_initial_status(config, kind=self.initial_distribution)
        
        #Lobbyists 
        self.add_lobbyists()

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
    
    def runs(self, lambda_v, phi_v):
        runs_data = []
        
        for run in range(self.nruns):
            
            self.runpath = os.path.join(self.config_path, f'{run}')
            
            if os.path.exists(f'{self.runpath}/status.json'):
                print(f'run {run}/{self.nruns} already performed and saved, going to the next one')
                continue
            else:
                os.makedirs(self.runpath, exist_ok=True)
            
            status, fd = self.single_run(lambda_v, phi_v)
            
            runs_data.append(fd)
            
            self.save_config(filename = f'{self.runpath}/config.json')
        
        with open(self.config_path+'/runs_data.json', 'w') as f:
            json.dump(runs_data, f)
    
    def execute_experiments(self):
    
        for _, (lambda_v, phi_v) in enumerate([(l, p) for l in self.lambdas for p in self.phis]):
            print(f'starting configuration lambda={lambda_v}, phi={phi_v}')
            
            self.config_path = os.path.join(self.scenario_path, f'{lambda_v}_{phi_v}/')            
            os.makedirs(self.config_path, exist_ok=True)
            
            self.runs(lambda_v, phi_v)
        
        return self
        

    def get_results(self):
        if not hasattr(self, "system_status"):
            raise RuntimeError("No results available. Did you call run()?")
        return self.model, self.system_status, self.model.lobbyists

    def save_config(self, filename=None):
        
        import networkx as nx
        
        d = {
            'p_o': self.p_o,
            'p_p': self.p_p,
            'lambda_values': self.lambdas,
            'phi_values': self.phis,
            'T': self.T,
            'n_lobbyists': self.n_lobbyists,
            'lobbyists': self.lobbyists_data
        }
        
        d['nruns'] = self.nruns
        
        nx.write_edgelist(self.graph, self.scenario_path+'graph.csv', delimiter=",", data=False)

        if filename is None:
            for id in d['lobbyists']:
                d['lobbyists'][id]['strategy'] = ''
            with open(self.scenario_path+'config.json', 'w') as f:
                json.dump(d, f)      
        else:
            with open(filename, 'w') as f:
                json.dump(d, f)

    def save_system_status(self, path):
        if not hasattr(self, "system_status"):
            raise RuntimeError("You must single_run() before calling save()")
        
        filename = os.path.join(path, 'status.json')
        with open(filename, 'w') as f:
            json.dump(self.system_status, f)        
        
        






















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
                    
        
        
        