from .almondoModel import AlmondoModel #questo poi sarà un import da ndlib una volta che il modello sarà caricato lì
import ndlib.models.ModelConfig as mc
import json

class ALMONDOSimulator(object):
    def __init__(
        self, 
        path: str,
        
        graph: object, 
        
        initial_distribution: str,
        T: int,
        
        p_o: float,
        p_p: float,
        
        lambdas: float | list, 
        phis: float | list,
        
        n_lobbyists: int,
        ms: list,
        strategies: list

    ):
        
        self.graph = graph
        self.p_o = p_o
        self.p_p = p_p
        self.lambdas = lambdas
        self.phis = phis
        self.T = T
        self.initial_distribution = initial_distribution
        self.path = path
        self.n_lobbyists = n_lobbyists
        self.ms = ms
        self.strategies = strategies
        assert len(self.ms) == len(self.strategies), "Lengths of ms and strategies must be the same!"
    
    def run(self):
        config = self.createConfig()
        self.model = AlmondoModel(self.graph, seed=1)
        self.model.set_initial_status(config, kind=self.initial_distribution)
        
        for m, strategy in zip(self.ms, self.strategies):
            self.model.add_lobbyist(m, strategy)

        self.system_status = self.model.steady_state(max_iterations=self.T)

        return self  # Allows method chaining

    def save(self, path):
        if not hasattr(self, "system_status"):
            raise RuntimeError("You must run() before calling save().")
        
        with open(path + '/status.json', 'w') as f:
            json.dump(self.system_status, f)

        return self  # Allows further chaining

    def get_results(self):
        if not hasattr(self, "system_status"):
            raise RuntimeError("No results available. Did you call run()?")

        return self.model, self.system_status, self.model.lobbyists
        
        
    def createConfig(self):
        
        config = mc.Configuration()
        
        
        config.add_model_parameter("p_o", self.p_o)
        config.add_model_parameter("p_p", self.p_p)
        
        
        if isinstance(self.lambdas, list):
            for i in self.graph.nodes():
                config.add_node_configuration("lambda", i, self.lambdas[i])
        elif isinstance(self.lambdas, float):
            for i in self.graph.nodes():
                config.add_node_configuration("lambda", i, self.lambdas)
        else:
            raise ValueError("lambdas must be either a float or a list")


        if isinstance(self.lambdas, list):
            for i in self.graph.nodes():
                config.add_node_configuration("phi", i, self.phis[i])
        elif isinstance(self.lambdas, float):
            for i in self.graph.nodes():
                config.add_node_configuration("phi", i, self.phis)
        else:
            raise ValueError("phis must be either a float or a list")
        return config    
            
                    
        
        
        