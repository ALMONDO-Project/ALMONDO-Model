import numpy as np

class LobbyistAgent:
    def __init__(self, model, graph, signalmatrix = None, seed=None):
        if seed is not None:
            np.random.seed(seed)        
        
        self.graph = graph
        self.n = self.graph.number_of_nodes()
        self.signalmatrix = signalmatrix
        self.model = model
        self.params = dict()
        #here should be added a definition of the model parameters and when adding parameters it should be checked if the user-defined parameters satisfy these specification (like possible values optional etc)
        
    def __str__(self):
        params = {k: getattr(self, k, None) for k in self.parameters.keys()}
        return f"LobbyistAgent(n={self.n}, parameters={params})"
    
    def add_parameter(self, paramname, paramvalue):
        self.params[paramname] = paramvalue
    
    def set_strategy(self, signalmatrix):
        self.signalmatrix = signalmatrix
        
    def get_strategy(self):
        return self.signalmatrix

    def set_model(self, model):
        self.model = model
    
    def get_model(self):
        return self.model
          
    def get_parameters(self):
        return {param: getattr(self, param) for param in self.params}
    
    def create_signal(self, t):
        if self.signalmatrix is None:
            return np.random.binomial(1, 0.5, self.n)
        else:
            return self.signalmatrix[t]
