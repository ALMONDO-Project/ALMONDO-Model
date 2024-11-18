import numpy as np

class LobbyistAgent:
    def __init__(self, model=1, signalmatrix=None, seed=None):
        if seed is not None:
            np.random.seed(seed)        
        self.signalmatrix = signalmatrix
        self.model = model
        #outreach probabilities è un parametro
        #budget è un parametro
        #entrambi i parametri sono opzionali
        #li devo poter *settare*
        #strategy è un parametro anch'esso da settare anch'esso opzionale
        #la devo poter *settare* 
        #dare la possibilità di salvare e uplodare un file di configurazione x i lobbisti
        
        

    def __str__(self):
        params = {k: getattr(self, k, None) for k in self.params.keys()}
        return f"LobbyistAgent(parameters={params})"
    
    def add_parameter(self, paramname, paramvalue):
        self.params[paramname] = paramvalue

    def set_strategy(self, signalmatrix=None, budget=None, probabilities=None):
        """
        Sets the strategy as a signal matrix. If budget and probabilities are provided,
        generates the matrix; otherwise, reads the passed signalmatrix.
        
        Parameters:
        - signalmatrix (np.ndarray): Directly provided signal matrix.
        - budget (float): Total budget for interactions.
        - probabilities (np.ndarray): Probability vector/matrix for interacting with each node.
        """
        if budget is not None and probabilities is not None:
            # Ensure probabilities is a numpy array for matrix operations
            probabilities = np.array(probabilities)
            # Scale probabilities by budget to create the signal matrix
            self.signalmatrix = budget * probabilities
        elif signalmatrix is not None:
            # Use provided signal matrix directly
            self.signalmatrix = signalmatrix
        else:
            raise ValueError("Provide either a signalmatrix or both budget and probabilities.")



    def set_model(self, model):
        self.model = model
    

    
    ########## GET FUNCTIONS ##############
    def get_strategy(self, t):
        """Retrieve the strategy for a specific time step `t`."""
        if self.signalmatrix is not None and t < len(self.signalmatrix):
            return self.signalmatrix[t]
        else:
            raise IndexError("Signal matrix is undefined or time step `t` is out of bounds.")
    def get_model(self):
        return self.model
          
    def get_parameters(self):
        return {param: getattr(self, param) for param in self.params}


###### FUNZIONI VISUALIZZAZIONE #######