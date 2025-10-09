from ndlib.models.DiffusionModel import DiffusionModel
from tqdm import tqdm
import numpy as np

# Author information
__author__ = ["Alina Sirbu", "Giulio Rossetti", "Valentina Pansanella"]
__email__ = [
    "alina.sirbu@unipi.it",
    "giulio.rossetti@isti.cnr.it",
    "valentina.pansanella@isti.cnr.it",
]


class AlmondoModel(DiffusionModel):
    """
    A class to model diffusion in a network, with additional functionality 
    for lobbying agents influencing node states. Extends DiffusionModel from ndlib.
    """
    # Small epsilon to prevent division by zero and extreme values
    EPSILON = 1e-15

    class LobbyistAgent:
        """
        A class to represent a lobbyist agent influencing nodes in the network.

        Attributes:
            m: Model of the lobbyist (0 for pessimistic, 1 for optimistic).
            strategy: A matrix representing the strategy for node influence over time.
        """

        def __init__(self, m: int, strategy: np.ndarray):
            """
            Initialize a LobbyistAgent.
            
            Arguments:
                m (int): Model type, 0 for pessimistic, 1 for optimistic.
                strategy (np.ndarray): The strategy matrix (T x N) for the agent's influence.
            """
            if not isinstance(m, int):
                raise TypeError("Lobbyist model must be an integer.")
            if m != 0 and m != 1:
                raise ValueError("Lobbyist's model 'm' must be an integer 0 (for pessimistic) or 1 (for optimistic).")
            self.m = m  # 0 (pessimistic) or 1 (optimistic)

            if not isinstance(strategy, np.ndarray):
                raise TypeError("Lobbyist strategy must be a numpy array.")
            
            if not np.all(np.isin(strategy, [0, 1])):
                raise ValueError(f"Strategy matrix for lobbyist contains values other than 0 and 1.")
            
            self.strategy = strategy  # Strategy matrix
            self.max_t, self.N = self.strategy.shape

    def __init__(self, graph, seed: int = None, verbose: bool = True):
        """
        Initialize the Almondo diffusion model.

        Arguments:
            graph (networkx.Graph): The graph representing the network of nodes.
            seed (int, optional): Random seed for reproducibility.
            verbose (bool, optional): Whether to print debug information.
        """
        super().__init__(graph, seed)
        self.discrete_state = False  # Use continuous state model
        self.parameters = {
            "model": {
                "p_o": {  # Probability of optimistic events
                    "descr": "Probability of event optimist model",
                    "range": [0, 1],
                    "optional": False
                },
                "p_p": {  # Probability of pessimistic events
                    "descr": "Probability of event pessimist model",
                    "range": [0, 1],
                    "optional": False
                }
            },
            "nodes": {
                "lambda": {  # Node-specific parameter lambda
                    "descr": "...",
                    "range": [0, 1],
                    "optional": False
                },
                "phi": {  # Node-specific parameter phi
                    "descr": "...",
                    "range": [0, 1],
                    "optional": False
                }
            },
            "edges": {}  # Empty, no edge-specific parameters
        }
        self.name = "Almondo"
        self.n = self.graph.number_of_nodes()  # Number of nodes in the graph
        self.seed = seed
        self.T = None
        self.status = None  # Initial status, will be assigned later
        self.verbose = verbose  # Verbose output for debugging
        
    def _print(self, *args, **kwargs):
        """Custom print method that respects verbose setting"""
        if self.verbose:
            print(*args, **kwargs)

    def set_initial_status(
        self, 
        configuration=None, 
        kind: str = 'uniform',
        status: list = None
    ) -> None:
        
        """
        Sets the initial status of all nodes in the network.

        Arguments:
            configuration (optional): Configuration for setting initial status.
            kind (str, optional): Type of distribution for status ('uniform' by default).
            uniform_range (list, optional): Range for the uniform distribution.
            unbiased_value (float, optional): Value to initialize nodes to if no status is provided.
            status (optional): Predefined status values for nodes.
            gaussian_params (optional): Parameters for Gaussian initialization (not implemented).
            initial_lambda (float, optional): Initial value for the lambda parameter of nodes.

        Returns:
            None
        """
        
        super().set_initial_status(configuration)
        
        if kind == 'uniform':
            self.status = np.random.uniform(low=0, high=1, size=self.n)  # Random uniform status for each node
        elif kind == 'custom':
            if len(status) != self.n:
                raise ValueError(f"The length of the status list {len(status)} does not match the number of nodes {self.n}.")
            self.status = np.array(status)
            if not all(0 <= val <= 1 for val in self.status):
                raise ValueError("All initial status values must be between 0 and 1.")
        else:
            raise ValueError("Other types of initial distributions are not implemented yet.")
        
        # Set node-specific parameters (lambda and phi)
        if not all(0 <= val <= 1 for val in self.params['nodes']['lambda'].values()):
                raise ValueError("All initial lambdas values must be between 0 and 1.")
        if not all(0 <= val <= 1 for val in self.params['nodes']['phi'].values()):
                raise ValueError("All initial phis values must be between 0 and 1.")
        
        self.lambdas = np.array([el for el in self.params['nodes']['lambda'].values()])
        self.phis = np.array([el for el in self.params['nodes']['phi'].values()])

        self.lobbyists = []  # Empty list of lobbyists initially
        self.system_status = []  # Empty list to store system status over time

    def add_lobbyist(self, m: int, strategy: np.ndarray) -> None:
        """
        Adds a lobbyist agent to the model.

        Arguments:
            m (int): Model type of the lobbyist (0 for pessimistic, 1 for optimistic).
            strategy (np.ndarray): Strategy matrix for the lobbyist.

        Returns:
            None
        """
        if isinstance(strategy, np.ndarray) and strategy.size > 0:
            if np.shape(strategy)[1] != self.graph.number_of_nodes():
                raise ValueError(f"Strategy matrix for lobbyist does not match the number of agents in the graph. Expected {self.graph.number_of_nodes()} columns, got {np.shape(strategy)[1]}.")
        new_lobbyist = self.LobbyistAgent(m, strategy)
        self.lobbyists.append(new_lobbyist)

    def get_lobbyists(self, strategy: bool = False) -> list:
        
        """
        Retrieves the list of lobbyists.

        Arguments:
            strategy (bool, optional): Whether to include the strategy matrix in the return data.

        Returns:
            list: A list of dictionaries with lobbyist details.
        """
        
        ls = []
        if len(self.lobbyists) > 0:
            if strategy:
                for i, l in enumerate(self.lobbyists):
                    ls.append({
                        'id': i,
                        'model': l.m,
                        'strategy': l.strategy
                    })
            else:
                for i, l in enumerate(self.lobbyists):
                    ls.append({
                        'id': i,
                        'model': l.m
                    })
            return ls
        else:
            return None

    def generate_lambda(self, w: np.ndarray, s: float, phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
       
        """
        Generates a lambda value for each node based on the current status and parameters.

        Arguments:
            w (np.ndarray): The current status of nodes.
            s (float): Strategy value for influence.
            phi (np.ndarray): The phi parameter for nodes.
            lam (np.ndarray): The lambda parameter for nodes.

        Returns:
            np.ndarray: The calculated lambda values for the nodes.
        """
       
        f = np.abs((1 - s) - w)  # Difference between current state and strategy (opt model)
        # f = np.abs(s - w)  # Difference between current state and strategy (pess model)
        lambdas = phi * f + (1 - phi) * lam

        # Ensure lambda stays in valid range
        lambdas = np.clip(lambdas, 0, 1)
                
        return lambdas  # Weighted influence

    def update(self, receivers: np.ndarray, s: float) -> np.ndarray:
        
        """
        Updates the status of receiver nodes based on a signal and their current status.
        Added numerical stability with EPSILON to avoid division by zero.

        Arguments:
            receivers (np.ndarray): List of nodes to receive the signal.
            s (float): The signal to propagate to receivers.

        Returns:
            np.ndarray: Updated statuses for the receiver nodes.
        """
       
        w = self.actual_status[receivers]  # Current status of the receiver nodes
        
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']

        # Clip probabilities to avoid extreme values (0 or 1) - numerical stability: it should not happen but just in case
        p_o = np.clip(p_o, self.EPSILON, 1 - self.EPSILON)
        p_p = np.clip(p_p, self.EPSILON, 1 - self.EPSILON)
        
        p = w * p_o + (1 - w) * p_p  # Combined probability based on current node's status (opt model)
        # p = (1-w) * p_o + w * p_p  # Combined probability based on current node's status (pess model)
        p = np.clip(p, self.EPSILON, 1 - self.EPSILON) # clipping to avoid extreme values
        
        phi = self.phis[receivers]
        lam = self.lambdas[receivers]

        # Generate lambda with bounds checking
        l = self.generate_lambda(self.actual_status[receivers], s, phi, lam)
        l = np.clip(l, 0, 1)  # Ensure lambda is in [0, 1]

        # Calculate update components with numerical stability
        # Using log-space for very small probabilities could be an option,
        # but for now we use careful clipping

        # For w1 calculation
        ratio_o_p = p_o / p
        ratio_not_o_not_p = (1 - p_o) / (1 - p)
        
        # Clip ratios to prevent overflow
        ratio_o_p = np.clip(ratio_o_p, self.EPSILON, 1/self.EPSILON)
        ratio_not_o_not_p = np.clip(ratio_not_o_not_p, self.EPSILON, 1/self.EPSILON)

        # opt model
        w1 = l * w + (1 - l) * w * (s * ratio_o_p + (1 - s) * ratio_not_o_not_p)

        # For w2 calculation
        ratio_p_p = p_p / p
        ratio_not_p_not_p = (1 - p_p) / (1 - p)
        
        # Clip ratios to prevent overflow
        ratio_p_p = np.clip(ratio_p_p, self.EPSILON, 1/self.EPSILON)
        ratio_not_p_not_p = np.clip(ratio_not_p_not_p, self.EPSILON, 1/self.EPSILON)


        w2 = l * (1 - w) + (1 - l) * (1 - w) * (s * ratio_p_p + (1 - s) * ratio_not_p_not_p)

        # pess model
        # w1 = l * (1-w) + (1 - l) * (1-w) * (s * ratio_o_p + (1 - s) * ratio_not_o_not_p)

        # w2 = l * w + (1 - l) * w * (s * ratio_p_p + (1 - s) * ratio_not_p_not_p)

        # Ensure denominator is not too small
        denominator = w1 + w2
        
        # Check for numerical issues
        if np.any(denominator < self.EPSILON):
            self._print(f"WARNING: Near-zero denominator in update()")
            self._print(f"min denominator: {np.min(denominator)}")
            self._print(f"Affected nodes: {np.sum(denominator < self.EPSILON)}")
            denominator = np.clip(denominator, self.EPSILON, None)
        
        # Check for NaN before division
        if np.any(np.isnan(w1)) or np.any(np.isnan(w2)):
            self._print(f"ERROR: NaN detected before division in update()")
            self._print(f"w1 NaN count: {np.sum(np.isnan(w1))}")
            self._print(f"w2 NaN count: {np.sum(np.isnan(w2))}")
            raise ValueError("NaN values detected in update calculation")

        updated_w = w1 / denominator # opt model
        # updated_w = w2 / denominator # pess model
        
        # Final clipping to ensure valid probability range
        updated_w = np.clip(updated_w, 0, 1)
        
        # Check for NaN in result
        if np.any(np.isnan(updated_w)):
            self._print(f"ERROR: NaN detected in updated_w")
            raise ValueError("NaN values in update result")
        
        return updated_w



    def lupdate(self, w: np.ndarray, lobbyist: LobbyistAgent, t: int) -> np.ndarray:
        
        """
        Updates the status of nodes with the influence of a given lobbyist with numerical stability.

        Arguments:
            w (np.ndarray): The current status of nodes.
            lobbyist (LobbyistAgent): The lobbyist influencing the nodes.
            t (int): Current time step for the lobbyist's strategy.

        Returns:
            np.ndarray: Updated node statuses considering lobbyist influence.
        """
        
        m = lobbyist.m
        s = lobbyist.strategy[t]
        
        if s is None:
            return w
                        
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']

        # Clip probabilities to avoid extreme values
        p_o = np.clip(p_o, self.EPSILON, 1 - self.EPSILON)
        p_p = np.clip(p_p, self.EPSILON, 1 - self.EPSILON)
        
        p = w * p_o + (1 - w) * p_p #subjective probability opt model
        # p = (1 - w) * p_o + w * p_p #subjective probability pess model
        p = np.clip(p, self.EPSILON, 1 - self.EPSILON) # clipping to avoid extreme values

        # Check for potential numerical issues
        if np.any(p < 1e-10) or np.any(p > 1 - 1e-10):
            self._print(f"WARNING: p very close to boundary in lupdate()")
            self._print(f"p range: [{np.min(p):.2e}, {np.max(p):.2e}]")
            self._print(f"w range: [{np.min(w):.2e}, {np.max(w):.2e}]")

        phi = self.phis #parametro phi della popolazione, per ora testati solo omogenei
        lam = self.lambdas #parametro lambda della popolazione, per ora testati solo omogenei
                    
        # opt model
        l1 = phi * w + (1 - phi) * lam #lambda per lobbista pessimista
        l2 = phi * (1-w) + (1 - phi) * lam #lambda per lobbista ottimista 
                    
        # pess model
        # l1 = phi * (1-w) + (1 - phi) * lam #lambda per lobbista pessimista
        # l2 = phi * w + (1 - phi) * lam #lambda per lobbista ottimista 

        l1 = np.clip(l1, 0, 1)
        l2 = np.clip(l2, 0, 1)

        if m == 0: # update with pessimist lobbyist
            # opt model

            # Calculate ratios with clipping to prevent overflow
            ratio_o_p = np.clip(p_o / p, self.EPSILON, 1/self.EPSILON)
            ratio_p_p = np.clip(p_p / p, self.EPSILON, 1/self.EPSILON)

            w1 =  s * (l1 * w + (1-l1) * w * ratio_o_p) + (1-s) * w

            w2 =  s * (l1 * (1 - w) + (1 - l1) * (1 - w) * ratio_p_p) + (1 - s) * (1 - w)

            # pess model
            # w1 =  s * (l1 * (1-w) + (1-l1) * (1-w) * ratio_o_p) + (1-s) * (1-w)

            # w2 =  s * (l1 * w + (1 - l1) * w * ratio_p_p) + (1 - s) * w

        elif m == 1: # update with optimist lobbyist
            # Calculate ratios with clipping to prevent overflow
            ratio_not_o_not_p = np.clip((1 - p_o) / (1 - p), self.EPSILON, 1/self.EPSILON)
            ratio_not_p_not_p = np.clip((1 - p_p) / (1 - p), self.EPSILON, 1/self.EPSILON)
            # opt model
            w1 = s * (l2 * w + (1 - l2) * w * ratio_not_o_not_p) + (1 - s) * w

            w2 = s * (l2 * (1 - w) + (1 - l2) * (1 - w) * ratio_not_p_not_p) + (1 - s) * (1 - w)

            # pess model
            # w1 = s * (l2 * (1 - w) + (1 - l2) * (1 - w) * ratio_not_o_not_p) + (1 - s) * (1 - w)

            # w2 = s * (l2 * w + (1 - l2) * w * ratio_not_p_not_p) + (1 - s) * w

        else:
            raise ValueError("Invalid model type for lobbyist")
        
        # Check for NaN before division
        if np.any(np.isnan(w1)) or np.any(np.isnan(w2)):
            self._print(f"ERROR: NaN detected before division in lupdate()")
            self._print(f"Lobbyist type: {m}")
            self._print(f"w1 NaN count: {np.sum(np.isnan(w1))}")
            self._print(f"w2 NaN count: {np.sum(np.isnan(w2))}")
            raise ValueError("NaN values detected in lupdate calculation")
        
        # Ensure denominator is not too small
        denominator = w1 + w2
        
        if np.any(denominator < self.EPSILON):
            self._print(f"WARNING: Near-zero denominator in lupdate()")
            self._print(f"min denominator: {np.min(denominator):.2e}")
            self._print(f"Lobbyist type: {m}")
            self._print(f"Affected nodes: {np.sum(denominator < self.EPSILON)}")
            denominator = np.clip(denominator, self.EPSILON, None)
        
        # Check for very large ratios that might indicate numerical issues
        max_ratio = np.max(w1 / denominator)
        min_ratio = np.min(w1 / denominator)
        if max_ratio > 1.0 + self.EPSILON or min_ratio < -self.EPSILON:
            self._print(f"WARNING: Division ratio outside [0,1] before clipping")
            self._print(f"Ratio range: [{min_ratio:.6f}, {max_ratio:.6f}]")
        
        # Perform division
        updated_w = w1 / denominator # opt model
        # updated_w = w2 / (w1 + w2) # pess model
        
        # Final clipping to ensure valid probability range
        updated_w = np.clip(updated_w, 0, 1)
        
        # Check for NaN in result
        if np.any(np.isnan(updated_w)):
            self._print(f"ERROR: NaN detected in updated_w")
            raise ValueError("NaN values in lupdate result")
        
        return updated_w
        

    def apply_lobbyist_influence(self, w: np.ndarray, t: int) -> np.ndarray:
       
        """
        Applies the influence of all lobbyists to the current node statuses.

        Arguments:
            w (np.ndarray): The current status of nodes.
            t (int): The current time step.

        Returns:
            np.ndarray: The updated statuses after lobbyist influence.
        """
                
        lobbyist_list = self.lobbyists.copy()        
        np.random.shuffle(lobbyist_list)
        
        if len(self.lobbyists) > 0:
            for lobbyist in lobbyist_list:
                if self.actual_iteration < lobbyist.max_t:
                    w = self.lupdate(w, lobbyist, t)
        return w

    def iteration(self) -> dict:
        """
        Performs one iteration of the diffusion process, updating node statuses.

        Arguments:
            node_status (bool, optional): Whether to return the node status after the iteration.

        Returns:
            dict: Information about the current iteration, including updated node statuses.
        """
        
        self.actual_status = self.status.copy() #copio lo stato iniziale per aggiornarlo a step durante il singolo time step, gli aggiornamenti interni non verranno salvati nello stato del sistema

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            return {"iteration": 0, "status": {i: value for i, value in enumerate(self.actual_status)}} #metto in system_status lo stato iniziale come iterazione 0 e vado all'iterazione successiva

        if self.params['model']['p_o'] < 0 or self.params['model']['p_o'] > 1:
            raise ValueError("Invalid value for p_o. It must be between 0 and 1.")
        if self.params['model']['p_p'] < 0 or self.params['model']['p_p'] > 1:
            raise ValueError("Invalid value for p_p. It must be between 0 and 1.")
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']
        
        self.actual_status = self.apply_lobbyist_influence(self.actual_status, self.actual_iteration) #interazioni con tutti i lobbisti in ordine casuale
        

        if np.any(np.isnan(self.actual_status)):
            raise ValueError("NaN found in actual_status after applying lobbyist influence")
        
        if np.all(self.actual_status >= 0) and np.all(self.actual_status <= 1):
             pass  
        else:
            indici_up = np.where(self.actual_status > 1)
            indici_down = np.where(self.actual_status < 0)                   
            if  np.any(indici_down):
                self._print(f"Nodes with weights <0 are: {indici_down};\n Values of weights < 0 are: {self.actual_status[indici_down]}")
            if np.any(indici_up):
                self._print(f"Nodes with weights >1 are: {indici_up};\n Values of weights >1 are: {self.actual_status[indici_up]}")
            raise ValueError("After applying lobbyist influence find status values less than 0 or grater than 1.")
 

        sender = np.random.randint(0, self.n) #scelgo un nodo che invierà un segnale a caso nel grafo
        
        try:        
            p = self.actual_status[sender] * p_o + (1 - self.actual_status[sender]) * p_p #calcolo la probabilità soggettiva del sender opt model
            # p = (1 - self.actual_status[sender]) * p_o + self.actual_status[sender] * p_p #calcolo la probabilità soggettiva del sender pess model
            signal = np.random.binomial(1, p) #genero il segnale con una binomiale in base alla probabilità soggettiva       
        except ValueError:
            return None
            
        receivers = np.array(list(self.graph.neighbors(sender))) #i nodi che ricevono il segnale sono i vicini del sender 
        if len(receivers) > 0:
            self.actual_status[receivers] = self.update(receivers, signal) #aggiorno lo stato dei nodi che ricevono il segnale
        
        if np.all(self.actual_status >= 0) and np.all(self.actual_status <= 1):
             pass  
        else:
            indici_up = np.where(self.actual_status > 1)
            indici_down = np.where(self.actual_status < 0)                       
            if  np.any(indici_down):
                self._print(f"Nodes with weights <0 are: {indici_down};\n Values of weights < 0 are: {self.actual_status[indici_down]}")
            if np.any(indici_up):
                self._print(f"Nodes with weights >1 are: {indici_up};\n Values of weights >1 are: {self.actual_status[indici_up]}")
            raise ValueError("Find status values less than 0 or grater than 1.")
        
        self.actual_iteration += 1 #incremento il contatore delle iterazioni
        self.status = self.actual_status #aggiorno lo stato del sistema

        return {"iteration": self.actual_iteration - 1, "status": {i: value for i, value in enumerate(self.actual_status)}} #ritorno lo stato aggiornato

    def iteration_bunch(self, T: int = 100, progress_bar:bool=True) -> list:
        
        """
        Runs the model for a specified number of iterations.

        Arguments:
            T (int, optional): The number of iterations to run.

        Returns:
            list: A list of the system status (dictionaries) at each iteration.
        """
        
        self.T = T
        np.random.seed(self.seed)
        
        for _ in tqdm(range(T), disable=not progress_bar):
            
            try: 
                its = self.iteration()
                self.system_status.append(its)
            except ValueError as e:
                self._print(f"Error in iteration: {e}")
                return self.system_status

        return self.system_status

    # Run the model until a steady state is reached or a maximum number of iterations
    def steady_state(self, 
                     max_iterations: int=1000000, 
                     nsteady:int=1000, 
                     sensibility:float=0.00001, 
                     progress_bar:bool=True, 
                     drop_evolution:bool=False) -> list:
        
        """ 
        Runs the model untill convergence or stopping condition is met.add()
        
        Arguments: 
            max_iterations (optional, int): stopping condition
            nsteady (optional, int): number of iterations with minimum number of opinion changes to declare convergence
            sensibility (optional, float): maximum opinion change tollerated to compute convergence
            progress_bar (optional, bool): show progress bar
            drop_evolution (optional, bool): keep in memory iterations dictionary (keep true if you want evolution plots in the end)
            
        Returns:
            list: A list of the system status at each iteration (dictionary)
        
        """
        
        self.T = max_iterations
        steady_it = 0  # Counter for consecutive steady iterations
        # Iterate until reaching a steady state or max_iterations
        for it in tqdm(range(max_iterations), disable=not progress_bar):
            its = self.iteration()
            if its is None:
                raise ValueError("Error in iteration")                
            
            # Check if the difference between consecutive states is below the threshold
            if it > 0:
                old = np.array([el for el in self.system_status[-1]['status'].values()]) # Previous status
                actual = np.array([el for el in its['status'].values()]) # Current status
                res = np.abs(old - actual)  # Calculate the difference
                if np.all((res < sensibility)):  # Check if the difference is small enough
                    steady_it += 1
                else:
                    steady_it = 0  # Reset if not steady

            self.system_status.append(its)  # Append current iteration status        

            # If steady state is achieved for nsteady consecutive iterations, stop
            if steady_it == nsteady:
                self._print(f'Convergence reached after {it} iterations')
                if drop_evolution:
                    return [self.system_status[-nsteady]]
                else:
                    return self.system_status[:-nsteady]
        
        if drop_evolution:
            self.system_status = [its] 
        # Return the status of the system at each iteration (if no steady state is reached)
        return self.system_status


    # Additional utility functions for numerical health checks
    def check_numerical_health(self):
        """
        Diagnostic function to check for numerical issues in the current state.
        Call this periodically during simulation if you suspect problems.
        """
        print("\n" + "="*60)
        print("NUMERICAL HEALTH CHECK")
        print("="*60)

        # Check current status
        print(f"Status range: [{np.min(self.status):.6f}, {np.max(self.status):.6f}]")
        print(f"Status mean: {np.mean(self.status):.6f}")
        print(f"Status std: {np.std(self.status):.6f}")

        # Check for values very close to boundaries
        near_zero = np.sum(self.status < 0.01)
        near_one = np.sum(self.status > 0.99)
        print(f"Nodes near 0 (< 0.01): {near_zero}")
        print(f"Nodes near 1 (> 0.99): {near_one}")

        # Check parameters
        print(f"\nParameters:")
        print(f"  p_o: {self.params['model']['p_o']}")
        print(f"  p_p: {self.params['model']['p_p']}")
        print(f"  lambda range: [{np.min(self.lambdas):.6f}, {np.max(self.lambdas):.6f}]")
        print(f"  phi range: [{np.min(self.phis):.6f}, {np.max(self.phis):.6f}]")

        # Check for NaN
        if np.any(np.isnan(self.status)):
            print("\nWARNING: NaN detected in status!")
            print(f" Number of NaN values: {np.sum(np.isnan(self.status))}")
        else:
            print("\nNo NaN values detected")

        print("="*60 + "\n")
