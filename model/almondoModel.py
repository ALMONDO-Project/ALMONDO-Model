from typing_extensions import final
from ndlib.models.DiffusionModel import DiffusionModel
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import random
import os
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import future.utils
import seaborn as sns



# Author information
__author__ = ["Alina Sirbu", "Giulio Rossetti", "Valentina Pansanella"]
__email__ = [
    "alina.sirbu@unipi.it",
    "giulio.rossetti@isti.cnr.it",
    "valentina.pansanella@isti.cnr.it",
]

# Define the AlmondoModel class, which extends ndlib's DiffusionModel
class AlmondoModel(DiffusionModel):

    # Initialization of the model, setting up the graph, and model parameters
    def __init__(self, graph, seed=None):
        super(self.__class__, self).__init__(graph, seed)

        # Define whether the state is continuous or discrete (here it's continuous)
        self.discrete_state = False

        # Define model parameters with descriptions, ranges, and whether they are optional
        self.parameters = {
            "model": {
                "p_o": {  # Probability of an optimistic event
                    "descr": "Probability of event optimist model",
                    "range": [0, 1],
                    "optional": False
                },
                "p_p": {  # Probability of a pessimistic event
                    "descr": "Probability of event pessimist model",
                    "range": [0, 1],
                    "optional": False
                },
                "alpha": {
                    "descr": "???",
                    "range": [0,1],
                    "optional": True,
                    "default": 0
                }, 
                "lambda": {
                    "descr": "???",
                    "range": [0,1],
                    "optional": True,
                    "default": 1
                }
            },
            "nodes": {},  # Node-specific parameters (empty for now)
            "edges": {}   # Edge-specific parameters (empty for now)
        }

        self.name = "Almondo"  # Name of the model
        self.n = self.graph.number_of_nodes()  # Number of nodes in the graph
        self.seed = seed  # Random seed (for reproducibility)
        self.T = None
        self.status = None  # Status of each node (initially not set)
        self.actual_status = None
        self.lobbyists = []  # List to store lobbyist agents influencing the system
        self.system_status = []
        
        
    # Function to set the initial status of nodes (w values for nodes)
    def set_initial_status(self, configuration=None, kind='uniform', uniform_range=[0, 1], unbiased_value=0.5, status=None, gaussian_params=None, initial_lambda = 0.5):
        super(AlmondoModel, self).set_initial_status(configuration)
        # If no status is provided, assign random values to the nodes' status
        if kind == 'uniform':
            # Check that uniform_range[0] >= 0 and uniform_range[1] <= 1
            if uniform_range[0] < 0 or uniform_range[1] > 1:
                raise ValueError("uniform_range must be within [0, 1]")
            self.status = np.random.uniform(low=uniform_range[0], high=uniform_range[1], size=self.n)  # Random uniform initial status for each node in the interval uniform_range

        elif kind == 'unbiased':
            # Check that unbiased_value is in [0,1]
            if unbiased_value < 0 or unbiased_value > 1:
                raise ValueError("unbiased_value must be between 0 and 1")
            self.status = np.full(self.n, unbiased_value)  # Unbiased initial status: each node has the same initial status set by the user

        elif kind == 'gaussian_mixture':
            # The initial distribution is generated from a Gaussian mixture
            # The Gaussian mixture should be tuned by parameters passed via gaussian_params
            if gaussian_params is None or 'means' not in gaussian_params or 'stds' not in gaussian_params or 'weights' not in gaussian_params:
                raise ValueError("gaussian_params must contain 'means', 'stds', and 'weights'")

            means = gaussian_params['means']
            stds = gaussian_params['stds']
            weights = gaussian_params['weights']

            # Check that means, stds, and weights have the same length and sum of weights is 1
            if len(means) != len(stds) or len(means) != len(weights):
                raise ValueError("means, stds, and weights must have the same length")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("The sum of weights must be 1")

            # Generate Gaussian mixture
            gaussians = [np.random.normal(loc=means[i], scale=stds[i], size=self.n) for i in range(len(means))]
            mixture = np.zeros(self.n)

            # Sample according to the weights
            for i in range(len(weights)):
                indices = np.random.choice(self.n, size=int(self.n * weights[i]), replace=False)
                mixture[indices] = gaussians[i][indices]

            # Ensure values are between 0 and 1
            self.status = np.clip(mixture, 0, 1)

        elif kind == 'user_defined' and status is not None:
            # Check that status is an array of self.n entries with each entry being a float in [0,1]
            if len(status) != self.n:
                raise ValueError(f"Status array must have {self.n} entries")
            if not all(0 <= val <= 1 for val in status):
                raise ValueError("Each status value must be a float in [0, 1]")

            self.status = status  # Use the status provided by the user

        else:
            raise ValueError("Invalid kind or missing status for 'user_defined'")
        
        print(self.params['model'])

    def generate_lambda(self, w, s):
        c = 0.1 #deve essere un parametro del modello? della simulazione? lo settiamo noi a 0.1
        try: 
            l = np.abs((1 - s) - w)**self.params['model']['alpha'] #alpha deve essere un parametro del modello passato dall'utente
            return l
        except:
            l = self.params['model']['lambda']
            return l 
        

    # Function to update node status based on a signal and current status (without lobbyist influence)
    def update(self, receivers, s):
        w = self.actual_status[receivers]
        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']
        p = w * p_o + (1 - w) * p_p  # Combined probability based on current node's status
        l = self.generate_lambda(self.actual_status[receivers], s) #ma quindi lambda non dipende in alcun modo dal lambda precedente?
        return l * w + (1 - l) * w * (s * (p_o / p) + (1 - s) * ((1 - p_o) / (1 - p)))

    # Function to update a node's status taking lobbyist influence into account
    def lupdate(self, w, lobbyist, t):
        m = lobbyist.get_model()  # Get lobbyist parameters remember that m = 1 is optimistic and m = 0 is pessimistic
        s = lobbyist.get_current_strategy(t)
        if s is not None:
          c = m * s # Create a signal from the lobbyist at time t
          p_o = self.params['model']['p_o']
          p_p = self.params['model']['p_p']
          p = w * p_o + (1 - w) * p_p
          l = self.generate_lambda(self.actual_status, c)
          return (1 - s) * w + s * w * (l + (1 - l) * (c * (p_o / p) + (1 - c) * ((1 - p_o) / (1 - p))))
        else:
          return w

    # Function to apply the influence of all lobbyists to the system's current status
    def apply_lobbyist_influence(self, w, t):
        for lobbyist in self.lobbyists:
            w = self.lupdate(w, lobbyist, t)  # Apply influence of each lobbyist
        return w

    # Perform one iteration of the model
    def iteration(self, node_status=True):
        self.actual_status = self.status.copy()  # Copy current status of the nodes

        # First iteration, no updates yet, just return the initial status
        if self.actual_iteration == 0:
            self.actual_iteration += 1
            if node_status:
                return {"iteration": 0, "status": {i: value for i, value in enumerate(self.actual_status)}, "sender": None, "signal": None}
            else:
                return {"iteration": 0, "status": {}, "sender": None, "signal": None}

        p_o = self.params['model']['p_o']
        p_p = self.params['model']['p_p']

        # Apply lobbyist influence to current status
        self.actual_status = self.apply_lobbyist_influence(self.actual_status, self.actual_iteration)
        # Randomly select a node to send a signal
        sender = random.randint(0, self.n - 1)

        # Generate a signal for the sender node (based on the node's current status)
        p = self.actual_status[sender] * p_o + (1 - self.actual_status[sender]) * p_p
        signal = np.random.binomial(1, p)  # Signal is binary (1 or 0)

        # Get the sender's neighbors and update their status based on the signal
        receivers = np.array(list(self.graph.neighbors(sender)))
        if len(receivers) > 0:
            self.actual_status[receivers] = self.update(receivers, signal)

        # Increment the iteration count and update the status of the nodes
        self.actual_iteration += 1
        self.status = self.actual_status

        # Return the current iteration's information, including the sender and signal
        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": {i: value for i, value in enumerate(self.actual_status)}, "sender": sender, "signal": signal}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {i: value for i, value in enumerate(self.actual_status)}, "sender": sender, "signal": signal}

    # Run the model for T iterations
    def iteration_bunch(self, T=100):
        self.T = T
        random.seed(self.seed)  # Set random seed for reproducibility
        for _ in tqdm(range(T)):  # Use tqdm for a progress bar
            its = self.iteration()  # Run one iteration
            self.system_status.append(its)  # Append the result to system status
        return self.system_status

    # Run the model until a steady state is reached or a maximum number of iterations
    def steady_state(self, max_iterations=1000000, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True, drop_evolution=True):
        self.T = max_iterations
        steady_it = 0  # Counter for consecutive steady iterations

        # Iterate until reaching a steady state or max_iterations
        for it in tqdm(range(max_iterations)):
            its = self.iteration(node_status=True)

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
                print(f'Convergence reached after {it} iterations')
                return self.system_status

        # Return the status of the system at each iteration (if no steady state is reached)
        return self.system_status

##################### SAVE FUNCTIONS ##############################################################################################################

    def save_all_status(self, save_dir, filename='status', format='json'):
        output_file = os.path.join(save_dir, f'{filename}.{format}')        
        if format == 'json':
            with open(output_file, 'w') as ofile:
                json.dump(self.system_status, ofile)
        elif format == 'pickle':
            with open(output_file, 'wb') as ofile:
                pickle.dump(self.system_status, ofile)

    def save_final_state(self, save_dir, filename='final', format='csv'):
        iteration_count = len(self.system_status)  # Total iterations completed
        output_file = os.path.join(save_dir, f'{filename}_{iteration_count}.{format}')
        final_status = [el for el in self.system_status[-1]['status'].values()]
        np.savetxt(output_file, final_status, delimiter=',')


    ######################## GET FUNCTIONS #############################################
    def get_nagents(self):
        return self.graph.number_of_nodes()
    def get_nconnections(self):
        return self.graph.number_of_edges()
    def get_model_parameters(self):
        pass
    def f(self):
        pass
    def get_statistics(self):
        pass

    ######################## PLOT FUNCTIONS #############################################
    def visualize_graph(self):
        #nella visualizzazione ci vanno i colori in base all'opinione
        pass
    
    def plot_evolution(self, path=None):
        def hex_to_rgb(value):
            '''
            Converts hex to rgb colours
            value: string of 6 characters representing a hex colour.
            Returns: list length 3 of RGB values'''
            value = value.strip("#") # removes hash symbol if present
            lv = len(value)
            return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


        def rgb_to_dec(value):
            '''
            Converts rgb to decimal colours (i.e. divides each value by 256)
            value: list (length 3) of RGB values
            Returns: list (length 3) of decimal values'''
            return [v/256 for v in value]

        def get_continuous_cmap(hex_list, float_list=None):
            ''' creates and returns a color map that can be used in heat map figures.
                If float_list is not provided, colour map graduates linearly between each color in hex_list.
                If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
                
                Parameters
                ----------
                hex_list: list of hex code strings
                float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
                
                Returns
                ----------
                colour map'''
            rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
            if float_list:
                pass
            else:
                float_list = list(np.linspace(0,1,len(rgb_list)))
            cdict = dict()
            for num, col in enumerate(['red', 'green', 'blue']):
                col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
                cdict[col] = col_list
            cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
            return cmp
        spaghetti_hex_list = ['#357db0', '#18A558', '#ce2626']
        spaghetti_cmap=get_continuous_cmap(spaghetti_hex_list)
        
        fig, ax = plt.subplots()

        spaghetti_hex_list = ['#357db0', '#18A558', '#ce2626']


        """
        Generates the plot

        :param filequintet: Output filequintet
        :param percentile: The percentile for the trend variance area
        """

        nodes2opinions = {}
        node2col = {}

        last_it = self.system_status[-1]['iteration'] + 1
        last_seen = {}

        for it in self.system_status:
            weights = np.array([el for el in it['status'].values()])
            sts = self.params['model']['p_o'] * weights + self.params['model']['p_p'] * (1-weights)  # update conditional probabilities of event will occur
            its = it['iteration']
            for n, v in enumerate(sts):
                if n in nodes2opinions:
                    last_id = last_seen[n]
                    last_value = nodes2opinions[n][last_id]

                    for i in range(last_id, its):
                        nodes2opinions[n][i] = last_value

                    nodes2opinions[n][its] = v
                    last_seen[n] = its
                else:
                    nodes2opinions[n] = [0]*last_it
                    nodes2opinions[n][its] = v
                    last_seen[n] = 0
                    if v < 0.33:
                        node2col[n] = spaghetti_hex_list[0]
                    elif 0.33 <= v <= 0.66:
                        node2col[n] = spaghetti_hex_list[1]
                    else:
                        node2col[n] = spaghetti_hex_list[2]

        mx = 0
        for k, l in future.utils.iteritems(nodes2opinions):
            if mx < last_seen[k]:
                mx = last_seen[k]
            x = list(range(0, last_seen[k]))
            y = l[0:last_seen[k]]
            ax.plot(x, y, lw=1.5, alpha=0.5, color=node2col[k])
        plt.xlabel('t')
        plt.ylabel(r'$p_{i,t}$')
        plt.title('Optimist model probability evolution')
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close()

    def visualize_initial_weights(self, path=None):
        initial_status = self.system_status[0]['status'].values()
        plt.figure(figsize=(10, 6))
        _, bins, _ = plt.hist(initial_status, 15, density=True)
        plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        plt.xlabel(r'$w_{i,t}$')
        plt.xlabel('Frequency')
        plt.title('Optimist model weights distribution')
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close()

    def visualize_final_weights(self, path=None):
        final_status = self.system_status[-1]['status'].values()
        plt.figure(figsize=(10, 6))
        _, bins, _ = plt.hist(final_status, 15, density=True)
        plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        plt.xlabel(r'$w_{i,t}$')
        plt.xlabel('Frequency')
        plt.title('Optimist model weights distribution')
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def visualize_final_probabilities(self, path=None):
        weights = np.array([el for el in self.system_status[-1]['status'].values()])
        probabilities = self.params['model']['p_o'] * weights + self.params['model']['p_p'] * (1-weights)  # update conditional probabilities of event will occur
        ax = sns.histplot(probabilities, bins = 50, color='lightblue', alpha=1.0, stat='percent')
        ax.set_xlabel(r'$p_{i,T}$')
        ax.set_ylabel('% agents')
        ax.set_title('Final probability distribution of optimist model')
        ax.set_xlim(0.0, 1.0)
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close()
    

############################# LOBBYIST METHODS #########################################################################################################

    def add_lobbyist(self, id):
        self.lobbyists.append(self.LobbyistAgent(id))
    
    def get_lobbyists(self):
        return self.lobbyists

    def get_lobbyist_by_id(self, id):
        return self.lobbyists[id]

    def remove_lobbyist_by_id(self, id):
        pass

    def set_lobbyist_model(self, id, model):
        lobbyist = self.get_lobbyist_by_id(id)
        lobbyist.set_model(model)

    def set_lobbyist_budget(self, id, budget):
        pass
        
    def set_lobbyist_user_strategy(self, id, probabilities):
        pass

    def set_lobbyist_strategy(self, id, matrix=None):
        if matrix is not None:
          lobbyist = self.get_lobbyist_by_id(id)
          lobbyist.set_strategy(matrix)

    class LobbyistAgent:
        def __init__(self, id, seed=None):
            if seed is not None:
                np.random.seed(seed)
            """
                L'utente deve
                - settare un modello m (non opzionale)
                    con possibili valori {"optimist", "pessimist"}
                    che poi andranno tradotti in 0 e 1
                - settare un budget b (opzionale)
                    vincoli:
                        - b >= 0
                        - b <= B? vogliamo mettere un valore massimo settabile? O in principio è possibile anche effettuare simulazioni con un budget infinito? da valutare
                - settare una strategia-utente su (opzionale)
                    vincoli:
                        - su = [su_1, su_2, ..., su_N] con N = dimensione del grafo del modello
                        - su_i ha come possibili valori il range [0,1]
                        - altro?
                - settare una strategia S (non opzionale)
                    vincoli:
                        - S => è una matrice T x N (deve essere x forza così?)
                        - S[t, i]: indica se il lobbista interagisce con il nodo i al tempo t
                        - se b e su sono definiti la matrice è calcolata in base a b e s in *qualche modo* -->
                            - fissare un numero massimo di interazioni che il lobbista vuole avere ad ogni timestep sennò le interazioni son tutte all'inizio? boh
                            - ...
                        - se b e su non sono definiti la matrice deve essere passata dall'utente in input (letta da file)
            """
            self.id = id
            self.model = None
            self.budget = None
            self.probabilities = None
            self.strategy = None

        ############################ SET FUNCTIONS ####################################################################
        def set_model(self, model):
            self.model = model
        def set_budget(self, budget):
            pass
        def set_user_strategy(self, probabilities):
            pass

        def set_strategy(self, matrix):
            self.strategy = matrix

        ########## GET FUNCTIONS ##############
        def get_current_strategy(self, t):
            if self.strategy is not None and t < len(self.strategy):
                return self.strategy[t]

        def get_strategy(self):
            if self.strategy is not None:
                return self.strategy

        def get_model(self):
            return self.model

        def get_budget(self):
            pass

        def get_user_strategy(self):
            pass

        def get_lobbyist_id(self):
            return self.id
