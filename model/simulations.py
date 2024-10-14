from ndlib.models.DiffusionModel import DiffusionModel
import ndlib.models.ModelConfig as mc
from lobbyistAgent import LobbyistAgent
from almondoModel import AlmondoModel
import numpy as np
import networkx as nx
import gzip
import pickle
import seaborn as sns
import os
import random

nruns = 100
save_dir = 'results/lobbyist'


p_o = 0.01
p_p = 0.99
l = 0.8
N = 1000
p_er = 0.01
n_lobbyists = 5
max_iterations = 10000

folder = '_'.join([str(e) for e in pars])
save_dir = os.path.join(save_dir, folder)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for run in range(nruns):
    file_name = os.path.join(save_dir, f'iterations_{run}')
    if not os.path.exists(file_name):
        g = nx.erdos_renyi_graph(pars[3], pars[4])
        initial_status = np.random.rand(pars[3])

        # Create lobbyist agents
        lobbyists = []
        for _ in range(pars[5]):
            model_type = random.choice(['optimistic', 'pessimistic'])
            #given the budget and maybe some other input parameters create the signal matrix and pass it to the LobbyistAgent
            
            lobbyist = LobbyistAgent(self, model=model_type, graph=g, signalmatrix = None, seed=None)
            lobbyists.append(lobbyist)

        # Model selection
        model = AlmondoModel(graph=g, seed=4, lobbyists=lobbyists)

        # Model configuration
        config = mc.Configuration()
        config.add_model_parameter("p_o", pars[0])
        config.add_model_parameter("p_p", pars[1])
        config.add_model_parameter("l", pars[2])
        model.set_initial_status(configuration=config, status=initial_status)
        
        iterations = model.steady_state(max_iterations=1000000, nsteady=1000, sensibility=0.0001)
        with gzip.open(file_name, 'wb') as f:
            pickle.dump(iterations, f)
    else:
        continue