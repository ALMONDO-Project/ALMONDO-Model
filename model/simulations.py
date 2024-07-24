from ndlib.models.DiffusionModel import DiffusionModel
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
import numpy as np
import networkx as nx
import gzip
import pickle
import seaborn as sns
import os

nruns = 100
save_dir = 'results/'

pars = [0.01, 0.99, 0.8, 1000, 0.01]

folder = '_'.join([str(e) for e in pars])
save_dir = os.path.join(save_dir, folder)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for run in range(nruns):
    file_name = os.path.join(save_dir, f'iterations_{run}')
    if not os.path.exists(file_name):
        g = nx.erdos_renyi_graph(pars[3],pars[4])
        initial_status = np.random.rand(pars[3])
        # Model selection
        model = op.AlmondoModel(graph=g, seed=4)
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


