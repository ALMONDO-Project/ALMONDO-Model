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

model_pars = {
    "po": 0.01,
    "pp": 0.99,
    "alpha": 0.2,
    "graphtype": "ER",
    "N": 1000,
    "per": 0.01,
    "nlobbyists": 2,
    "initial_distr": "uniform",
    "T": 100
}

save_dir = [f"{k}{v}" for k,v in model_pars.items()]
save_dir = '_'.join(save_dir)

sim_pars = {
    "nruns": 10,
    "save_dir": save_dir
}

save_dir = os.path.join("results/", save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for run in range(sim_pars['nruns']):
    run_dir = os.path.join(save_dir, f'run{run}')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    plot_dir = os.path.join(run_dir, f'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    G = nx.erdos_renyi_graph(n=model_pars['N'], p=model_pars['per'])
    nx.write_edgelist(G, f"{run_dir}/graph.edgelist", delimiter=',')
    
    model = AlmondoModel(graph=G, seed=4)

    # Model configuration
    config = mc.Configuration()
    config.add_model_parameter("p_o", model_pars['po'])
    config.add_model_parameter("p_p", model_pars['pp'])
    config.add_model_parameter("alpha", model_pars['alpha'])
    model.set_initial_status(configuration=config, kind=model_pars['initial_distr'])

    models = [0, 1]
    matrix1 = np.random.randint(0, 2, size=(model_pars['T'], model_pars['N']))
    matrix2 = np.random.randint(0, 2, size=(model_pars['T'], model_pars['N']))

    matrices = [matrix1, matrix2]

    for id in range(2):
        model.add_lobbyist(id)
        model.set_lobbyist_model(id, models[id])
        model.set_lobbyist_strategy(id, matrix = matrices[id])

    iterations = model.iteration_bunch(T=model_pars['T'])
    
    model.save_all_status(run_dir, filename='status', format='json')
    model.save_final_state(run_dir, filename='final', format='csv')
    model.plot_evolution(path=f"{plot_dir}/evolution.png")
    model.visualize_initial_weights(path=f"{plot_dir}/initial.png")
    model.visualize_final_weights(path=f"{plot_dir}/final.png")
    model.visualize_final_probabilities(path=f"{plot_dir}/final_probabilities.png")
    
    iterations = model.steady_state()
    
    model.save_all_status(run_dir, filename='status', format='json')
    model.save_final_state(run_dir, filename='final', format='csv')
    model.plot_evolution(path=f"{plot_dir}/evolution.png")
    model.visualize_initial_weights(path=f"{plot_dir}/initial.png")
    model.visualize_final_weights(path=f"{plot_dir}/final.png")
    model.visualize_final_probabilities(path=f"{plot_dir}/final_probabilities.png")
    