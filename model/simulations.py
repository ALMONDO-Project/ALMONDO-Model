import ndlib.models.ModelConfig as mc
from almondoModel import AlmondoModel
import numpy as np
import networkx as nx
import os
import uuid

# Function to generate a unique simulation ID for each run
def generate_unique_id():
    return str(uuid.uuid4())  # Generate a random UUID

def generate_param_dir(params):
    return "_".join(f"{k}{str(v).replace('.', '_')}" for k, v in sorted(params.items()))

for po in [0.9, 0.8, 0.5]:
  for pp in [0.1, 0.2, 0.5]:
    for alpha in [0, 0.1, 0.5, 0.9]:
      graphtype = 'ER'
      N = 1000
      per = 0.1
      nlob = 2
      initialdistr = 'uniform'
      T = 100

      # Simulation parameters
      model_pars = {
          "po": po,
          "pp": pp,
          "alpha": alpha,
          "graphtype": graphtype,
          "N": N,
          "per": per,
          "nlobbyists": nlob,
          "initial_distr": initialdistr,
          "T": T
      }

      sim_pars = {
          "nruns": 10
      }

      os.makedirs('results/', exist_ok=True)

      # Generate parameter-based directory name
      param_dir = generate_param_dir(model_pars)
      param_dir_path = os.path.join("results/", f"params_{param_dir}")
      os.makedirs(param_dir_path, exist_ok=True)

      # Unique identifier for this script run
      sim_id = generate_unique_id()

      # Create directory for this simulation
      save_dir = os.path.join(param_dir_path, f"sim_{sim_id}")
      os.makedirs(save_dir, exist_ok=True)

      # Simulation loop
      for run in range(sim_pars['nruns']):
          run_dir = os.path.join(save_dir, f'run{run}')
          os.makedirs(run_dir, exist_ok=True)

          plot_dir = os.path.join(run_dir, 'plots')
          os.makedirs(plot_dir, exist_ok=True)

          # Generate graph
          G = nx.erdos_renyi_graph(n=model_pars['N'], p=model_pars['per'])
          nx.write_edgelist(G, f"{run_dir}/graph.edgelist", delimiter=',')

          # Simulate model (assuming AlmondoModel has appropriate methods)
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

          # Configure lobbyists
          for id in range(model_pars['nlobbyists']):
              model.add_lobbyist(id)
              model.set_lobbyist_model(id, models[id])
              model.set_lobbyist_strategy(id, matrix=matrices[id])

          # Run iterations
          iterations = model.iteration_bunch(T=model_pars['T'])
          model.save_all_status(run_dir, filename=f'status', format='json')
          model.save_final_state(run_dir, filename=f'final', format='csv')
          model.plot_evolution(path=f"{plot_dir}/evolution")
          model.visualize_initial_weights(path=f"{plot_dir}/initial_weights")
          model.visualize_final_weights(path=f"{plot_dir}/final_weights")
          model.visualize_final_probabilities(path=f"{plot_dir}/final_probabilities")

          # Optional: Steady-state analysis
          steady_state = model.steady_state(drop_evolution=True)
          # model.save_all_status(run_dir, filename=f'status_steady', format='json')
          model.save_final_state(run_dir, filename=f'final_steady', format='csv')
          # model.plot_evolution(path=f"{plot_dir}/evolution")
          model.visualize_final_probabilities(path=f"{plot_dir}/final_probabilities")