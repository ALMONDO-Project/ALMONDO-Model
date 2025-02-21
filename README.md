# AlmondoModel

AlmondoModel is a diffusion model designed to simulate the diffusion of influence across a network with the influence of lobbyists having different strategies. It extends the `DiffusionModel` class from the `ndlib` library, enabling the simulation of opinion evolution and the effects of lobbying activities over time. The model can be customized with various parameters such as probabilities for optimistic and pessimistic events, node influence factors, and more.

## Features

- **Agent-based simulation**: Simulates a network of nodes influenced by lobbyists with varying strategies.
- **Optimistic/Pessimistic models**: Lobbyists can have either optimistic or pessimistic models influencing the diffusion process.
- **Flexible configuration**: Customize various parameters including event probabilities, influence factors, and strategies.
- **Steady-state detection**: The model can run until it reaches a steady state or the maximum iteration limit is reached.

## Requirements
Python 3.x

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/almondomodel.git
cd almondomodel
pip install -r requirements.txt
```

## Example of model usage

The AlmondoModel can be used to simulate influence diffusion on a network with lobbyist interventions. Below is an example usage:

```python
import ndlib.models.ModelConfig as mc
import networkx as nx
from .almondoModel import AlmondoModel # This will be imported from ndlib once the model is loaded there
import numpy as np

p_o = 0.01
p_p = 0.99
lambda_v = 0.5
phi_v = 0.5

# Create a sample graph (e.g., a random graph)
G = nx.erdos_renyi_graph(100, 0.1)

#Create a configuration 
config = mc.Configuration()

# Set general parameters for the model
config.add_model_parameter("p_o", p_o)
config.add_model_parameter("p_p", p_p)
config.add_node_configuration("lambda", i, lambda_v)
config.add_node_configuration("phi", i, phi_v)

# Initialize the Almondo model
model = AlmondoModel(G, seed=42)

# Set initial status for all nodes
model.set_initial_status(config, kind='uniform', uniform_range=[0, 1])

# Add lobbyists with strategies (example)
strategy = np.random.uniform(0, 1, size=(10, 10))  # Example strategy matrix
model.add_lobbyist(m=1, strategy=strategy)

# Run the model for 100 iterations
system_status = model.steady_state(max_iterations=100000)

# Print final system status
print(system_status[-1])
```

## Simulator
### Initializing the Simulator
You can initialize the simulator with the required parameters. Here's an example:

```python

simulator = ALMONDOSimulator(
    N=100,  # Number of agents
    initial_distribution='random',  # Initial distribution of agent states
    T=100,  # Maximum number of iterations
    p_o=0.1,  # Opinion dynamics parameter
    p_p=0.5,  # Opinion dynamics parameter
    lambda_values=0.2,  # Influence susceptibility
    phi_values=0.3,  # Resistance to influence
    base='results/',  # Directory to store results
    scenario='scenario_1',  # Scenario name for organizing results
    nruns=100,  # Number of simulation runs
    n_lobbyists=5,  # Number of lobbyists
    lobbyists_data={}  # Dictionary containing lobbyists' strategies
)
```

### Running a Single Simulation
After initializing the simulator, you can run a single simulation with specific lambda_v and phi_v values:

```python
status, final_distributions = simulator.single_run(lambda_v=0.2, phi_v=0.3)
```

### Running Multiple Simulations (Monte Carlo)
You can run multiple simulations (Monte Carlo simulations) to get statistical results:

``` python
simulator.runs(lambda_v=0.2, phi_v=0.3, overwrite=False)
```

If you want to run experiments for all combinations of lambda and phi values, you can execute the following:
``` python
simulator.execute_experiments(overwrite_runs=False)
```

### Saving Results
The simulator automatically saves results after each run. If you want to manually save the simulation configuration or system status, you can use the following methods:

``` python
simulator.save_config()  # Save the configuration
simulator.save_system_status(path)  # Save the system status to a file
```

### Getting Simulation Results
To retrieve the results after running the simulation, use:

``` python
model, system_status, lobbyists = simulator.get_results()
```

### Parameters
* N: Number of agents in the network.
* initial_distribution: The type of initial agent states (e.g., 'random', 'uniform').
* T: Maximum number of iterations for the simulation.
* p_o: Opinion dynamics parameter.
* p_p: Opinion dynamics parameter.
* lambda_values: Influence susceptibility values for agents (float or list).
* phi_values: Resistance to influence values for agents (float or list).
* base: Base directory for storing simulation results.
* scenario: Scenario name for organizing results.
* nruns: Number of simulation runs for Monte Carlo simulations.
* n_lobbyists: Number of lobbyists in the simulation.
* lobbyists_data: A dictionary containing lobbyists' strategies and parameters.

### Files and Directories
- results/: Base directory for saving simulation results.
- results/{scenario}/strategies/: Directory where lobbyists' strategies are stored.
- results/{scenario}/figures/: Directory where default figures are stored.
- results/{scenario}/config.json: The configuration file of the simulation.
- results/{scenario}/{lambda_v}_{phi_v}/runs_data.json: File containing the data from all runs of the experiment.

## Experiment example
``` python

 """
    Set up an experiment here. 
    With the 'params' dictionaries you can specify immutable parameters across experiments.
    Such parameters are:
    p_o (float): probability of optimist model
    p_p (float): probability of pessimist model
    initial_distribution (string): kind of initial distribution (customizable distributions are not implemented at the present time, only uniform initial distirbution can be used)
    lambda_values (list): list of lambda_v to test in the experiments, a lamdba value can be a float or a list of length N
    phi_values (list): list of phi_v to test in the experiment, a phi_v can be a float or a list of length N
    base (str): folder where results are stored
    scenario (str): folder where you want this set of experiments to be stored
    N (int): number of agents in the population
    lobbyists_data (dict): each key is the id of a lobbyist, each value is the information for that lobbyist
        a single lobbyist is identified by:
         - m (int): model, where 1 = optimist and 0 = pessimist
         - B (int): budget
         - c (int): cost of a signal
         - strategies (list): list of strategies to use (filenames to retrieve the strategies form)
         - T (int): number of active time steps 
         
    Pass these parameters to ALMONDOSimulator, specifying the number of runs nruns you want to perform. 
    
    Use the method ALMONDOSimulator.execute_experiments() to run the simulations. Use the attribute overwrite_runs to overwrite existing runs. 
    
    In these experiments we are going to create a population of nl lobbyists where 1/2 of lobbyists are optimist and 1/2 of lobbyists are pessimists. Each lobbyist
    has a budget of 300000 and is active for 3000 iterations. The cost of a single signal is 1. 
    
    """      

NLs = [0, 1, 2, 3, 4, 20] #number of lobbyists in the simulations

params = {
    'N': 500,
    'p_o': 0.01,
    'p_p': 0.99,
    'initial_distribution': 'uniform',
    'T': 10000,
    'lambda_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'phi_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # 'lambda_values': [0.0, 1.0],
    # 'phi_values': [0.0, 1.0],
    'base': 'results'
}    


for nl in NLs:
    params['scenario'] = f'balanced_budgets/{nl}_lobbyists'
    params['n_lobbyists'] = nl
    if nl > 0:
        params['lobbyists_data'] = dict()
        for id in range(nl):
            params['lobbyists_data'][id] = {'m': id%2, 'B': 300000, 'c': 1, 'strategies': [], 'T': 3000}    

    os.makedirs(params['base'], exist_ok=True)
    path = os.path.join(params['base'], params['scenario'])
    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, 'initial_config.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f'performing simulations for {nl} lobbyists with balanced budgets 300000')
                
    simulator = ALMONDOSimulator(**params, nruns=nruns)
    simulator.execute_experiments(overwrite_runs=False)   
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
This simulator was created as part of ongoing research into opinion dynamics and lobbying influence. It relies on ndlib for network-based modeling and networkx for graph management.
