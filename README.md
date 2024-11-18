# AlmondoModel Simulation with Lobbyist Agents

This repository contains a simulation framework based on the **AlmondoModel** to study opinion dynamics in a network under the influence of lobbyist agents. The simulation is implemented using Python and the [NDlib](https://ndlib.readthedocs.io) library, with support for various configurations and output visualizations.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Parameters](#parameters)
6. [Outputs](#outputs)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview
The simulation models a network of agents whose opinions evolve under:
- Interpersonal influence based on network structure.
- External lobbying strategies applied by **lobbyist agents**.

### Key Goals:
- Study opinion evolution dynamics in different network topologies.
- Assess the impact of lobbying strategies.
- Generate and visualize results, including the network evolution and opinion dynamics.

---

## Features
- **Network Topologies:** Easily configurable (e.g., Erdos-Renyi graphs).
- **Lobbyist Strategies:** Supports custom influence matrices for each lobbyist.
- **Customizable Parameters:** Fine-tune network size, lobbying influence, and interaction probabilities.
- **Visualization:** Automatically generates and saves evolution plots, initial/final weight distributions, and final probabilities.

---

## Installation
### Prerequisites
- Python 3.7+
- Libraries:
  - `ndlib`
  - `numpy`
  - `networkx`
  - `seaborn`
  - `pickle`
  - `matplotlib`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/almondo-lobbyist-simulation.git
   cd almondo-lobbyist-simulation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---


# Almondo Model

The **Almondo Model** is an extension of the `ndlib` library for simulating opinion dynamics in networked systems, with a focus on the influence of lobbying strategies. This model introduces several features to model and analyze how opinions evolve under the influence of events and external agents.

## Features

- **Custom Initialization**: Supports various initialization strategies for node states, including uniform, unbiased, Gaussian mixture, and user-defined distributions.
- **Lobbyist Influence**: Incorporates external lobbying agents that influence node opinions based on predefined strategies.
- **Event Modeling**: Simulates the probability of optimistic and pessimistic events affecting nodes' status.
- **Dynamic Lambda**: A tunable parameter governing how strongly node opinions are updated.
- **Visualization**: Provides tools for plotting opinion evolution, initial and final weight distributions, and final probabilities.
- **Steady State Detection**: Stops simulation upon reaching a steady state within a predefined sensitivity threshold.

## Installation

Clone this repository and ensure you have the following Python libraries installed:

- `numpy`
- `matplotlib`
- `seaborn`
- `future`
- `tqdm`
- `ndlib`

Install missing dependencies using pip:

```bash
pip install numpy matplotlib seaborn future tqdm ndlib
```

## Usage

### 1. Import the Model

```python
from AlmondoModel import AlmondoModel
import networkx as nx

# Create a graph
G = nx.erdos_renyi_graph(100, 0.1)

# Instantiate the model
model = AlmondoModel(graph=G)
```

### 2. Set Initial Status

Initialize the model's nodes using one of the following strategies:

```python
# Uniform initialization
model.set_initial_status(kind='uniform', uniform_range=[0, 1])

# Unbiased initialization
model.set_initial_status(kind='unbiased', unbiased_value=0.5)

# Gaussian Mixture
gaussian_params = {
    'means': [0.2, 0.8],
    'stds': [0.1, 0.1],
    'weights': [0.5, 0.5]
}
model.set_initial_status(kind='gaussian_mixture', gaussian_params=gaussian_params)

# User-defined
status = [random.random() for _ in range(model.n)]
model.set_initial_status(kind='user_defined', status=status)
```

### 3. Run Simulations

Run the simulation for a fixed number of iterations:

```python
results = model.iteration_bunch(T=100)
```

Or until a steady state is reached:

```python
results = model.steady_state(max_iterations=1000, nsteady=50, sensibility=0.0001)
```

### 4. Save Results

Save the system's state after simulation:

```python
model.save_all_status(save_dir="./results", filename="simulation_results", format="json")
model.save_final_state(save_dir="./results", filename="final_state", format="csv")
```

### 5. Visualize Results

Generate plots to analyze the simulation:

```python
# Plot opinion evolution
model.plot_evolution()

# Visualize initial weights
model.visualize_initial_weights()

# Visualize final probabilities
model.visualize_final_probabilities()
```

## Model Parameters

| Parameter     | Description                                  | Range   |
|---------------|----------------------------------------------|---------|
| `p_o`         | Probability of an optimistic event           | [0, 1]  |
| `p_p`         | Probability of a pessimistic event           | [0, 1]  |
| `alpha`       | Tuning parameter for the lambda function     | [0, 1]  |

## Authors

- **Alina Sirbu** - [alina.sirbu@unipi.it](mailto:alina.sirbu@unipi.it)
- **Giulio Rossetti** - [giulio.rossetti@isti.cnr.it](mailto:giulio.rossetti@isti.cnr.it)
- **Valentina Pansanella** - [valentina.pansanella@isti.cnr.it](mailto:valentina.pansanella@isti.cnr.it)


## Simulator usage
Run the simulation with:
```bash
python simulations.py
```

### Example Configuration
Customize parameters in `simulations.py`:
```python
model_pars = {
    "po": 0.01,                 # Probability of opinion exchange
    "pp": 0.99,                 # Probability of persuasion
    "alpha": 0.2,               # Influence weight parameter
    "graphtype": "ER",          # Network type (e.g., ER for Erdos-Renyi)
    "N": 1000,                  # Number of agents
    "per": 0.01,                # Edge creation probability in ER graph
    "nlobbyists": 2,            # Number of lobbyist agents
    "initial_distr": "uniform", # Initial opinion distribution
    "T": 100                    # Number of time steps
}
```

### Output Directory
Results are stored in `results/` with subdirectories for each run, e.g.:
```
results/
  po0.01_pp0.99_alpha0.2_graphtypeER_N1000_per0.01_nlobbyists2_initial_distruniform_T100/
    run0/
      graph.edgelist
      plots/
        evolution.png
        initial.png
        final.png
        final_probabilities.png
      status.json
      final.csv
    run1/
      ...
```

---

## Parameters
### Model Parameters
| Parameter       | Description                                                                 | Default  |
|------------------|-----------------------------------------------------------------------------|----------|
| `po`            | Probability of opinion exchange between nodes.                             | `0.01`   |
| `pp`            | Probability of persuasion by interacting agents.                           | `0.99`   |
| `alpha`         | Weight parameter for influence dynamics.                                    | `0.2`    |
| `graphtype`     | Type of graph (`ER` for Erdos-Renyi).                                        | `"ER"`   |
| `N`             | Number of agents (nodes in the graph).                                      | `1000`   |
| `per`           | Edge creation probability for ER graphs.                                    | `0.01`   |
| `nlobbyists`    | Number of lobbyist agents influencing the network.                          | `2`      |
| `initial_distr` | Initial distribution of opinions (`uniform` or custom).                     | `"uniform"` |
| `T`             | Number of time steps for the simulation.                                    | `100`    |

### Simulation Run Parameters
| Parameter  | Description                                   | Default |
|------------|-----------------------------------------------|---------|
| `nruns`    | Number of independent simulation runs.        | `10`    |
| `save_dir` | Directory to save results and visualizations. | -       |

---

## Outputs
For each run, the following are generated:
1. **Graph Edgelist:** Saved in CSV format.
2. **Simulation Status:** Saved in JSON and CSV formats.
3. **Visualizations:** Includes:
   - **Evolution Plot:** Tracks opinion dynamics over time.
   - **Initial Weights:** Distribution of initial opinion weights.
   - **Final Weights:** Distribution of final opinion weights.
   - **Final Probabilities:** Visualization of agent opinion probabilities.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


