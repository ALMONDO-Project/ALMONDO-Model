import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
import numpy as np
import random
import os
import json

RUNS = 100

def create_graph(n, density, directed=True):
    g = nx.erdos_renyi_graph(n, density, directed=directed)
    return g

def init_model(g, p_o = 0.01, p_p = 0.99, l=0.8, seed = 4):
    # Model selection
    model = op.AlmondoModel(g, seed)
    config = mc.Configuration()
    config.add_model_parameter("p_o", p_o)
    config.add_model_parameter("p_p", p_p)
    config.add_model_parameter("l", l)
    model.set_initial_status(configuration=config)
    return model

def create_folders(model, datapath='data/out/model/'):
    params = model.get_model_parameters()
    print(params)
    folder_name = 'runs'
    for k,v in params.items():
        newpair = f'{k}{str(v)}'
        folder_name += f'-{newpair}'
    print(folder_name)
    path = os.path.join(datapath, folder_name)
    print(path)
    # Split the path into individual directories
    directories = path.split('/')
    # Start building the path from the root
    current_path = os.getcwd()
    # Iterate over each directory in the path
    for directory in directories:
        current_path = os.path.join(current_path, directory)
        # Check if the directory exists
        if not os.path.exists(current_path):
            # If it doesn't, create it
            os.makedirs(current_path)
    return path
            
def execute(model, max_iterations=1000000, nsteady=1000, sensibility=0.00001):
    return model.steady_state(max_iterations, nsteady, sensibility)

def jsonify(iterations):
    iterations_to_save = []
    for iteration in iterations:
        new_iteration = {}
        for key, value in iteration.items():
            if isinstance(value, np.ndarray):
                l = value.tolist()
                new_iteration[key] = l
            else:
                new_iteration[key] = value
        iterations_to_save.append(new_iteration)
    return iterations_to_save
    
def save(iterations, run, path):
    iterations = jsonify(iterations)
    with open(f'{path}/{run}.json', 'w', encoding='utf-8') as file:
        json.dump(iterations, file, indent=4)

def done(model, datapath='data/out/model', run=-1):
    params = model.get_model_parameters()
    print(params)
    folder_name = 'runs'
    for k,v in params.items():
        newpair = f'{k}{str(v)}'
        folder_name += f'-{newpair}'
    print(folder_name)
    path = os.path.join(datapath, folder_name)
    path = os.path.join(path, f'{run}.json')
    
    return os.path.exists(path)
    
    
    
def main():
    params_combinations= [
     {'p_o': 0.01, 'p_p': 0.99, 'l': 0.0}, 
     {'p_o': 0.01, 'p_p': 0.99, 'l': 0.5},
     {'p_o': 0.01, 'p_p': 0.99, 'l': 1.0},
     {'p_o': 0.1, 'p_p': 0.9, 'l': 0.0},
     {'p_o': 0.1, 'p_p': 0.9, 'l': 0.5},
     {'p_o': 0.1, 'p_p': 0.9, 'l': 1.0},
     {'p_o': 0.2, 'p_p': 0.8, 'l': 0.0},
     {'p_o': 0.2, 'p_p': 0.8, 'l': 0.5},
     {'p_o': 0.2, 'p_p': 0.8, 'l': 1.0}, 
     {'p_o': 0.4, 'p_p': 0.6, 'l': 0.0},
     {'p_o': 0.4, 'p_p': 0.6, 'l': 0.5},
     {'p_o': 0.4, 'p_p': 0.6, 'l': 0.9},
     {'p_o': 0.5, 'p_p': 0.5, 'l': 0.0},
     {'p_o': 0.5, 'p_p': 0.5, 'l': 0.5},
     {'p_o': 0.5, 'p_p': 0.5, 'l': 0.9}
     ]
    
    for params in params_combinations:
        p_o = params['p_o']
        p_p = params['p_p']
        l = params['l']
        for i in range(RUNS):
            g = create_graph(1000, 0.01, True)
            model = init_model(g, p_o=p_o, p_p=p_p, l=l)
            if done(model, run=i): 
                continue
            path = create_folders(model)
            iterations = execute(model)      
            save(iterations, i, path)                 

if __name__ == '__main__':
    main()   