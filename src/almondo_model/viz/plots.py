## Example to generate opinion distribution and opinion evolution plots
import json
import os 
from tqdm import tqdm  # Use tqdm for Jupyter Notebook
from opinion_distribution import OpinionDistribution
from opinion_evolution import OpinionEvolution

values = ['weights', 'probabilities']

nl = 0
print(f'doing {nl} lobbyists')

path = f'../results/balanced_budgets/{nl}_lobbyists/'
filename = os.path.join(path, 'config.json')
    
with open(filename, 'r') as f: #qua va messo il path del file initial_config.json
    params = json.load(f)

total_iterations = len(values) * len(params['lambda_values']) * len(params['phi_values']) * params['nruns']
with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:

    for value in values:
        for _, (lambda_v, phi_v) in enumerate([(l, p) for l in params['lambda_values'] for p in params['phi_values']]):    
            paramspath = os.path.join(path, f'{lambda_v}_{phi_v}/')        
            for run in range(0,4): #range(params['nruns']):
                runpath = os.path.join(paramspath, str(run))
                if not os.path.exists(runpath+f'/{value}_final_distribution.png'):
                    with open(runpath+'/status.json', 'r') as f:
                        trends = json.load(f)
                    od = OpinionDistribution(trends, params['p_o'], params['p_p'], values=value)
                    od.plot(runpath+f'/{value}_final_distribution.png',values=value, 
                            stat=True, title=True, transparent_bg=False, transparent_plot_area=False)
                if not os.path.exists(runpath+f'/{value}_evolution.png'):
                    with open(runpath+'/status.json', 'r') as f:
                        trends = json.load(f)    
                    oe = OpinionEvolution(trends, params['p_o'], params['p_p'],kind=value)
                    oe.plot(runpath+f'/{value}_evolution.png')
                pbar.update(1)