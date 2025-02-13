from classes.metrics import AverageMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import pandas as pd

def main(scenario, n_lobbyists):
    SCENARIO = scenario
    scenario_path = os.path.join("simulations", SCENARIO)
    
    settings = {
        'p_o': 0.01,
        'p_p': 0.99,
        'initial_distribution': 'uniform',
        'path': scenario_path,
        'T': 10000,
        'n_lobbyists': n_lobbyists,    # SINGLE LOBBYIST!
        'ms': None,
        'strategies': None
    }
    
    NRUNS = 100

    graphparams = {
        'N': 500,
        'type': 'complete'
    }

    N = graphparams['N']
    graph = nx.complete_graph(N)

    # Define the possible values of lambdas and phis to test 
    lambda_values = [0.0, 0.5, 1.0]
    phi_values = [0.0, 0.5, 1.0]
    
    data = []
    
    for _, (lambda_v, phi_v) in enumerate([(l, p) for l in lambda_values for p in phi_values]):
            
            configparams = {
                'lambdas': lambda_v,
                'phis': phi_v
            }
            
            for kind in ['weights', 'probabilities']:    
                
                pars = {
                    'nruns': NRUNS, 
                    'kind': kind,
                    'graph': graph,
                    'initial_distribution': settings['initial_distribution'],
                    'T': settings['T'],
                    'p_o': settings['p_o'],
                    'p_p': settings['p_p'],
                    'lambdas': configparams['lambdas'],
                    'phis': configparams['phis'],
                    'n_lobbyists': settings['n_lobbyists'],
                    'ms': settings['ms'],
                    'strategies': settings['strategies']
                }    
                
                config_path = os.path.join(f"simulations/{scenario}/{lambda_v}_{phi_v}/")
                os.makedirs(config_path, exist_ok=True)  
                
                am = AverageMetrics(**pars, path = config_path)
                am = am.compute()
                metrics = am.get_results()

                row = {"lambda": lambda_v, "phi": phi_v, "kind": kind}
                
                for metric, values in metrics.items():
                    if isinstance(values, dict) and "avg" in values and "std" in values:
                        row[f"avg_{metric}"] = values["avg"]
                        row[f"std_{metric}"] = values["std"]
                
                data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_file = os.path.join(scenario_path, "metrics.csv")
    df.to_csv(output_file, index=False)

    print(f"Aggregated metrics saved to {output_file}")
    
    
    # Generate heatmaps and line plots
    for kind in ['weights', 'probabilities']:
        df_kind = df[df['kind'] == kind]
        metrics = [col for col in df_kind.columns if col.startswith("avg_")]
        
        for metric in metrics:
            metric = metric.replace('avg_', '')
            pivot_table = df_kind.pivot(index="lambda", columns="phi", values=f'avg_{metric}')
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title(f"Heatmap of avg_{metric} for {kind}")
            plt.xlabel("phi")
            plt.ylabel("lambda")
            plt.savefig(os.path.join(scenario_path, f"heatmap_{kind}_avg_{metric}.png"))
            plt.close()
            
            plt.figure(figsize=(8, 6))
            for phi_val in df_kind["phi"].unique():
                subset = df_kind[df_kind["phi"] == phi_val]
                plt.errorbar(subset["lambda"], subset[f"avg_{metric}"], yerr=subset[f"std_{metric}"], label=f"phi={phi_val}", marker='o')
            
            plt.title(f"Line plot of {metric} with error bars for {kind}")
            plt.xlabel("lambda")
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(os.path.join(scenario_path, f"lineplot_{phi_val}_{kind}_{metric}.png"))
            plt.close()
            
                
                
if __name__ == "__main__":
    main("0_lobbyists", 0)