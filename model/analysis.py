from classes.metrics import AverageMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import json
import pandas as pd

n_lobbyists = 3

data = []

for _, (lambda_v, phi_v) in enumerate([(l, p) for l in [0.0, 0.5, 1.0] for p in [0.0, 0.5, 1.0]]):
        
        for kind in ['weights', 'probabilities']:    
            
            am = AverageMetrics(
                nruns = 100, 
                kind = kind, 
                
                p_o = 0.01,
                p_p = 0.99,
                
                n_lobbyists = n_lobbyists,
                ms = [1, 0, 1],
                
                data = None,
                path = f'old_simulations/{n_lobbyists}_lobbyists/{lambda_v}_{phi_v}/',    
            )
            
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
output_file = os.path.join(f'old_simulations/{n_lobbyists}_lobbyists/', "metrics.csv")
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
        plt.savefig(os.path.join(f'old_simulations/{n_lobbyists}_lobbyists/', f"heatmap_{kind}_avg_{metric}.png"))
        plt.close()
        
        plt.figure(figsize=(8, 6))
        for phi_val in df_kind["phi"].unique():
            subset = df_kind[df_kind["phi"] == phi_val]
            plt.errorbar(subset["lambda"], subset[f"avg_{metric}"], yerr=subset[f"std_{metric}"], label=f"phi={phi_val}", marker='o')
        
        plt.title(f"Line plot of {metric} with error bars for {kind}")
        plt.xlabel("lambda")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(f'old_simulations/{n_lobbyists}_lobbyists/', f"lineplot_{phi_val}_{kind}_{metric}.png"))
        plt.close()