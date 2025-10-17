import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

def process_results(paths, metrics_to_plot=None, log_scale_metrics=None):
    """
    Process simulation results and generate aggregate metrics and heatmaps.

    Args:
        paths (list[str]): list of base paths with results.
        metrics_to_plot (list[str], optional): subset of metric columns to plot.
            If None, all metrics found are plotted.
        log_scale_metrics (list[str], optional): metrics to display with log scale.
    """
    if log_scale_metrics is None:
        log_scale_metrics = []

    for basepath in paths:
        config_path = os.path.join(basepath, 'config.json')
        if not os.path.exists(config_path):
            print(f"Missing config.json in {basepath}")
            continue

        with open(config_path, 'r') as f:
            params = json.load(f)

        lambda_values = params['lambda_values']
        gw_values = params.get('gw_values', [0.5])
        n_lobbyists = params['n_lobbyists']
        nruns = params['nruns']
        lobbyists_data = params['lobbyists_data']

        kinds = ['weights', 'probabilities']
        data = []

        for folder in os.listdir(basepath):
            folder_path = os.path.join(basepath, folder)
            try:
                lambda_v = float(folder.split('_')[0])
                gw_v = float(folder.split('_')[1])
            except:
                continue

            for kind in kinds:
                metrics_file = os.path.join(folder_path, f'{kind}_average_metrics.json')
                if not os.path.exists(metrics_file):
                    continue

                with open(metrics_file, 'r') as f:
                    avg_metrics = json.load(f)

                columns = ['kind', 'n_lobbyists', 'lambda', 'gw', 'nruns']
                values = [kind, n_lobbyists, lambda_v, gw_v, nruns]

                for metric_name, metric_data in avg_metrics.items():
                    if metric_name == 'lobbyists_performance':
                        for id in lobbyists_data.keys():
                            values.append(metric_data[str(id)]['avg'])
                            columns.append(f'avg_{metric_name}_{id}')
                    else:
                        values.append(metric_data['avg'])
                        columns.append(f'avg_{metric_name}')

                row = dict(zip(columns, values))
                data.append(row)

        df = pd.DataFrame(data)
        out_csv = os.path.join(basepath, 'aggregate_metrics.csv')
        df.to_csv(out_csv, index=False)
        print(f"Saved aggregate metrics to {out_csv}")

        # === Heatmap function ===
        def heatmap(kind, metric, log_scale=False, figname=None):
            df_filtered = df[df["kind"] == kind]
            heatmap_data = df_filtered.pivot(index="lambda", columns="gw", values=metric)

            if heatmap_data.empty:
                return

            plt.figure(figsize=(8, 6))
            if metric == 'avg_average_opinions':
                ax = sns.heatmap(heatmap_data, vmin=0, vmax=1,
                                 cmap="Blues", annot=True, annot_kws={"size": 4}, fmt=".2f")
            elif metric in log_scale_metrics and log_scale:
                ax = sns.heatmap(heatmap_data, cmap="Blues",
                                 annot=True, annot_kws={"size": 4}, fmt=".0f",
                                 norm=LogNorm(vmin=1))
            else:
                ax = sns.heatmap(heatmap_data, cmap="Blues",
                                 annot=True, annot_kws={"size": 4}, fmt=".2f")

            plt.title(f"Heatmap of {metric} for {kind}")
            plt.xlabel("Share of budget devoted to reputational investment (gw)", fontsize=11)
            plt.ylabel("Degree of under-reaction (λ)", fontsize=11)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=11)

            figpath = os.path.join(basepath, 'figures')
            os.makedirs(figpath, exist_ok=True)
            plt.savefig(os.path.join(figpath, figname), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        # === Generate plots ===
        if metrics_to_plot is None:
            metrics_to_plot = [col for col in df.columns if col.startswith("avg")]

        for kind in ['probabilities', 'weights']:
            if kind in df['kind'].unique():
                for col in metrics_to_plot:
                    if col in df.columns:
                        figname = f'heatmap_{kind}_{col}.png'
                        heatmap(kind, col, log_scale=True, figname=figname)
                        print(f"Saved {figname} to {os.path.join(basepath, 'figures')}")
