import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import future.utils

class OpinionEvolution(object):
    def __init__(self, trends, p_o, p_p, kind='probabilities'):
        
        """
        :param p_o: The model p_o parameter
        :param p_p: The model p_p parameter
        :param trends: The computed simulation trends (status.json in the run folder)
        :param values: The type of values to extract ("probabilities" or "weights").
        """
        
        self.kind = kind
        
        self.node2col = {}
        self.nodes2opinions = {}
            
        self.last_it = trends[-1]['iteration'] + 1
        self.last_seen = {}
        
        def transform(w: list, p_o: int, p_p: int):
            w = np.array(w)
            p = w * p_o + (1 - w) * p_p # optimistic model
            # p = (1 - w) * p_o + w * p_p # pessimistic model
            p = p.tolist()
            return p

        for it in trends:
            weights = np.array([el for el in it['status'].values()])
            if kind == 'probabilities':
                sts = transform(weights, p_o, p_p)  # update conditional probabilities of event will occur
            else:
                sts = weights
            its = it['iteration']
            for n, v in enumerate(sts):
                if n in self.nodes2opinions:
                    last_id = self.last_seen[n]
                    last_value = self.nodes2opinions[n][last_id]

                    for i in range(last_id, its):
                        self.nodes2opinions[n][i] = last_value

                    self.nodes2opinions[n][its] = v
                    self.last_seen[n] = its
                else:
                    self.nodes2opinions[n] = [0]*self.last_it
                    self.nodes2opinions[n][its] = v
                    self.last_seen[n] = 0
                    if v < 0.33:
                        self.node2col[n] = '#357db0'
                    elif 0.33 <= v <= 0.66:
                        self.node2col[n] = '#18A558'
                    else:
                        self.node2col[n] = '#ce2626'
    
    def plot(self, filename=None, ax = None,
             figure_size=(10, 6), grid: bool = False,
             transparent_bg: bool = False, transparent_plot_area: bool = False):
        """
        This method plots the evolution of agents' opinions over iterations.
        Arguments:
        - filename: The file path to save the plot, if None, it will show the plot instead and return the figure.
        - ax: The matplotlib axis to plot on, if None, it creates a new one
        - figure_size: The size of the figure.
        - grid: f True, it adds the horizontal grid to the plot.
        - transparent_bg: If True, the background of the figure will be transparent.
        - transparent_plot_area: If True, the plot area will have a transparent background.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figure_size)
        else:
            fig = ax.get_figure()

        mx = 0
        for k, l in future.utils.iteritems(self.nodes2opinions):
            if mx < self.last_seen[k]:
                mx = self.last_seen[k]
            x = list(range(0, self.last_seen[k]+1))
            y = l[0:(self.last_seen[k]+1)]
            ax.plot(x, y, lw=1.5, alpha=0.5, color=self.node2col[k])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # to plot only integer data (iterations cannot be float)
        ax.set_ylim(0.0, 1.0)
        
        # Set figure and plot area background
        if transparent_bg:
            fig.patch.set_facecolor('none')
        else:
            fig.patch.set_facecolor('white')

        if transparent_plot_area:
            ax.set_facecolor('none')
        else:
            ax.set_facecolor('white')  # default value
        
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel(f"Agents' {self.kind.capitalize()}",fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        if grid:
            ax.grid(axis='y')
        
        if filename is not None:
            bg_color = 'none' if transparent_bg else 'white'
            plt.savefig(filename, dpi=300, facecolor=bg_color, bbox_inches='tight')
        else:
            plt.show()
            return fig
            
        plt.close()