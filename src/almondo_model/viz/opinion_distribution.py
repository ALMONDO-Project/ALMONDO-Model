import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class OpinionDistribution(object):
    def __init__(self, 
                 trends: dict, 
                 p_o,
                 p_p,
                 iteration: int | str = -1, 
                 values: str = "probabilities"):
        
        """
        :param model: The model object
        :param trends: The computed simulation trends
        :param iteration: The iteration number or the string "last" for plotting final state
        :param values: The type of values to extract ("probabilities" or "weights").
        """
        
        self.trends = trends
        self.iteration = iteration
        
        if iteration == 'last':
            self.it = self.trends[-1]['iteration']
            self.ops = self.trends[-1]['status']
        else:
            self.ops = self.trends[iteration]['status'].values()
        
        if values == 'probabilities':
            weights = np.array([el for el in self.ops])
            self.values = p_o * weights + p_p * (1-weights) # opt model
            # self.values = p_o * (1- weights) + p_p * weights  # pess model
            
        elif values == 'weights':
            self.values = np.array([el for el in self.ops])
    
    def get_values(self):
        return self.values

    def plot(self, filename=None, ax = None, values: str = "probabilities", 
             stat: bool = True, title: bool = True, 
             transparent_bg: bool = False, transparent_plot_area: bool = False):
        """
        This method plots the distribution of the final agent values.
        Arguments:
        - filename: The file path to save the plot, if None, it will show the plot instead.
        - ax: The matplotlib axis to plot on, if None, it creates a new one.
        - values: The type of values to plot ("probabilities" or "weights").
        - stat: If True, it shows the mean and standard deviation on the plot.
        - title: If True, it adds a title to the plot.
        - transparent_bg: If True, the background of the figure will be transparent.
        - transparent_plot_area: If True, the plot area will have a transparent background.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))    
        data = self.get_values()
        
        # Check if all values are the same (or nearly the same) to properly adjust bin width
        if np.std(data) < 1e-6:  # Very small standard deviation means essentially constant
            # For constant data, create custom bins with desired width to avoid a single line
            unique_val = data[0]
            bin_width = 0.015  # Adjust this to control bar width
            bins = [unique_val - bin_width/2, unique_val + bin_width/2]
        else:
            # For distributed data, use regular binning
            bins = 50
            
        ax = sns.histplot(data, bins = bins, color='lightblue', alpha=1.0, stat='percent',ax=ax)
        # Set figure and plot area background
        if transparent_bg: # figure background
            fig.patch.set_facecolor('none')
        else:
            fig.patch.set_facecolor('white')

        if transparent_plot_area: # plot area background
            ax.set_facecolor('none')
        else:
            ax.set_facecolor('white')  # default value
        
        ax.set_xlabel(f'Final agent {values}'r' $p_{i,T}$', fontsize=12)
        ax.set_ylabel('% agents',fontsize=12)
        ax.set_xlim(0.0, 1.0)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        if title:
            ax.set_title(f'Final {values} distribution of optimist model',fontsize=14)
        
        if stat:
            ax.text(
                0.99,
                0.95,
                ("Mean: %.2f; Std: %.2f" % (np.mean(self.get_values()), np.std(self.get_values()))).lstrip("0"),
                transform=ax.transAxes,
                size=11,
                horizontalalignment="right"
            )
        plt.tight_layout()

        if filename is not None:
            bg_color = 'none' if transparent_bg else 'white'
            plt.savefig(filename, dpi=300, facecolor=bg_color, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()