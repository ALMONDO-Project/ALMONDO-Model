from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import future.utils
import seaborn as sns
import json

class OpinionDistribution(object):
    def __init__(self, model, trends, iteration='last', values = 'probabilities'):
        """
        :param model: The model object
        :param trends: The computed simulation trends
        """
        self.system_status = trends
        self.model = model
        self.trends = trends
        self.iteration = iteration
        
        if iteration == 'last':
            self.it = -1
            self.it = self.trends[self.it]['iteration']
            self.ops = self.trends[self.it]['status']
        else:
            self.it = iteration
            self.ops = self.trends[self.it]['status'].values()
        
        if values == 'probabilities':
            weights = np.array([el for el in self.ops])
            self.values = self.model.params['model']['p_o'] * weights + self.model.params['model']['p_p'] * (1-weights)
        elif values == 'weights':
            self.values = np.array([el for el in self.ops])
        
        
    def plot(self, filename=None, ax = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))       
        sns.histplot(self.values, color='lightblue', alpha=1.0, stat='percent')
        plt.xlabel(r'$p_{i,T}$')
        plt.ylabel('% agents')
        plt.title('Final probability distribution of optimist model')
        plt.xlim(0.0, 1.0)
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight', facecolor='white')
        plt.close()