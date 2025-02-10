from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import future.utils
import seaborn as sns
import json

class OpinionDistribution(object):
    def __init__(self, 
                 model: object, 
                 trends: dict, 
                 iteration: int | str, 
                 values: str):
        
        """
        :param model: The model object
        :param trends: The computed simulation trends
        :param iteration: The iteration number or the string "last" for plotting final state
        """
        
        self.system_status = trends
        self.model = model
        self.trends = trends
        self.iteration = iteration
        
        if iteration == 'last':
            self.it = self.trends[-1]['iteration']
            self.ops = self.trends[-1]['status']
        else:
            self.ops = self.trends[iteration]['status'].values()
        
        if values == 'probabilities':
            weights = np.array([el for el in self.ops])
            self.values = self.model.params['model']['p_o'] * weights + self.model.params['model']['p_p'] * (1-weights)
        elif values == 'weights':
            self.values = np.array([el for el in self.ops])
        
        
    def plot(self, filename=None, ax = None):        
        plt.hist(self.values(), bins=30, edgecolor='black')
        plt.xlabel(r'$p_{i,T}$')
        plt.ylabel('% agents')
        plt.title('Final probability distribution of optimist model')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 100.0)
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight', facecolor='white')
        plt.close()