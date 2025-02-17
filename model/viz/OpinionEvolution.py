import numpy as np
import matplotlib.pyplot as plt
import future.utils

class OpinionEvolution(object):
    def __init__(self, trends, p_o, p_p):
        
        """
        :param p_o: The model p_o parameter
        :param p_p: The model p_p parameter
        :param trends: The computed simulation trends (status.json in the run folder)
        """
        
        print(type(trends))
        print(trends)
        
        self.node2col = {}
        self.nodes2opinions = {}
            
        self.last_it = trends[-1]['iteration'] + 1
        self.last_seen = {}
        
        def transform(w: list, p_o: int, p_p: int):
            w = np.array(w)
            p = w * p_o + (1 - w) * p_p
            p = p.tolist()
            return p

        for it in trends:
            weights = np.array([el for el in it['status'].values()])
            sts = transform(weights, p_o, p_p)  # update conditional probabilities of event will occur
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
    
    def plot(self, filename=None, ax = None):   
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))    
        mx = 0
        for k, l in future.utils.iteritems(self.nodes2opinions):
            if mx < self.last_seen[k]:
                mx = self.last_seen[k]
            x = list(range(0, self.last_seen[k]))
            y = l[0:self.last_seen[k]]
            ax.plot(x, y, lw=1.5, alpha=0.5, color=self.node2col[k])
        plt.xlabel('t')
        plt.xticks(range(0, mx, 1))
        plt.ylabel(r'$p_{i,t}$')
        plt.title('Optimist model probability evolution')
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight', facecolor='white')
        plt.close()