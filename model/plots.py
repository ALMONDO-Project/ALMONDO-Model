from ndlib.models.DiffusionModel import DiffusionModel
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
import numpy as np
import networkx as nx
import gzip
import pickle
import os
import seaborn as sns
import future.utils
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def plotevolution(iterations, p_o, p_p, save_dir):
    spaghetti_hex_list = ['#357db0', '#18A558', '#ce2626']
    spaghetti_cmap=get_continuous_cmap(spaghetti_hex_list)
    
    fig, ax = plt.subplots()

    spaghetti_hex_list = ['#357db0', '#18A558', '#ce2626']


    """
    Generates the plot

    :param filequintet: Output filequintet
    :param percentile: The percentile for the trend variance area
    """

    nodes2opinions = {}
    node2col = {}

    last_it = iterations[-1]['iteration'] + 1
    last_seen = {}

    for it in iterations:
        weights = it['status']
        sts = p_o * weights + p_p * (1-weights)  # update conditional probabilities of event will occur
        its = it['iteration']
        for n, v in enumerate(sts):
            if n in nodes2opinions:
                last_id = last_seen[n]
                last_value = nodes2opinions[n][last_id]

                for i in range(last_id, its):
                    nodes2opinions[n][i] = last_value

                nodes2opinions[n][its] = v
                last_seen[n] = its
            else:
                nodes2opinions[n] = [0]*last_it
                nodes2opinions[n][its] = v
                last_seen[n] = 0
                if v < 0.33:
                    node2col[n] = spaghetti_hex_list[0]
                elif 0.33 <= v <= 0.66:
                    node2col[n] = spaghetti_hex_list[1]
                else:
                    node2col[n] = spaghetti_hex_list[2]

    mx = 0
    for k, l in future.utils.iteritems(nodes2opinions):
        if mx < last_seen[k]:
            mx = last_seen[k]
        x = list(range(0, last_seen[k]))
        y = l[0:last_seen[k]]
        ax.plot(x, y, lw=1.5, alpha=0.5, color=node2col[k])
    plt.xlabel('t')
    plt.ylabel(r'$p_{i,t}$')
    plt.title('Optimist model probability evolution')
    plt.tight_layout()
    file_name = os.path.join(save_dir, f'evolution_{run}.png')
    plt.savefig(file_name, bbox_inches='tight', facecolor='white')
    plt.close()

def weights_plot(iterations, save_dir):
    weights = iterations[-1]['status']
    ax = sns.histplot(weights, bins=50, color='lightblue', alpha=1.0)
    ax.set_xlabel(r'$w_{i,T}$')
    ax.set_ylabel('% agents')
    ax.set_title('Final weights distribution of optimist model')
    plt.tight_layout()
    file_name = os.path.join(save_dir, f'finalweights_{run}.png')
    plt.savefig(file_name, bbox_inches='tight', facecolor='white')
    
def probabilites_plot(iterations, p_o, p_p, save_dir):
    weights = iterations[-1]['status']
    probabilities = p_o * weights + p_p * (1-weights)  # update conditional probabilities of event will occur
    ax = sns.histplot(probabilities, bins = 50, color='lightblue', alpha=1.0, stat='percent')
    # sns.kdeplot(probabilities, color='tab:blue', cut=0, ax=ax)
    ax.set_xlabel(r'$p_{i,T}$')
    ax.set_ylabel('% agents')
    ax.set_title('Final probability distribution of optimist model')
    ax.set_xlim(0.0, 1.0)
    plt.tight_layout()
    file_name = os.path.join(save_dir, f'finalprobabilities_{run}.png')
    plt.savefig(file_name, bbox_inches='tight', facecolor='white')

nruns = 100
save_dir = 'results/'

pars = [0.01, 0.99, 0.8, 1000, 0.01]

g = nx.erdos_renyi_graph(pars[3],pars[4])
initial_status = np.random.rand(pars[3])

folder = '_'.join([str(e) for e in pars])
save_dir = os.path.join(save_dir, folder)

for run in range(nruns):
    file_name = os.path.join(save_dir, f'iterations_{run}')
    with gzip.open(file_name, 'rb') as f:
        iterations = pickle.load(f)
    weights_plot(iterations, save_dir)
    probabilites_plot(iterations, pars[0], pars[1], save_dir)
    plotevolution(iterations, pars[0], pars[1], save_dir)


