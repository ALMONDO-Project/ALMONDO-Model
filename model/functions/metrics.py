import numpy as np
from scipy import stats

def nclusters(opinions, threshold):
    opinions = [float(el) for el in opinions]
    opinions = sorted(opinions)
    start = opinions[0]
    max_val = start + threshold
    c = (start, max_val)
    cluster = dict()
    for o in opinions:
        if o <= max_val:
            if c in cluster.keys():
                cluster[c] += 1
            else:
                cluster[c] = 1
        else:
            max_val = o + threshold
            c = (o, max_val)
            cluster[c] = 1
    C_num = len(opinions)**2
    C_den = 0
    for k in cluster.keys():
        C_den += cluster[k]*cluster[k]
    C = C_num / C_den
    return C

def entropy(opinions, n, nbins):
    bincounts, _ = np.histogram(opinions, bins = np.linspace(0, 1, nbins))
    probabilities = bincounts/n
    entr = stats.entropy(probabilities)
    return entr

def pwdist(opinions):
    distances = []
    for i in range(len(opinions)):
        x_i = opinions[i]
        distances_i = []
        for j in range(len(opinions)):
            if i != j:
                x_j = opinions[j]
                d_ij = abs(x_i-x_j)
                distances_i.append(d_ij)
        avg_i = sum(distances_i)/len(distances_i)
        distances.append(avg_i)
    return sum(distances)/len(distances)

def nits(status):
    return int(status[-1]['iterations'])


# -*- coding: utf-8 -*-
"""
This function computes the expected average relative entropy of final beliefs 
with respect to the model the lobbyist supports.
This can be considered as a performance index for the lobbyist with respect 
to its communication strategy.
 
# Author information
__author__ = ["Verdiana Del Rosso"]
__email__ = [
    "verdiana.delrosso@unicam.it",
]
"""

def lobbyist_performance(opinions, model, p_o, p_p):
    
    """ 
    Perform relative entropy of final beliefs 
    with respect to the model the lobbyist supports.
    
    Parameters:
    w: agents' weights
    model: model the lobbyist supports - remember that m = 1 is optimistic and m = 0 is pessimistic 
    p_o: probability of the optimistic model
    p_p: probability of the pessimistic model
    
    Returns:
    strategy_performance: Relative entropy of final beliefs
    """
    
    if model == 0:
        p_lob = p_p
    elif model == 1:
        p_lob = p_o
        
    p = np.array(opinions) * p_o + (1 - np.array(opinions)) * p_p # final beliefs of agents
    
    rel_entropy = p_lob*np.log(p_lob/p)+(1-p_lob)*np.log((1-p_lob)/(1-p))
    
    #strategy_performance = sum(rel_entropy)/N # mean
    strategy_performance = np.mean(rel_entropy) # mean
    
    return strategy_performance  # lobbyist performance index (entropy)

