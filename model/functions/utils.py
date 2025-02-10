import numpy as np

def transform(w: list, settings: dict):
    w = np.array(w)
    p = w * settings['p_o'] + (1 - w) * settings['p_p']
    p = p.tolist()
    return p