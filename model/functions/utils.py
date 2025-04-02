import numpy as np

def transform(w: list, p_o, p_p):
    w = np.array(w)
    # p = w * p_o + (1 - w) * p_p
    p = (1-w) * p_o + w * p_p
    p = p.tolist()
    return p