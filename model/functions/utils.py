import numpy as np

def transform(w: list, p_o, p_p):
    w = np.array(w)
    p = w * p_o + (1 - w) * p_p # optimistic model
    # p = (1-w) * p_o + w * p_p # pessimistic model
    p = p.tolist()
    return p