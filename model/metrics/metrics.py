from typing import List, Literal
from typing import List, Union
import numpy as np

def nclusters(opinions: Union[List[float], np.ndarray], threshold: float = 0.01) -> float:
    """
    Computes the effective number of clusters based on a threshold.

    Args:
        opinions (list | numpy array): List or array of opinion values.
        threshold (float, optional): Threshold to bin opinions into clusters. Defaults to 0.01.

    Returns:
        float: Effective number of clusters.

    Raises:
        ValueError: If opinions contains fewer than 2 values.
    """
    
    # # Validate input type
    # if not isinstance(opinions, (list, np.ndarray)) or not all(isinstance(o, (int, float)) for o in opinions):
    #     raise TypeError("opinions must be a list or numpy array of numerical values.")

    # Ensure there are at least two opinions
    if len(opinions) < 2:
        raise ValueError("At least two opinions are required to compute clusters.")

    # Convert to numpy array for efficiency
    opinions = np.sort(opinions)

    # Initialize clustering
    clusters = []
    start = opinions[0]
    max_val = start + threshold
    count = 0

    # Group into clusters
    for o in opinions:
        if o <= max_val:
            count += 1
        else:
            clusters.append(count)
            count = 1
            max_val = o + threshold
    clusters.append(count)  # Add last cluster count

    # Compute effective number of clusters
    C_num = len(opinions) ** 2
    C_den = sum(c ** 2 for c in clusters)
    return C_num / C_den

def pwdist(opinions: Union[List[float], np.ndarray]) -> float:
    """
    Computes the average pairwise absolute distance of opinion values.

    Args:
        opinions (list or numpy array): List or array of opinion values.

    Returns:
        float: The average pairwise distance.

    Raises:
        ValueError: If opinions contains less than 2 values.
    """

    # Ensure there are at least two opinions to compute distances
    if len(opinions) < 2:
        raise ValueError("At least two opinions are required to compute pairwise distances.")

    # Compute pairwise absolute differences using numpy broadcasting
    dist_matrix = np.abs(opinions[:, None] - opinions)
    np.fill_diagonal(dist_matrix, np.nan)  # Ignore self-distances

    # Compute the mean pairwise distance
    return np.nanmean(dist_matrix)


def lobbyist_performance(opinions: List[float], model: Literal[0, 1], p_o: float, p_p: float):
    """
    Computes the expected average relative entropy of final beliefs 
    with respect to the model the lobbyist supports.

    Parameters:
    - opinions: List of opinion values (between 0 and 1)
    - model: The model the lobbyist supports (0 = pessimistic, 1 = optimistic)
    - p_o: Probability of the optimistic model
    - p_p: Probability of the pessimistic model

    Returns:
    - strategy_performance: A performance index (float)
    
    Raises:
    - ValueError: If model is not 0 or 1
    """
    
    # Validate model
    if not isinstance(model, int) or model not in (0, 1):
        raise ValueError(f"Invalid value for 'model': {model}. Expected 0 or 1.")
    
    # # Validate opinions
    # if not isinstance(opinions, list or np.ndarray) or not all(isinstance(o, (np.float64)) for o in opinions):
    #     print(opinions)
    #     print([type(el) for el in opinions])
    #     raise TypeError("opinions must be a list of numerical values.")
    
    # Ensure probabilities are valid
    if not (0 <= p_o <= 1 and 0 <= p_p <= 1):
        raise ValueError("p_o and p_p must be between 0 and 1.")
    
    # Assign probability based on model
    p_lob = p_p if model == 0 else p_o

    rel_entropy = p_lob * np.log((p_lob) / (opinions)) + \
                  (1 - p_lob) * np.log((1 - p_lob) / (1 - opinions))

    strategy_performance = np.mean(rel_entropy)

    return strategy_performance
