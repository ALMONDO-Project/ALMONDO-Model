import random
import numpy as np
import os
from tqdm import tqdm

# def generate_ms(n_lobbyists: int) -> list[int]:
#     if n_lobbyists > 0:
#         return [nl % 2 for nl in range(n_lobbyists)] #numero "ugale" di lobbisti ottimisti e pessimisti (se il totale è pari, altrimenti c'è un 1 in più)
#     else:
#         return None
    
# def generate_strategies(folder_path: str,
#                         n_possibilites: int, 
#                         N: int,         # numero nodi 
#                         T: int = 3000,  # timestep totali 
#                         B: int = 300000, # budget
#                         c: int = 1
#                         ) -> None:
    
#     print('Gnerating strategies')
    
#     inter_per_time = B // (c * T)
#     for i in tqdm(range(n_possibilites)):
#         m = np.zeros((T, N), dtype=int)
#         for t in range(T):
#             indices = np.random.choice(N, inter_per_time, replace=False)
#             m[t, indices] = 1
        
#         np.savetxt(os.path.join(folder_path, f"strategy_{i}.txt"), m, fmt="%i")
    
#     print('Strategies generated and saved')
    
#     return


# def read_random_strategies(strategies_path: str, n_lobbyists: int) -> list[np.ndarray]:
#     print(f'strategies path = {strategies_path}')
#     strategy_names = random.sample(os.listdir(strategies_path), n_lobbyists)
#     strategies = []
#     for strategy_name in strategy_names:
#         filepath = os.path.join(strategies_path, strategy_name)
#         print(f"Reading f{filepath}")
#         strategies.append(np.loadtxt(filepath).astype(int))
#     return strategies


# def read_random_strategy(strategies_path: str) -> np.ndarray:
#     print(f'Strategies path = {strategies_path}')
#     strategy_name = random.choice(os.listdir(strategies_path))
#     filepath = os.path.join(strategies_path, strategy_name)
#     print(f"Reading {filepath}")
#     return np.loadtxt(filepath).astype(int), strategy_name

def create_single_random_strategy(B: int, # total budget
                                    T: int, # total time steps
                                    N: int, # number of agents
                                    c: int =1 # cost of a signal
                                    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Create the strategy matrix TxN and randomly selects B/c signals in the TxN matrix to set equals to 1.

    Args:
        B (int): The total budget of lobbyist
        T (int): The number of active time steps of lobbyist
        N (int): The number of agents
        c (int): The cost to send a signal

    Returns:
        numpy.ndarray: A matrix TxN of 0s with B/c randomly selected elements set to 1.
        list: A list of the (row, column) indices that were set to 1, i.e. the list of (time_step, agent) of sent signals
    """
    matrix = np.zeros((T, N), dtype=int)
    total_elements = T * N
    num_signals = B//c  # number of signals
    if num_signals > total_elements:
        print("Number of signals is greater than the total number of elements in the matrix."
              "Lobbyist will always send signals to all agents at each iteration.")
        num_signals = total_elements
        

    # Generate k unique random linear indices
    linear_indices = np.random.choice(total_elements, size=num_signals, replace=False)

    # Convert linear indices to row and column indices
    row_indices, col_indices = np.unravel_index(linear_indices, (T, N))

    # Create a list of (row, column) index pairs
    selected_indices = list(zip(row_indices, col_indices))

    # Set the corresponding elements in the matrix to 1
    matrix[row_indices, col_indices] = 1

    return matrix

def create_single_random_strategy_per_time(B: int, # total budget
                                             T: int, # total time steps
                                             N: int, # number of agents
                                             c: int =1 # cost of a signal
                                             ) -> np.ndarray:
    """
    Create the strategy matrix TxN, randomly selects fixed number of signals at each time step in the TxN matrix
      and sets them equals to 1. Per time step, the number of signals is fixed B/(c*T).

    Args:
        B (int): The total budget of lobbyist
        T (int): The number of active time steps of lobbyist
        N (int): The number of agents
        c (int): The cost to send a signal

    Returns:
        numpy.ndarray: A matrix TxN of 0s with randomly selected elements set to 1. Per time step, the number of signals is fixed B/(c*T).
        list: A list of the (row, column) indices that were set to 1, i.e. the list of (time_step, agent) of sent signals
    """
    inter_per_time = B // (c * T)
    matrix = np.zeros((T, N), dtype=int)
    for t in range(T):
        indices = np.random.choice(N, inter_per_time, replace=False)
        matrix[t, indices] = 1
    return matrix