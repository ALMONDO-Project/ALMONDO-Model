import random
import numpy as np
import os
from tqdm import tqdm

def generate_ms(n_lobbyists: int) -> list[int]:
    if n_lobbyists > 0:
        return [nl % 2 for nl in range(n_lobbyists)] #numero "ugale" di lobbisti ottimisti e pessimisti (se il totale è pari, altrimenti c'è un 1 in più)
    else:
        return None
    
def generate_strategies(folder_path: str,
                        n_possibilites: int, 
                        N: int,         # numero nodi 
                        T: int = 3000,  # timestep totali 
                        B: int = 300000, # budget
                        c: int = 1
                        ) -> None:
    
    print('Gnerating strategies')
    
    inter_per_time = B // (c * T)
    for i in tqdm(range(n_possibilites)):
        m = np.zeros((T, N), dtype=int)
        for t in range(T):
            indices = np.random.choice(N, inter_per_time, replace=False)
            m[t, indices] = 1
        
        np.savetxt(os.path.join(folder_path, f"strategy_{i}.txt"), m, fmt="%i")
    
    print('Strategies generated and saved')
    
    return


def read_random_strategies(strategies_path: str, n_lobbyists: int) -> list[np.ndarray]:
    print(f'strategies path = {strategies_path}')
    strategy_names = random.sample(os.listdir(strategies_path), n_lobbyists)
    strategies = []
    for strategy_name in strategy_names:
        filepath = os.path.join(strategies_path, strategy_name)
        print(f"Reading f{filepath}")
        strategies.append(np.loadtxt(filepath).astype(int))
    return strategies


def read_random_strategy(strategies_path: str) -> np.ndarray:
    print(f'Strategies path = {strategies_path}')
    strategy_name = random.choice(os.listdir(strategies_path))
    filepath = os.path.join(strategies_path, strategy_name)
    print(f"Reading {filepath}")
    return np.loadtxt(filepath).astype(int), strategy_name

