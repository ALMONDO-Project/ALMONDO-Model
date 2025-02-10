import random
import numpy as np
import os

def generate_ms(n_lobbyists: int) -> list[int]:
    return [nl % 2 for nl in range(n_lobbyists)] #numero "ugale" di lobbisti ottimisti e pessimisti (se il totale è pari, altrimenti c'è un 1 in più)
    
def generate_strategies(folder_path: str,
                        n_possibilites: int, 
                        N: int,         # numero nodi 
                        T: int = 3000,  # timestep totali 
                        B: int = 30000, # budget
                        c: int = 1      # costo unitario
                        ) -> None:
    """
    Creare una lista di n_possibilities (uguale a NRUNS?) matrici e salvare su file.
    Le matrici sono binarie e di forma timestep x n_lobbisti.
    Ogni entrata (t, i) indica se il lobbista interagisce
    con il nodo i al tempo t o no. Il numero totale di interazioni
    moltiplicato per il costo unitario sia uguale il budget.
    Temporaneamente assumiamo che i timestep abbiano un numero
    costante di interazioni
    """
    inter_per_time = B // (c * T)
    for i in range(n_possibilites):
        m = np.zeros((T, N), dtype=int)
        for t in range(T):
            indices = np.random.choice(N, inter_per_time, replace=False)
            m[t, indices] = 1
        
        np.savetxt(os.path.join(folder_path, f"strategy_{i}.txt"), m, fmt="%i")
    return

def read_random_strategies(strategies_path: str, n_lobbyists: int) -> list[np.ndarray]:
    strategy_names = random.sample(os.listdir(strategies_path), n_lobbyists)
    strategies = []
    for strategy_name in strategy_names:
        filepath = os.path.join(strategies_path, strategy_name)
        print(f"Reading f{filepath}")
        strategies.append(np.loadtxt(filepath).astype(int))
    return strategies
