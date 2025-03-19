from classes.metrics import Metrics

def main():
    
    """
    Script to compute metrics for a given case study with a given number of lobbyists.
    It creates avg_metrics.csv and a
    
    """      
    
    NLs = [0, 1, 2, 3, 4, 20] #number of lobbyists in the simulations  
    Bs = [30, 60, 150, 300, 500, 750, 1000]  # lobbyists budget in the simulation
    for nl in NLs:
        for b in Bs:  
            basepath = f'../results/balanced_budgets/{nl}_lobbyists/{b}_budget/'
            filename = 'config.json'
            
            metrics = Metrics(nl=nl, basepath=basepath, filename=filename)
            
            metrics.compute_metrics(kind='probabilities')
    
                    
if __name__ == "__main__":
    main()