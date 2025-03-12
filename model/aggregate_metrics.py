from classes.metrics import Metrics

def main():
    
    """
    Script to compute metrics for a given case study with a given number of lobbyists.
    It creates avg_metrics.csv and a
    
    """      
    
    NLs = [0, 1, 2, 3, 4, 20] #number of lobbyists in the simulations  
    
    for nl in NLs:
        basepath = f'../results/balanced_budgets/{nl}_lobbyists'
        filename = 'config.json'
        
        metrics = Metrics(nl=nl, basepath=basepath, filename=filename)
        
        metrics.compute_metrics(kind='probabilities')
    
                    
if __name__ == "__main__":
    main()