from almondo_model.classes import MetricsPhiC
from classes.metrics import Metrics

def main():
    
    """
    Script to compute metrics for a given case study with a given number of lobbyists.
    It creates avg_metrics.csv and a
    
    """      
    
    #NLs = [2] #number of lobbyists in the simulations
    #Bs = [10]  # lobbyists budget in the simulation

    paths = [
        '/home/leonardo/PycharmProjects/ALMONDO-Model/src/almondo_model/results/balanced_budgets/phi_c_SA_1_lobbyists'
    ]

    for path in paths:
            basepath = path
            filename = 'config.json'
            
            metrics = MetricsPhiC(nl=1, basepath=basepath, filename=filename)
            
            metrics.compute_metrics(kind='probabilities', Overwrite=True)
    
                    
if __name__ == "__main__":
    main()