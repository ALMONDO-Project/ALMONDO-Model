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
        '/home/leonardo/PycharmProjects/ALMONDO-Model/src/almondo_model/results/gw_lambda_SA_1_lobbyists_gw00.5_k10_phi1.0'
    ]

    for path in paths:
            basepath = path
            filename = 'config.json'
            
            metrics = MetricsPhiC(nl=1, basepath=basepath, filename=filename)
            
            metrics.compute_metrics(kind='probabilities', Overwrite=True)
    
                    
if __name__ == "__main__":
    main()