from almondo_model.classes.simulator import ALMONDOSimulator
import matplotlib.pyplot as plt
import numpy as np


sim = ALMONDOSimulator(
    N=500,
    initial_distribution="uniform",
    T=100,
    p_o=0.01,
    p_p=0.99,
    gw0=0.5,
    k = 10.0,
    lambda_values=[0.99],
    phi_values=[0.0],
    nruns=1,
    n_lobbyists=1,
    lobbyists_data={0: {"B": 10000, "c": 1, "T": 100, "m": 0, "gw": 0.4, "strategies": [], "strategy_type": "random"}},
                    #1: {"B": 10000, "c": 1, "T": 100, "m": 1, "gw": 1.0, "strategies": [], "strategy_type": "random"}},
    verbose=True,
    scenario="/home/leonardo/PycharmProjects/ALMONDO-Model/results/debug"
)

status, final_data = sim.single_run(lambda_v=0.97, phi_v=0.0, drop_ev=True)

print("Final credibility check:", final_data['final_credibility'])
print("Final weights:", final_data["final_weights"])
print("Final probabilities:", final_data["final_probabilities"])


# Example: assume final_data is what you returned from single_run
initial_probs = final_data['initial_probabilities']
final_probs = final_data['final_probabilities']

# Option 1: overlapping histograms
plt.figure(figsize=(8,5))
plt.hist(initial_probs, bins=30, alpha=0.5, label='Initial probabilities', color='blue', density=True)
plt.hist(final_probs, bins=30, alpha=0.5, label='Final probabilities', color='red', density=True)
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Distribution of Initial and Final Probabilities')
plt.legend()
plt.show()