import codesign.codesign
import time

"""
This script contains the high-level logic of a co-design experiment.
"""


start_time = time.time()

team_size = 2

codesign_orchestrator = codesign.codesign.CoDesign(n_agents=team_size, design_opt_style='both') 

codesign_orchestrator.run_experiment_gp(n=40, training_episodes=500, adaptive_strategy=True)

end_time = time.time()
print("Elapsed time [s]: ", end_time - start_time)


