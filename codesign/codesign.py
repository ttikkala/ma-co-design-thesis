import rltrainer.trainer
import bodies.create_ants
from tasks.energybased import load_environment
from codesign.obsnormaliser import ObservationNormaliser
import td3.networks as networks
import matd3.replay as replay
from matd3.matd3 import batch_to_torch
import pso.pso_modified
import gp.random_data

import torch
import copy
import time
import os
import numpy as np
import pyswarms
import csv
import sys
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from scipy.stats import norm
import pickle
import sys

""" 
    This script contains the co-design class which handles starting each co-design experiment,
    calling the trainer class to train the controller and optimising the surrogate function
    to find the new design.
"""

class CoDesign():

    def __init__(self, n_agents, design_opt_style, pop_nets_file_path = None, device='cuda', design_params=None, design_filenames=None):
        # Init co-design class
        # components: env (dm_control soccer env with custom task), observation normaliser, RL trainer
        self.n_agents = n_agents
        self.n_players = self.n_agents * 2
        self.device = device
        self.design_opt_style = design_opt_style

        match self.design_opt_style:
            case 'both':
                self.same_morphology = True
            case 'joint':
                self.same_morphology = False
        
        if not design_params:
            self.design_params = {'0': [0.2828, 0.5657, 0.2828, 0.5657], 
                                  '1': [0.2828, 0.5657, 0.2828, 0.5657]} # original ant leg lengths
        else:
            self.design_params = design_params
        
        self.design_dim = len(self.design_params['0'])
        self.design_bounds = {'lower': 0.1, 'upper': 1.0}

        if not design_filenames:
            self.design_filenames = {'0': "/home/tiia/thesis/bodies/ant_highres.xml", 
                                     '1': "/home/tiia/thesis/bodies/ant_highres.xml"}
        else:
            self.design_filenames = design_filenames

        self.env = load_environment(team_size=n_agents, time_limit=50, 
                                    filename_0=self.design_filenames['0'], filename_1=self.design_filenames['1'],
                                    disable_walker_contacts=True, pitch_width=32, pitch_height=24, goal_vel=1.0)
        
        time_step = self.env.reset()
        self.observation_dim_one_agent = sum(v.size for v in time_step.observation[0].values()) + self.design_dim
        self.observation_dim_all_players = self.observation_dim_one_agent * self.n_players
        action_specs = self.env.action_spec()
        self.action_dim_one_agent = action_specs[0].shape[0]
        self.action_dim_all_players = self.action_dim_one_agent * self.n_players

        self.pop_replay_buffer = replay.SimpleReplayBuffer(
            max_replay_buffer_size=2000000,
            observation_dim=self.observation_dim_all_players,
            action_dim=self.action_dim_all_players,
            reward_dim=n_agents,
            env_info_sizes={},)
        
        self.pop_nets = init_pop_nets(n_agents=self.n_agents, observation_dim_one_agent=self.observation_dim_one_agent, 
                                      action_dim_one_agent=self.action_dim_one_agent, observation_dim_all_players=self.observation_dim_all_players, 
                                      action_dim_all_players=self.action_dim_all_players, file_path=pop_nets_file_path)

        self.obsnorm_ind = ObservationNormaliser(shape = self.observation_dim_all_players, device = device)
        self.obsnorm_ind.set_to_pretrained()

        self.obsnorm_pop = ObservationNormaliser(shape = self.observation_dim_all_players, device = device)
        self.obsnorm_pop.set_to_pretrained()

        self.log_timestamp = time.ctime().replace(' ', '_')

        self.ind_nets = init_indiv_nets(self.n_agents, self.observation_dim_one_agent, self.action_dim_one_agent, 
                                        self.observation_dim_all_players, self.action_dim_all_players)
        self._reset_ind_nets()

        folder = 'experiment_data_test_runs'
        self._main_file_path = '/scratch/tiia/' + folder + '/' + 'main_' + self.log_timestamp
        # Create experiment folder
        if not os.path.exists(self._main_file_path):
            os.makedirs(self._main_file_path)

        self.file_path = self._main_file_path + '/' + self.log_timestamp
        # Create experiment folder
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        self.trainer = rltrainer.trainer.RLTrainer(n_agents=self.n_agents, env=self.env, obsnorm_ind=self.obsnorm_ind, obsnorm_pop=self.obsnorm_pop,
                                                       log_timestamp=self.log_timestamp, ind_nets=self.ind_nets,
                                                       pop_nets=self.pop_nets, pop_replay_buffer=self.pop_replay_buffer,
                                                       design_params=self.design_params, file_path=self.file_path)
        
        # Initialise Gaussian process
        self.design_params_history = gp.random_data.design_params
        self.rewards_history = gp.random_data.rewards


    def _reset_experiment_new_design(self):
        # Reset env, observation normaliser and RL trainer when there's a new design

        self.env = load_environment(team_size=self.n_agents, time_limit=50, 
                                    filename_0=self.design_filenames['0'], filename_1=self.design_filenames['1'], 
                                    disable_walker_contacts=True, pitch_width=32, pitch_height=24, goal_vel=1.0)
        time_step = self.env.reset()

        self.log_timestamp = time.ctime().replace(' ', '_')
        self.file_path = self._main_file_path + '/' + self.log_timestamp
        # Create experiment folder
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        self.ind_nets = init_indiv_nets(self.n_agents, self.observation_dim_one_agent, self.action_dim_one_agent, 
                                        self.observation_dim_all_players, self.action_dim_all_players)
        self._reset_ind_nets()

        self.trainer = rltrainer.trainer.RLTrainer(n_agents=self.n_agents, env=self.env, obsnorm_ind=self.obsnorm_ind, obsnorm_pop=self.obsnorm_pop,
                                                       log_timestamp=self.log_timestamp, ind_nets=self.ind_nets,
                                                       pop_nets=self.pop_nets, pop_replay_buffer=self.pop_replay_buffer,
                                                       design_params=self.design_params, file_path=self.file_path)
    

    def _reset_ind_nets(self, copy_from_pop=False):
        # Sets weights of ind nets
        if copy_from_pop:
            for i in range(self.n_agents):
                self.ind_nets['q1_nets'][i].load_state_dict(self.pop_nets['q1_nets'][i].state_dict())
                self.ind_nets['q2_nets'][i].load_state_dict(self.pop_nets['q2_nets'][i].state_dict())
                self.ind_nets['target_q1_nets'][i].load_state_dict(self.pop_nets['target_q1_nets'][i].state_dict())
                self.ind_nets['target_q2_nets'][i].load_state_dict(self.pop_nets['target_q2_nets'][i].state_dict())
                self.ind_nets['policy_nets'][i].load_state_dict(self.pop_nets['policy_nets'][i].state_dict())
                self.ind_nets['target_policy_nets'][i].load_state_dict(self.pop_nets['target_policy_nets'][i].state_dict())

        else:
            policy_save_path0 = "/scratch/tiia/experiment_data_test_runs/main_Sat_Dec__6_14:35:08_2025/Mon_Dec__8_01:32:47_2025/rlmodels/soccer_roles_matd3_policy0.pt"
            policy_save_path1 = "/scratch/tiia/experiment_data_test_runs/main_Sat_Dec__6_14:35:08_2025/Mon_Dec__8_01:32:47_2025/rlmodels/soccer_roles_matd3_policy1.pt"
            saved_policy_state0 = torch.load(policy_save_path0, map_location=self.device)  
            saved_policy_state1 = torch.load(policy_save_path1, map_location=self.device)
            self.ind_nets['policy_nets'][0].load_state_dict(copy.deepcopy(saved_policy_state0))
            self.ind_nets['policy_nets'][1].load_state_dict(copy.deepcopy(saved_policy_state1))

            q1_save_path0 = "/scratch/tiia/experiment_data_test_runs/main_Sat_Dec__6_14:35:08_2025/Mon_Dec__8_01:32:47_2025/rlmodels/soccer_roles_matd3_q10.pt"
            q1_save_path1 = "/scratch/tiia/experiment_data_test_runs/main_Sat_Dec__6_14:35:08_2025/Mon_Dec__8_01:32:47_2025/rlmodels/soccer_roles_matd3_q11.pt"
            saved_q1_state0 = torch.load(q1_save_path0, map_location=self.device)  
            saved_q1_state1 = torch.load(q1_save_path1, map_location=self.device)
            self.ind_nets['q1_nets'][0].load_state_dict(copy.deepcopy(saved_q1_state0))
            self.ind_nets['q1_nets'][1].load_state_dict(copy.deepcopy(saved_q1_state1))
            self.ind_nets['q2_nets'][0].load_state_dict(copy.deepcopy(saved_q1_state0))
            self.ind_nets['q2_nets'][1].load_state_dict(copy.deepcopy(saved_q1_state1))        

            q1_targ_save_path0 = "/scratch/tiia/experiment_data_test_runs/main_Sat_Dec__6_14:35:08_2025/Mon_Dec__8_01:32:47_2025/rlmodels/soccer_roles_matd3_q1_target0.pt"
            q1_targ_save_path1 = "/scratch/tiia/experiment_data_test_runs/main_Sat_Dec__6_14:35:08_2025/Mon_Dec__8_01:32:47_2025/rlmodels/soccer_roles_matd3_q1_target1.pt"
            saved_q1_targ_state0 = torch.load(q1_targ_save_path0, map_location=self.device)  
            saved_q1_targ_state1 = torch.load(q1_targ_save_path1, map_location=self.device)
            self.ind_nets['target_q1_nets'][0].load_state_dict(copy.deepcopy(saved_q1_targ_state0))
            self.ind_nets['target_q1_nets'][1].load_state_dict(copy.deepcopy(saved_q1_targ_state1))
            self.ind_nets['target_q2_nets'][0].load_state_dict(copy.deepcopy(saved_q1_targ_state0))
            self.ind_nets['target_q2_nets'][1].load_state_dict(copy.deepcopy(saved_q1_targ_state1))
        

    def _create_new_walkers(self, params_0, params_1):
        # Call create_ants with desired params and store xmls in accessible way
        output_dir = self.file_path + '/xmls'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        new_design_filename_0 = bodies.create_ants.generate_ants(params_0, agent_id=0, log_timestamp=self.log_timestamp, output_dir=output_dir)#, input_file="/home/tiia/thesis/bodies/ant_highres_lower_power.xml")
        new_design_filename_1 = bodies.create_ants.generate_ants(params_1, agent_id=1, log_timestamp=self.log_timestamp, output_dir=output_dir)#, input_file="/home/tiia/thesis/bodies/ant_highres_lower_power.xml")

        return (new_design_filename_0, new_design_filename_1)

    def _optimise_designs_q(self):
        # Call PSO to optimise the designs based on predictions of pop Q-net

        batch = self.pop_replay_buffer.random_batch(batch_size = 1024)
        batch = batch_to_torch(batch, device=self.device)
        initial_states = batch['observations']
        actions_batch = batch['actions']
        normed_states = self.obsnorm_pop.normalise(initial_states)
        design_indxs = [304, 612, 920, 1228]
        start_idx = 304

        pso_start_time = time.time()

        def f_qval_both(x_input, **kwargs):
            # Get cost through predictions with the population Q-function
            # Both agents have the same design, look for one with highest combined Q-value for both agents
            shape = x_input.shape # shape of (n_particles, design_dim)
            cost = np.zeros((shape[0],))
            x_input = torch.from_numpy(x_input)

            with torch.no_grad():

                for i in range(shape[0]): # loop over particles
                    actions = []

                    x = x_input[i:i+1,:] # location of particle i in design space
                    full_state = normed_states.clone()
                    local_state_agent0 = full_state[:, 0:self.observation_dim_one_agent]

                    # start_idx = design_indxs[0]
                    local_state_agent0[:, start_idx:start_idx+len(x[0])] = x
                    action_0 = self.pop_nets['policy_nets'][0](local_state_agent0)
                    actions.append(action_0)

                    local_state_agent1 = full_state[:, (1 * self.observation_dim_one_agent):((1+1) * self.observation_dim_one_agent)]
                    local_state_agent1[:, start_idx:start_idx+len(x[0])] = x
                    action_1 = self.pop_nets['policy_nets'][1](local_state_agent1)
                    actions.append(action_1)
                    
                    away_team_actions = actions_batch[:, (self.n_agents * 8):]
                    actions = torch.cat(actions + [away_team_actions], dim=1)

                    away_team_states = full_state[:, (2 * self.observation_dim_one_agent):]

                    q_input_states = torch.cat((local_state_agent0, local_state_agent1, away_team_states), dim=1)

                    output_0 = torch.min(self.pop_nets['q1_nets'][0](q_input_states, actions), self.pop_nets['q2_nets'][0](q_input_states, actions))
                    output_1 = torch.min(self.pop_nets['q1_nets'][1](q_input_states, actions), self.pop_nets['q2_nets'][1](q_input_states, actions))

                    output = output_0 + output_1
                    loss = - output.mean().sum()

                    fval = float(loss.item())
                    cost[i] = fval # This is pos. because loss is neg.

            return cost


        def f_qval_joint(x_input, **kwargs):
            # Get cost through predictions with the population Q-function
            # Agents have different designs, look for designs that maximise combined Q-value of both agents
            shape = x_input.shape # shape of (n_particles, design_dim)
            cost = np.zeros((shape[0],))
            x_input = torch.from_numpy(x_input)
            
            with torch.no_grad():

                for i in range(shape[0]): # loop over particles
                    actions = []

                    x = x_input[i:i+1,:] # location of particle i in design space
                    full_state = normed_states.clone()
                    local_state_agent0 = full_state[:, 0:self.observation_dim_one_agent]

                    # start_idx = design_indxs[0]
                    local_state_agent0[:, start_idx:start_idx+self.design_dim] = x[:,:self.design_dim]
                    action_0 = self.pop_nets['policy_nets'][0](local_state_agent0)
                    actions.append(action_0)

                    local_state_agent1 = full_state[:, (1 * self.observation_dim_one_agent):((1+1) * self.observation_dim_one_agent)]
                    local_state_agent1[:, start_idx:start_idx+self.design_dim] = x[:,self.design_dim:]
                    action_1 = self.pop_nets['policy_nets'][1](local_state_agent1)
                    actions.append(action_1)
                    
                    away_team_actions = actions_batch[:, (self.n_agents * 8):]
                    actions = torch.cat(actions + [away_team_actions], dim=1)

                    away_team_states = full_state[:, (2 * self.observation_dim_one_agent):]

                    q_input_states = torch.cat((local_state_agent0, local_state_agent1, away_team_states), dim=1)

                    output_0 = torch.min(self.pop_nets['q1_nets'][0](q_input_states, actions), self.pop_nets['q2_nets'][0](q_input_states, actions))
                    output_1 = torch.min(self.pop_nets['q1_nets'][1](q_input_states, actions), self.pop_nets['q2_nets'][1](q_input_states, actions))
                    output = output_0 + output_1
                    loss = - output.mean().sum()
                    fval = float(loss.item())
                    cost[i] = fval # This is pos. because loss is neg.

            return cost
        
        options = {'c1': 2.05, 'c2': 2.05, 'w': 0.72984378}
        n_particles = 700

        if self.design_opt_style == 'both':
            bounds = (np.array([self.design_bounds['lower']]*4), np.array([self.design_bounds['upper']]*4))
            optimizer = pso.pso_modified.PSOModified(n_particles=n_particles, dimensions=self.design_dim, bounds=bounds, options=options)
            cost, new_design_params = optimizer.optimize(f_qval_both, print_step=100, iters=500, verbose=3)
            new_design_params_0 = new_design_params
            new_design_params_1 = new_design_params
        elif self.design_opt_style == 'joint':
            bounds = (np.array([self.design_bounds['lower']]*8), np.array([self.design_bounds['upper']]*8))
            optimizer = pso.pso_modified.PSOModified(n_particles=n_particles, dimensions=self.design_dim * 2, bounds=bounds, options=options)
            cost, new_design_params = optimizer.optimize(f_qval_joint, print_step=100, iters=500, verbose=3)
            new_design_params_0 = new_design_params[:self.design_dim]
            new_design_params_1 = new_design_params[self.design_dim:]

        else:
            sys.exit("No valid design optimisation style provided")


        pso_end_time = time.time()
        print("PSO time [s]: ", pso_end_time - pso_start_time)

        return (new_design_params_0, new_design_params_1)


    def _optimise_designs_gp(self, is_first=False, adaptive_strategy=True):
        # Call PSO to optimise the designs based on Gaussian Process Regression to performance data

        pso_start_time = time.time()

        if not is_first:
            # Add newest performance data to GPR
            filename = os.path.join(self.file_path, 'ind_rewards.csv')
            df = pd.read_csv(filename)
            avg_rew = 0
            
            for i in range(self.n_agents):
                agent_id = i
                agent_df = df[df['agent_id'] == agent_id]
            
                last_episodes = agent_df.tail(20)
                
                # Calculate average 
                if len(last_episodes) > 0:
                    avg_test_reward = float(last_episodes['test_reward'].mean())
                    avg_rew += avg_test_reward
            
            self.rewards_history = np.append(self.rewards_history, np.array(avg_rew))

            design_path = os.path.join(self.file_path, 'design_params.csv')
            design_df = pd.read_csv(design_path)
            agent_0_params = design_df[design_df['agent'] == 0]
            
            if len(agent_0_params) > 0:
                params = [
                    float(agent_0_params['front_leg'].values[0]),
                    float(agent_0_params['front_ankle'].values[0]),
                    float(agent_0_params['back_leg'].values[0]),
                    float(agent_0_params['back_ankle'].values[0])
                ]
            
            self.design_params_history = np.vstack([self.design_params_history, params])

        # Fit GP
        kernel = ConstantKernel(1.0) * RBF(length_scale=[0.3]*4, length_scale_bounds=(0.3, 5.0))
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=0.1, normalize_y=True)
        gaussian_process.fit(self.design_params_history, self.rewards_history)
        gaussian_process.kernel_

        X1,X2,X3,X4 = np.mgrid[0.1:1.1:0.1, 0.1:1.1:0.1, 0.1:1.1:0.1, 0.1:1.1:0.1]
        X_grid = np.stack([X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel()], axis=1)

        mean_prediction, std_prediction = gaussian_process.predict(X_grid, return_std=True)
        print("GP mean pred: ", mean_prediction)
        print("GP max on grid: ", max(mean_prediction))
        print("GP STD: ", std_prediction)
        print("Kernel: ", gaussian_process.kernel_)

        n_samples = len(self.rewards_history)
        best_observed = np.max(self.rewards_history)
        
        print(f"Best observed reward: {best_observed:.2f}")
        print(f"Kernel: {gaussian_process.kernel_}")
        
        # Adaptive strategy selection
        if adaptive_strategy:
            if n_samples < 140+15:
                # Early phase: Pure exploration (high UCB)
                exploration_weight = 0.1
                
                def f(x_input, **kwargs):
                    mean, std = gaussian_process.predict(x_input, return_std=True)
                    return -(mean + exploration_weight * std)
                                    
            elif n_samples < 140+30:
                # Middle phase: Expected Improvement
                xi = 0.005

                def f(x_input, **kwargs):
                    mean, std = gaussian_process.predict(x_input, return_std=True)
                    std = np.maximum(std, 1e-9)
                    
                    z = (mean - best_observed - xi) / std
                    ei = (mean - best_observed - xi) * norm.cdf(z) + std * norm.pdf(z)
                    
                    return -ei
                                
            else:
                # Late phase: Exploitation (low LCB)
                exploration_weight = 0.001
                
                def f(x_input, **kwargs):
                    mean, std = gaussian_process.predict(x_input, return_std=True)
                    return -(mean - exploration_weight * std)
                    
        else:
            exploration_weight = 0.001
                
            def f(x_input, **kwargs):
                mean, std = gaussian_process.predict(x_input, return_std=True)
                return -(mean - exploration_weight * std)

        # Climb GP with PSO
        options = {'c1': 2.05, 'c2': 2.05, 'w': 0.72984378}
        n_particles = 700

        bounds = (np.array([self.design_bounds['lower']]*4), np.array([self.design_bounds['upper']]*4))
        optimizer = pso.pso_modified.PSOModified(n_particles=n_particles, dimensions=self.design_dim, bounds=bounds, options=options)
        cost, new_design_params = optimizer.optimize(f, print_step=100, iters=500, verbose=3)
        new_design_params_0 = new_design_params
        new_design_params_1 = new_design_params

        pso_end_time = time.time()
        print("PSO time [s]: ", pso_end_time - pso_start_time)

        print(f"Optimal point: {new_design_params_0}")

        
        # Save GPR state
        gpr_save_dir = os.path.join(self._main_file_path, 'gpr_models')
        if not os.path.exists(gpr_save_dir):
            os.makedirs(gpr_save_dir)
        
        # Save the trained GPR model
        gpr_filename = os.path.join(gpr_save_dir, f'gpr_{self.log_timestamp}.pkl')
        with open(gpr_filename, 'wb') as f:
            pickle.dump(gaussian_process, f)
        
        # Also save the training data 
        metadata = {
            'design_params_history': self.design_params_history,
            'rewards_history': self.rewards_history,
            'timestamp': self.log_timestamp,
            'n_points': len(self.rewards_history),
            'kernel': str(gaussian_process.kernel_),
            'optimal_design': new_design_params_0
        }
        metadata_filename = os.path.join(gpr_save_dir, f'gpr_metadata_{self.log_timestamp}.pkl')
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)

        return (new_design_params_0, new_design_params_1)


    def _explore_designs(self):

        design_params_0 = np.random.uniform(low=self.design_bounds['lower'], high=self.design_bounds['upper'], size=self.design_dim)
        if not self.same_morphology:
            design_params_1 = np.random.uniform(low=self.design_bounds['lower'], high=self.design_bounds['upper'], size=self.design_dim)
        else:
            design_params_1 = design_params_0
        
        return (design_params_0, design_params_1)

    def _train_nets(self, exploration_eps = 20, training_eps = 300, n_render = 0, update_pop = True):
        
        for _ in range(exploration_eps):
            self.trainer.collect_training_data() 
        for i in range(training_eps):

            print("Episode ", i)
            if i >= training_eps - n_render:
                self.trainer.single_training_step(self.file_path, iter=i, is_captured=True, update_pop_nets=update_pop)
            else:
                self.trainer.single_training_step(self.file_path, iter=i, update_pop_nets=update_pop)
        
        torch.set_printoptions(threshold=100_000)
        print("Means, vars ind \n", self.obsnorm_ind.get_means_vars())
        print("Means, vars pop \n", self.obsnorm_pop.get_means_vars())

        self.trainer.save_agents()

    def _save_output(self):
        # Save pop networks
        model_save_folder = self.file_path + '/rlmodels'
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)

        self.q_save_path = model_save_folder + '/pop_soccer_matd3_q1.pt'
        self.target_q_save_path = model_save_folder + '/pop_soccer_matd3_q1_target.pt'
        self.policy_save_path = model_save_folder + '/pop_soccer_matd3_policy.pt'

        for i in range(self.n_agents):
            torch.save(self.pop_nets['q1_nets'][i].state_dict(), self.q_save_path[:-3] + str(i) + self.q_save_path[-3:])
            torch.save(self.pop_nets['target_q1_nets'][i].state_dict(), self.target_q_save_path[:-3] + str(i) + self.target_q_save_path[-3:])
            torch.save(self.pop_nets['policy_nets'][i].state_dict(), self.policy_save_path[:-3] + str(i) + self.policy_save_path[-3:])


    def run_experiment(self, n_pso=20, n_random=15, training_episodes=300):
        # Main loop, run simulation and optimise designs based on results
        save_design_params(self.file_path, self.design_params)

        # fill up replay buffer for 20 episodes before first PSO round
        for _ in range(20):
            self.trainer.collect_training_data(save_pop=True, noise=False) 
        
        for i in range(n_random * 2):

            if i % 2 == 0:
                design_params_0, design_params_1 = self._optimise_designs_gp()
                self.design_params['0'] = design_params_0
                self.design_params['1'] = design_params_1
            else:
                design_params_0, design_params_1 = self._explore_designs()
                self.design_params['0'] = design_params_0
                self.design_params['1'] = design_params_1

            self._reset_experiment_new_design()

            new_design_filename_0, new_design_filename_1 = self._create_new_walkers(self.design_params['0'], self.design_params['1'])
            self.design_filenames['0'] = new_design_filename_0
            self.design_filenames['1'] = new_design_filename_1

            save_design_params(self.file_path, self.design_params)

            self._train_nets(update_pop=True, training_eps=training_episodes)

            # Clean up at the end of a co-design iteration
            self._save_output()

        for _ in range(n_pso - n_random):

            design_params_0, design_params_1 = self._optimise_designs_gp()
            self.design_params['0'] = design_params_0
            self.design_params['1'] = design_params_1 

            self._reset_experiment_new_design()

            new_design_filename_0, new_design_filename_1 = self._create_new_walkers(self.design_params['0'], self.design_params['1'])
            self.design_filenames['0'] = new_design_filename_0
            self.design_filenames['1'] = new_design_filename_1

            save_design_params(self.file_path, self.design_params)

            self._train_nets(update_pop=True, training_eps=training_episodes)

            # Clean up at the end of a co-design iteration
            self._save_output()


    def run_experiment_gp(self, n=40, training_episodes=500, adaptive_strategy=True):
        # Main loop, run simulation and optimise designs based on results
        save_design_params(self.file_path, self.design_params)

        # fill up replay buffer for 20 episodes before first PSO round
        for _ in range(20): 
            self.trainer.collect_training_data(save_pop=True, noise=False) 
        
        design_params_0, design_params_1 = self._optimise_designs_gp(is_first=True, adaptive_strategy=adaptive_strategy)
        self.design_params['0'] = design_params_0
        self.design_params['1'] = design_params_1

        self._reset_experiment_new_design()

        new_design_filename_0, new_design_filename_1 = self._create_new_walkers(self.design_params['0'], self.design_params['1'])
        self.design_filenames['0'] = new_design_filename_0
        self.design_filenames['1'] = new_design_filename_1

        save_design_params(self.file_path, self.design_params)

        self._train_nets(update_pop=True, training_eps=training_episodes, n_render=1)

        # Clean up at the end of a co-design iteration
        self._save_output()

        for i in range(n-1):

            design_params_0, design_params_1 = self._optimise_designs_gp(adaptive_strategy=adaptive_strategy)
            self.design_params['0'] = design_params_0
            self.design_params['1'] = design_params_1

            self._reset_experiment_new_design()

            new_design_filename_0, new_design_filename_1 = self._create_new_walkers(self.design_params['0'], self.design_params['1'])
            self.design_filenames['0'] = new_design_filename_0
            self.design_filenames['1'] = new_design_filename_1

            save_design_params(self.file_path, self.design_params)

            self._train_nets(update_pop=True, training_eps=training_episodes, n_render=1)

            # Clean up at the end of a co-design iteration
            self._save_output()


    def pretrain(self, training_eps=300, rounds=5, n_render=1):
        # Pre-train a policy that sees random designs (only Q-based co-design)

        self._train_nets(training_eps=training_eps, update_pop=False, n_render=n_render) 

        for i in range(rounds):
            if i > 3:
                update_pop = True 
            else:
                update_pop = False

            design_params_0, design_params_1 = self._explore_designs()
            self.design_params['0'] = design_params_0
            self.design_params['1'] = design_params_1
            
            self._reset_experiment_new_design()
            new_design_filename_0, new_design_filename_1 = self._create_new_walkers(self.design_params['0'], self.design_params['1'])
            self.design_filenames['0'] = new_design_filename_0
            self.design_filenames['1'] = new_design_filename_1

            save_design_params(self.file_path, self.design_params)

            self._train_nets(training_eps=training_eps, update_pop=update_pop, n_render=n_render)

            # Clean up at the end of a co-design iteration
            self._save_output()


def init_indiv_nets(n_agents, observation_dim_one_agent, action_dim_one_agent, observation_dim_all_players, action_dim_all_players, device='cuda'):

    # Initialise critic and actor networks and target networks
    hidden_sizes_q = [1024,512,256]
    hidden_sizes_policy = [400,300]
    q1_nets = []
    q2_nets = []
    target_q1_nets = []
    target_q2_nets = []
    policy_nets = []
    target_policy_nets = []

    for _ in range(n_agents):
        q1_nets.append(networks.QvalueNetwork(hidden_sizes = hidden_sizes_q, 
                                        input_size = observation_dim_all_players + action_dim_all_players).to(device=device))
        q2_nets.append(networks.QvalueNetwork(hidden_sizes = hidden_sizes_q, 
                                        input_size = observation_dim_all_players + action_dim_all_players).to(device=device))
        target_q1_nets.append(networks.QvalueNetwork(hidden_sizes = hidden_sizes_q, 
                                                input_size = observation_dim_all_players + action_dim_all_players).to(device=device))
        target_q2_nets.append(networks.QvalueNetwork(hidden_sizes = hidden_sizes_q, 
                                                input_size = observation_dim_all_players + action_dim_all_players).to(device=device))
        policy_nets.append(networks.PolicyNetwork(hidden_sizes = hidden_sizes_policy, 
                                                input_size = observation_dim_one_agent, output_size=action_dim_one_agent).to(device=device))
        target_policy_nets.append(networks.PolicyNetwork(hidden_sizes = hidden_sizes_policy, 
                                                    input_size = observation_dim_one_agent, output_size=action_dim_one_agent).to(device=device))

    policy_save_path0 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_policy0.pt"
    policy_save_path1 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_policy1.pt"
    saved_policy_state0 = torch.load(policy_save_path0, map_location=device)  
    saved_policy_state1 = torch.load(policy_save_path1, map_location=device)
    policy_nets[0].load_state_dict(copy.deepcopy(saved_policy_state0))
    policy_nets[1].load_state_dict(copy.deepcopy(saved_policy_state1))

    q1_save_path0 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_q10.pt"
    q1_save_path1 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_q11.pt"
    saved_q1_state0 = torch.load(q1_save_path0, map_location=device)  
    saved_q1_state1 = torch.load(q1_save_path1, map_location=device)
    q1_nets[0].load_state_dict(copy.deepcopy(saved_q1_state0))
    q1_nets[1].load_state_dict(copy.deepcopy(saved_q1_state1))

    q1_targ_save_path0 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_q1_target0.pt"
    q1_targ_save_path1 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_q1_target1.pt"
    saved_q1_targ_state0 = torch.load(q1_targ_save_path0, map_location=device)  
    saved_q1_targ_state1 = torch.load(q1_targ_save_path1, map_location=device)
    target_q1_nets[0].load_state_dict(copy.deepcopy(saved_q1_targ_state0))
    target_q1_nets[1].load_state_dict(copy.deepcopy(saved_q1_targ_state1))


    return {'hidden_sizes_q': hidden_sizes_q, 
            'hidden_sizes_policy': hidden_sizes_policy, 
            'q1_nets': q1_nets, 
            'q2_nets': q2_nets, 
            'target_q1_nets': target_q1_nets, 
            'target_q2_nets': target_q2_nets, 
            'policy_nets': policy_nets, 
            'target_policy_nets': target_policy_nets}


def init_pop_nets(n_agents, observation_dim_one_agent, action_dim_one_agent, observation_dim_all_players, action_dim_all_players, file_path='/scratch/tiia/experiment_data_test_runs/main_Thu_Nov_27_16:39:30_2025/Fri_Nov_28_11:58:40_2025', device='cuda'):
    
    # Initialise critic and actor networks and target networks
    hidden_sizes_q = [1024,512,256]
    # hidden_sizes_q = [512,256,128]
    hidden_sizes_policy = [400,300]
    q1_nets = []
    q2_nets = []
    target_q1_nets = []
    target_q2_nets = []
    policy_nets = []
    target_policy_nets = []

    for _ in range(n_agents):
        q1_nets.append(networks.QvalueNetwork(hidden_sizes = hidden_sizes_q, 
                                        input_size = observation_dim_all_players + action_dim_all_players).to(device=device))
        q2_nets.append(networks.QvalueNetwork(hidden_sizes = hidden_sizes_q, 
                                        input_size = observation_dim_all_players + action_dim_all_players).to(device=device))
        target_q1_nets.append(networks.QvalueNetwork(hidden_sizes = hidden_sizes_q, 
                                                input_size = observation_dim_all_players + action_dim_all_players).to(device=device))
        target_q2_nets.append(networks.QvalueNetwork(hidden_sizes = hidden_sizes_q, 
                                                input_size = observation_dim_all_players + action_dim_all_players).to(device=device))
        policy_nets.append(networks.PolicyNetwork(hidden_sizes = hidden_sizes_policy, 
                                                input_size = observation_dim_one_agent, output_size=action_dim_one_agent).to(device=device))
        target_policy_nets.append(networks.PolicyNetwork(hidden_sizes = hidden_sizes_policy, 
                                                    input_size = observation_dim_one_agent, output_size=action_dim_one_agent).to(device=device))


    policy_save_path0 = '/scratch/tiia/experiment_data_test_runs/main_Mon_Dec_29_12-05-30_2025/Wed_Dec_31_01-25-58_2025/rlmodels/pop_soccer_matd3_policy0.pt'
    policy_save_path1 = '/scratch/tiia/experiment_data_test_runs/main_Mon_Dec_29_12-05-30_2025/Wed_Dec_31_01-25-58_2025/rlmodels/pop_soccer_matd3_policy1.pt'
    saved_policy_state0 = torch.load(policy_save_path0, map_location=device)  
    saved_policy_state1 = torch.load(policy_save_path1, map_location=device)
    policy_nets[0].load_state_dict(copy.deepcopy(saved_policy_state0))
    policy_nets[1].load_state_dict(copy.deepcopy(saved_policy_state1))

    q1_save_path0 = "/scratch/tiia/experiment_data_test_runs/main_Mon_Dec_29_12-05-30_2025/Wed_Dec_31_01-25-58_2025/rlmodels/pop_soccer_matd3_q10.pt"
    q1_save_path1 = "/scratch/tiia/experiment_data_test_runs/main_Mon_Dec_29_12-05-30_2025/Wed_Dec_31_01-25-58_2025/rlmodels/pop_soccer_matd3_q11.pt"
    saved_q1_state0 = torch.load(q1_save_path0, map_location=device)  
    saved_q1_state1 = torch.load(q1_save_path1, map_location=device)
    q1_nets[0].load_state_dict(copy.deepcopy(saved_q1_state0))
    q1_nets[1].load_state_dict(copy.deepcopy(saved_q1_state1))

    q1_targ_save_path0 = "/scratch/tiia/experiment_data_test_runs/main_Mon_Dec_29_12-05-30_2025/Wed_Dec_31_01-25-58_2025/rlmodels/pop_soccer_matd3_q1_target0.pt"
    q1_targ_save_path1 = "/scratch/tiia/experiment_data_test_runs/main_Mon_Dec_29_12-05-30_2025/Wed_Dec_31_01-25-58_2025/rlmodels/pop_soccer_matd3_q1_target1.pt"
    saved_q1_targ_state0 = torch.load(q1_targ_save_path0, map_location=device)  
    saved_q1_targ_state1 = torch.load(q1_targ_save_path1, map_location=device)
    target_q1_nets[0].load_state_dict(copy.deepcopy(saved_q1_targ_state0))
    target_q1_nets[1].load_state_dict(copy.deepcopy(saved_q1_targ_state1))


    return {'hidden_sizes_q': hidden_sizes_q, 
            'hidden_sizes_policy': hidden_sizes_policy, 
            'q1_nets': q1_nets, 
            'q2_nets': q2_nets, 
            'target_q1_nets': target_q1_nets, 
            'target_q2_nets': target_q2_nets, 
            'policy_nets': policy_nets, 
            'target_policy_nets': target_policy_nets}



def save_design_params(file_path, params):
    filename = os.path.join(file_path, 'design_params.csv')
    
    # Check if file exists and is empty to write header
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    
    with open(filename, 'a') as fd:
        cwriter = csv.writer(fd)
        
        if write_header:
            cwriter.writerow(['agent', 'front_leg', 'front_ankle', 'back_leg', 'back_ankle'])
        
        cwriter.writerow(['0', params['0'][0], params['0'][1], params['0'][2], params['0'][3]])
        cwriter.writerow(['1', params['1'][0], params['1'][1], params['1'][2], params['1'][3]])


