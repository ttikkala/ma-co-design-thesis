import numpy as np
import torch
import os
import csv
import cv2
import copy

import rendering.renderhelper as renderhelper
import matd3.matd3
import matd3.replay as replay
import rltrainer.replay_wrapper as replay_wrapper
import td3.networks as networks

"""
    This script contains the trainer class, which handles the high-level reinforcement learning
    training logic. 
"""


class RLTrainer:

    def __init__(self, n_agents, env, obsnorm_ind, obsnorm_pop, log_timestamp, ind_nets, pop_nets, pop_replay_buffer, design_params, file_path, device='cuda'):
        self.n_agents = n_agents
        self.n_players = n_agents * 2
        self.team_size = n_agents

        self.log_timestamp = log_timestamp

        self.env = env
        time_step = env.reset()
        self.design_dim = len(design_params['0'])
        self.observation_dim_one_agent = sum(v.size for v in time_step.observation[0].values()) + self.design_dim
        self.observation_dim_all_players = self.observation_dim_one_agent * self.n_players
        action_specs = self.env.action_spec()
        self.action_dim_one_agent = action_specs[0].shape[0]
        self.action_dim_all_players = self.action_dim_one_agent * self.n_players

        # self.obsnorm_ind = obsnorm_ind
        self.obsnorm_pop = obsnorm_pop

        self.logging_dict = {'rewards_to_log_train': [],
                            'rewards_to_log_test': []
                            }
        
        self.device = device
        self.video = None # renderhelper.setupVideoWriterHD(self.env)

        model_save_folder = file_path + '/rlmodels'
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)

        self.q_save_path = model_save_folder + '/soccer_roles_matd3_q1.pt'
        self.target_q_save_path = model_save_folder + '/soccer_roles_matd3_q1_target.pt'
        self.policy_save_path = model_save_folder + '/soccer_roles_matd3_policy.pt'

        self.ind_replay_buffer = replay.SimpleReplayBuffer(
            max_replay_buffer_size=500000,
            observation_dim=self.observation_dim_all_players,
            action_dim=self.action_dim_all_players,
            reward_dim=n_agents,
            env_info_sizes={},)
        
        self.pop_replay_buffer = pop_replay_buffer
        self.pop_nets = pop_nets

        self.design_params = design_params

        self.tau = 0.001
        self.agents = matd3.matd3.MATD3(q1_nets=ind_nets['q1_nets'], q2_nets=ind_nets['q2_nets'], target_q1_nets=ind_nets['target_q1_nets'], 
                                        target_q2_nets=ind_nets['target_q2_nets'], 
                                        policy_nets=ind_nets['policy_nets'], target_policy_nets=ind_nets['target_policy_nets'], tau=self.tau, n_agents=n_agents, 
                                        obsnorm=self.obsnorm_pop, device=device, policy_learning_rate=5e-5, q_learning_rate=1e-4, policy_delay=4)
        
        self.pop_agents = matd3.matd3.MATD3(q1_nets=pop_nets['q1_nets'], q2_nets=pop_nets['q2_nets'], target_q1_nets=pop_nets['target_q1_nets'], 
                                        target_q2_nets=pop_nets['target_q2_nets'], 
                                        policy_nets=pop_nets['policy_nets'], target_policy_nets=pop_nets['target_policy_nets'], tau=self.tau, n_agents=n_agents, 
                                        obsnorm=self.obsnorm_pop, device=device, policy_learning_rate=5e-5, q_learning_rate=1e-4, policy_delay=4)

        self.away_policies = self.init_away_team()

    def save_agents(self):
        for i in range(self.n_agents):
            torch.save(self.agents.q1_nets[i].state_dict(), self.q_save_path[:-3] + str(i) + self.q_save_path[-3:])
            torch.save(self.agents.target_q1_nets[i].state_dict(), self.target_q_save_path[:-3] + str(i) + self.target_q_save_path[-3:])
            torch.save(self.agents.policy_nets[i].state_dict(), self.policy_save_path[:-3] + str(i) + self.policy_save_path[-3:])


    def collect_training_data(self, save_pop=False, noise=True):
        time_step = self.env.reset()
        action_specs = self.env.action_spec()

        cum_reward0 = 0
        cum_reward1 = 0
        actions = []

        state = flatten_state_all_players(time_step.observation, self.design_params['0'], self.design_params['1'])

        while not time_step.last():
            actions.clear()

            # Concat all players' states into one, incl. away team
            current_state_torch = torch.from_numpy(state).to(self.device).float()
            state_normed = self.obsnorm_pop.normalise(current_state_torch)

            for i in range(self.n_players):

                if i < self.team_size:
                    # MATD3 home team

                    with torch.no_grad():
                        current_state_player = state_normed[i*self.observation_dim_one_agent:(i+1)*self.observation_dim_one_agent]
                        action = self.agents.policy_nets[i](current_state_player)
                        action = action.cpu().numpy()

                    if noise:
                        action += np.random.normal(scale=0.2, size=action.shape)

                    action = np.clip(action, -1.0, 1.0)
                    action = np.reshape(action, action_specs[i].shape)
                    actions.append(action)

                else:
                    # Away team plays with a pre-trained policy
                    with torch.no_grad():
                        current_state_player = state_normed[i*self.observation_dim_one_agent:((i+1)*self.observation_dim_one_agent - self.design_dim)]
                        action = self.away_policies[i - self.team_size](current_state_player)
                        action = action.cpu().numpy()

                    actions.append(action)


            # Step through env
            time_step = self.env.step(actions)

            # Get reward
            rewards = time_step.reward
            rewards_to_replay = rewards[0:self.team_size]

            # Get next state
            state_t = flatten_state_all_players(time_step.observation, self.design_params['0'], self.design_params['1'])
            
            # Get terminal signal
            terminal = 0

            actions_flat = [item for row in actions for item in row]

            # Add sample to replay buffer
            self.ind_replay_buffer.add_sample(state, actions_flat, rewards_to_replay, state_t, terminal, env_info={})
            if save_pop:
                self.pop_replay_buffer.add_sample(state, actions_flat, rewards_to_replay, 
                                              state_t, terminal, env_info={})

            cum_reward0 += rewards[0]
            cum_reward1 += rewards[1]
            state = state_t

        self.logging_dict['rewards_to_log_train'].append(float(cum_reward0))
        self.logging_dict['rewards_to_log_train'].append(float(cum_reward1))


    def eval_policy(self, is_captured=False):
        if is_captured and self.video is None:
            self.init_video()

        with torch.no_grad():
            time_step = self.env.reset()
            action_specs = self.env.action_spec()

            cum_reward0 = 0
            cum_reward1 = 0
            actions = []

            state = flatten_state_all_players(time_step.observation, self.design_params['0'], self.design_params['1'])   
            
            # TD3 for player 0, random actions for player 1
            while not time_step.last():
                actions.clear()

                current_state_torch = torch.from_numpy(state).to(self.device).float()
                state_normed = self.obsnorm_pop.normalise(current_state_torch)

                for i in range(self.n_players):

                    if i < self.team_size:
                        # MATD3 home team
                        current_state_player = state_normed[i*self.observation_dim_one_agent:(i+1)*self.observation_dim_one_agent]
                        action = self.agents.policy_nets[i](current_state_player)
                        action = action.cpu().numpy()

                        action = np.clip(action, -1.0, 1.0)
                        action = np.reshape(action, action_specs[i].shape)
                        actions.append(action)

                    else:
                        # Away team plays with a pre-trained policy
                        current_state_player = state_normed[i*self.observation_dim_one_agent:((i+1)*self.observation_dim_one_agent - self.design_dim)]
                        action = self.away_policies[i - self.team_size](current_state_player)
                        action = action.cpu().numpy()

                        actions.append(action)


                # Step through env
                time_step = self.env.step(actions)

                # Render env output to video
                if is_captured:
                    self.video.write(renderhelper.grabFrameHD(self.env, camera_id=0))

                # Get reward
                rewards = time_step.reward

                # Get next state
                state_t = flatten_state_all_players(time_step.observation, self.design_params['0'], self.design_params['1'])
                
                cum_reward0 += rewards[0]
                cum_reward1 += rewards[1]
                state = state_t

            self.logging_dict['rewards_to_log_test'].append(float(cum_reward0))
            self.logging_dict['rewards_to_log_test'].append(float(cum_reward1))
        
        if is_captured and not (self.video is None):
            self.end_video()


    def single_training_step(self, file_path, iter, is_captured=False, update_pop_nets=False, update_ind_norm=True):
        """
        This function collects first training data, then performs several
        training iterations and finally evaluates the current policy.
        """
        training_iters_ind = 1000
        training_iters_pop = 250

        if iter % 3 == 0:
            save_pop_rb = True
        else:
            save_pop_rb = False

        # Collect training data
        self.collect_training_data(save_pop=save_pop_rb)

        # Use a wrapper class to split training data of individual nets 70-30 between individual RB and population RB
        wrapped_replay_buffers = replay_wrapper.ReplayWrapper(ind_replay=self.ind_replay_buffer, pop_replay=self.pop_replay_buffer)
        
        if update_pop_nets:
            pop_stats = self.pop_agents.train(self.pop_replay_buffer, train_iters = training_iters_pop, update_norm=True, batch_size = 128, obs_dim = 308)
        
        ind_stats = self.agents.train(wrapped_replay_buffers, train_iters = training_iters_ind, update_norm=update_ind_norm, batch_size = 128, obs_dim = 308)
        
        print("Inference")
        # Collect testing data
        self.eval_policy(is_captured=is_captured)

        # Save stats & rewards into a csv file
        save_logged_data(file_path, episode=iter, rewards_training = self.logging_dict['rewards_to_log_train'], 
                                                rewards_testing = self.logging_dict['rewards_to_log_test'])
        save_logged_stats_ind(file_path, ind_stats, iteration=iter)
        
        if update_pop_nets:
            save_logged_stats_pop(file_path, pop_stats, iteration=iter)

        for key in self.logging_dict:
            self.logging_dict[key].clear()


    def pretrain_pop_nets(self, file_path, iter):
        """
        """
        training_iters_pop = 10000

        pop_stats = self.pop_agents.train(self.pop_replay_buffer, train_iters = training_iters_pop, batch_size = 128, obs_dim = 308)

        save_logged_stats_pop(file_path, pop_stats, iteration=iter)

        for key in self.logging_dict:
            self.logging_dict[key].clear()


    def init_video(self):
        self.video = renderhelper.setupVideoWriterHD(self.env, video_stamp=self.log_timestamp)
    
    def end_video(self):
        # End video writing
        self.video.release()
        cv2.destroyAllWindows()
        self.video = None

    def init_away_team(self, device='cuda'):

        policy_nets = []
        hidden_sizes_policy = [400,300]

        away_input_size = self.observation_dim_one_agent - self.design_dim

        for _ in range(self.team_size):
            policy_nets.append(networks.PolicyNetwork(hidden_sizes = hidden_sizes_policy, 
                                                input_size = away_input_size, output_size=self.action_dim_one_agent).to(device=device))

        policy_save_path0 = "/home/tiia/thesis/rlmodels/Thu_Nov_27_10:40:09_2025/soccer_roles_matd3_policy0.pt"
        policy_save_path1 = "/home/tiia/thesis/rlmodels/Thu_Nov_27_10:40:09_2025/soccer_roles_matd3_policy1.pt"
        saved_policy_state0 = torch.load(policy_save_path0, map_location=device)  
        saved_policy_state1 = torch.load(policy_save_path1, map_location=device)
        policy_nets[0].load_state_dict(copy.deepcopy(saved_policy_state0))
        policy_nets[1].load_state_dict(copy.deepcopy(saved_policy_state1))

        return policy_nets




def flatten_state_all_players(obs, design_params_0, design_params_1):
    all_arrays = []
    num_obs = len(obs)

    for i, ordered_dict in enumerate(obs):
        for array in ordered_dict.values():
            all_arrays.append(array.flatten())

        if i == 0:
            all_arrays.append(np.array(design_params_0))
        elif i == 1:
            all_arrays.append(np.array(design_params_1))
        else:
            all_arrays.append(np.array([0.2828, 0.5657, 0.2828, 0.5657])) # away team always standard design
    

    return np.concatenate(all_arrays)


def save_logged_data(file_path, episode, rewards_training, rewards_testing, n_agents=2):
    """ Saves logged rewards to a csv file."""
    filename = os.path.join(file_path, 'ind_rewards.csv')
    
    # Check if file exists and is empty to write header
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, 'a') as fd:
        cwriter = csv.writer(fd)
        
        if write_header:
            cwriter.writerow(['episode', 'agent_id', 'train_reward', 'test_reward'])
        
        for agent_id in range(n_agents):
            row = [
                episode, 
                agent_id,
                rewards_training[agent_id],
                rewards_testing[agent_id]
            ]
            cwriter.writerow(row)


def save_logged_stats_ind(file_path, stats, iteration, n_agents=2):
    filename = os.path.join(file_path, 'ind_learning_stats.csv')
    
    # Check if file exists and is empty to write header
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    
    with open(filename, 'a') as fd:
        cwriter = csv.writer(fd)
        
        if write_header:
            cwriter.writerow(['iteration', 'agent_id', 'q_loss', 'policy_loss', 'mean_y', 'max_q1',
                            'std_y', 'mean_q1_target', 'std_q1_target', 'mean_reward', 'mean_action_magn'])
        
        # Write data for all agents across all iterations
        for iter_idx, iter_stats in enumerate(stats):
            for agent_id in range(n_agents):
                agent_key = f'agent_{agent_id}'
                if agent_key in iter_stats:
                    agent_data = iter_stats[agent_key]
                    row = [
                        iteration, 
                        agent_id,
                        agent_data.get('q_loss', ''),
                        agent_data.get('policy_loss', ''),
                        agent_data.get('mean_y', ''),
                        agent_data.get('max_q1', ''),
                        agent_data.get('std_y', ''),
                        agent_data.get('mean_q1_target', ''),
                        agent_data.get('std_q1_target', ''),
                        agent_data.get('mean_reward', ''),
                        agent_data.get('mean_action_magn', '')
                    ]
                    cwriter.writerow(row)


def save_logged_stats_pop(file_path, stats, iteration, n_agents=2):
    filename = os.path.join(file_path, 'pop_learning_stats.csv')
    
    # Check if file exists and is empty to write header
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    
    with open(filename, 'a') as fd:
        cwriter = csv.writer(fd)
        
        if write_header:
            cwriter.writerow(['iteration', 'agent_id', 'q_loss', 'policy_loss', 'mean_y', 'max_q1',
                            'std_y', 'mean_q1_target', 'std_q1_target', 'mean_reward', 'mean_action_magn'])
        
        # Write data for all agents across all iterations
        for iter_idx, iter_stats in enumerate(stats):
            for agent_id in range(n_agents):
                agent_key = f'agent_{agent_id}'
                if agent_key in iter_stats:
                    agent_data = iter_stats[agent_key]
                    row = [
                        iteration, 
                        agent_id,
                        agent_data.get('q_loss', ''),
                        agent_data.get('policy_loss', ''),
                        agent_data.get('mean_y', ''),
                        agent_data.get('max_q1', ''),
                        agent_data.get('std_y', ''),
                        agent_data.get('mean_q1_target', ''),
                        agent_data.get('std_q1_target', ''),
                        agent_data.get('mean_reward', ''),
                        agent_data.get('mean_action_magn', '')
                    ]
                    cwriter.writerow(row)


