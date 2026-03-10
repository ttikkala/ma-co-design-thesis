from dm_control import suite
from dm_control import viewer
import numpy as np
import copy

import td3.networks as networks
from matd3.obsnormaliser import ObservationNormaliser

import matd3.replay as replay
import matd3.matd3_away

from tasks.away_task import load_environment

import os
import time
import numpy as np
import torch
import rendering.renderhelper as renderhelper
import datalogging.loggerhelper as loggerhelper
import cv2

"""
This script contains the training logic of training the away team to play a 2v2 game of soccer against the home team.
"""


def save_agents(agents, q_save_path, target_q_save_path, policy_save_path):
    for i in range(n_agents):
        torch.save(agents.q1_nets[i].state_dict(), q_save_path[:-3] + str(i) + q_save_path[-3:])
        torch.save(agents.target_q1_nets[i].state_dict(), target_q_save_path[:-3] + str(i) + target_q_save_path[-3:])
        torch.save(agents.policy_nets[i].state_dict(), policy_save_path[:-3] + str(i) + policy_save_path[-3:])


def flatten_state_all_players(obs):
    all_arrays = []

    for ordered_dict in obs:
        for array in ordered_dict.values():
            all_arrays.append(array.flatten())
        
    return np.concatenate(all_arrays)


def collect_training_data(noise=True):
    
    time_step = env.reset()
    cum_reward0 = 0
    cum_reward1 = 0
    actions = []

    state = flatten_state_all_players(time_step.observation)

    while not time_step.last():
        actions.clear()

        # Concat all players' states into one, incl. away team
        current_state_torch = torch.from_numpy(state).to(device).float()
        state_normed = obsnorm.normalise(current_state_torch)

        for i in range(n_players):

            if i < team_size:
                with torch.no_grad():
                    action = home_policy_nets[i](state_normed[i*observation_dim_one_agent:(i+1)*observation_dim_one_agent])
                    action = action.cpu().numpy()
                actions.append(action)

            else:
                with torch.no_grad():
                    current_state_player = state_normed[i*observation_dim_one_agent:(i+1)*observation_dim_one_agent]
                    action = policy_nets[i - team_size](current_state_player)
                    action = action.cpu().numpy()

                action += np.random.normal(scale=0.2, size=action.shape)

                action = np.clip(action, -1.0, 1.0)
                action = np.reshape(action, action_specs[i].shape)
                actions.append(action)
        

        # Step through env
        time_step = env.step(actions)

        # Get reward
        rewards = time_step.reward
        rewards_to_replay = rewards[team_size:]

        # Get next state
        state_t = flatten_state_all_players(time_step.observation)
        
        # Get terminal signal
        # TODO goals??
        terminal = 0
        # terminal = time_step.last()

        actions_flat = [item for row in actions for item in row]

        # Add sample to replay buffer
        replay_buffer.add_sample(state, actions_flat, rewards_to_replay, state_t, terminal, env_info={})

        cum_reward0 += rewards[2]
        cum_reward1 += rewards[3]
        state = state_t


        rewards_to_log_train0.append(float(cum_reward0))
        rewards_to_log_train1.append(float(cum_reward1))



def eval_policy(is_captured=False):

    with torch.no_grad():
        time_step = env.reset()
        cum_reward0 = 0
        cum_reward1 = 0
        actions = []

        state = flatten_state_all_players(time_step.observation)   
        
        # TD3 for player 0, random actions for player 1
        while not time_step.last():
            actions.clear()

            current_state_torch = torch.from_numpy(state).to(device).float()
            state_normed = obsnorm.normalise(current_state_torch)

            for i in range(n_players):

                if i < team_size:
                    with torch.no_grad():
                        action = home_policy_nets[i](state_normed[i*observation_dim_one_agent:(i+1)*observation_dim_one_agent])
                        action = action.cpu().numpy()
                    actions.append(action)

                else:
                    current_state_player = state_normed[i*observation_dim_one_agent:(i+1)*observation_dim_one_agent]
                    action = policy_nets[i - team_size](current_state_player)
                    action = action.cpu().numpy()

                    action = np.clip(action, -1.0, 1.0)
                    action = np.reshape(action, action_specs[i].shape)
                    actions.append(action)

            # Step through env
            time_step = env.step(actions)

            # Render env output to video
            if is_captured:
                video.write(renderhelper.grabFrame(env, camera_id=0))

            # Get reward
            rewards = time_step.reward

            # Get next state
            state_t = flatten_state_all_players(time_step.observation)
            
            cum_reward0 += rewards[2]
            cum_reward1 += rewards[3]
            state = state_t

            rewards_to_log_test0.append(float(cum_reward0))
            rewards_to_log_test1.append(float(cum_reward1))



def single_training_step(file_path, iter, is_captured=False):
    """
    This function collects first training data, then performs several
    training iterations and finally evaluates the current policy.
    """
    training_iters = 1000
    stats = []
    # Collect training data
    collect_training_data(noise = True)
    
    stats = agents.train(replay_buffer, train_iters = training_iters, batch_size = 128, obs_dim = 304)

    # Collect testing data
    eval_policy(is_captured=is_captured)

    # Save the cum. rewards achieved into a csv file
    loggerhelper.save_logged_data_matd3(file_path, epoch=iter, rewards_training0=rewards_to_log_train0, rewards_training1=rewards_to_log_train1, 
                     rewards_testing0=rewards_to_log_test0, rewards_testing1=rewards_to_log_test1, states=states_to_log, actions=actions_to_log)
    loggerhelper.save_logged_stats_matd3(file_path, stats, iteration=iter)


start_time = time.time()
log_timestamp = time.ctime().replace(' ', '_')

device = 'cuda'

# Load task
team_size = 2
n_agents = team_size # only agents on one side would fall under an instance of MADDPG
n_players = 2 * team_size
env = load_environment(team_size, time_limit = 100.0, disable_walker_contacts=True, pitch_width=32, pitch_height=24)

# Set up writing renders to video
video = renderhelper.setupVideoWriter(env)


# Get env specs
action_specs = env.action_spec()
time_step = env.reset() # step_type, reward, discount, observation
action_dim_one_agent = action_specs[0].shape[0]
action_dim_all_players = action_dim_one_agent * n_players
observation_dim_one_agent = sum(v.size for v in time_step.observation[0].values())
observation_dim_all_players = observation_dim_one_agent * n_players


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

home_policy_nets = []
for _ in range(n_agents):
    home_policy_nets.append(networks.PolicyNetwork(hidden_sizes = hidden_sizes_policy, 
                                            input_size = observation_dim_one_agent, output_size=action_dim_one_agent).to(device=device))
    

# Away team
policy_save_path0 = "/home/tiia/thesis/rlmodels/Thu_Oct__9_15:49:56_2025/soccer_roles_matd3_policy0.pt"
policy_save_path1 = "/home/tiia/thesis/rlmodels/Thu_Oct__9_15:49:56_2025/soccer_roles_matd3_policy1.pt"
saved_policy_state0 = torch.load(policy_save_path0, map_location=device)  
saved_policy_state1 = torch.load(policy_save_path1, map_location=device)
# policy_nets[0].load_state_dict(copy.deepcopy(saved_policy_state0))
# policy_nets[1].load_state_dict(copy.deepcopy(saved_policy_state1))

# q1_save_path0 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_q10.pt"
# q1_save_path1 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_q11.pt"
# saved_q1_state0 = torch.load(q1_save_path0, map_location=device)  
# saved_q1_state1 = torch.load(q1_save_path1, map_location=device)
# q1_nets[0].load_state_dict(copy.deepcopy(saved_q1_state0))
# q1_nets[1].load_state_dict(copy.deepcopy(saved_q1_state1))

# q1_targ_save_path0 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_q1_target0.pt"
# q1_targ_save_path1 = "/home/tiia/thesis/rlmodels/Wed_Nov__5_14:14:44_2025/soccer_roles_matd3_q1_target1.pt"
# saved_q1_targ_state0 = torch.load(q1_targ_save_path0, map_location=device)  
# saved_q1_targ_state1 = torch.load(q1_targ_save_path1, map_location=device)
# target_q1_nets[0].load_state_dict(copy.deepcopy(saved_q1_targ_state0))
# target_q1_nets[1].load_state_dict(copy.deepcopy(saved_q1_targ_state1))

# Home team
home_policy_nets[0].load_state_dict(copy.deepcopy(saved_policy_state0))
home_policy_nets[1].load_state_dict(copy.deepcopy(saved_policy_state1))


# Target update rate
tau = 0.001

# replay buffer from rlkit
replay_buffer = replay.SimpleReplayBuffer(
            max_replay_buffer_size=500000,
            observation_dim=observation_dim_all_players,
            action_dim=action_dim_all_players,
            reward_dim=n_agents,
            env_info_sizes={},)

# store cumulative rewards, states, actions
rewards_to_log_train0 = []
rewards_to_log_test0 = []
rewards_to_log_train1 = []
rewards_to_log_test1 = []

states_to_log = []
actions_to_log = []

# store data
folder = 'experiment_data_test_runs'
file_path = '/scratch/tiia/' + folder + '/' + log_timestamp
# Create experiment folder
if not os.path.exists(file_path):
    os.makedirs(file_path)

# Save torch model
model_save_folder = '/home/tiia/thesis/rlmodels/' + log_timestamp
if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)

policy_save_path = model_save_folder + '/soccer_roles_matd3_policy.pt'
q_save_path = model_save_folder + '/soccer_roles_matd3_q1.pt'
target_q_save_path = model_save_folder + '/soccer_roles_matd3_q1_target.pt'


obsnorm = ObservationNormaliser(shape = observation_dim_all_players, device = device)
obsnorm.set_to_pretrained()

# Init MADDPG collection of agents
agents = matd3.matd3_away.MATD3Away(q1_nets=q1_nets, q2_nets=q2_nets, target_q1_nets=target_q1_nets, target_q2_nets=target_q2_nets, 
                           policy_nets=policy_nets, target_policy_nets=target_policy_nets, tau=tau, n_agents=n_agents, 
                           obsnorm=obsnorm, device=device, policy_learning_rate=5e-5, q_learning_rate=1e-4, policy_delay=4)


# Train
exploration_episodes = 20
training_episodes = 500
n_render = 0


for _ in range(exploration_episodes):
    collect_training_data(noise=True) 
for i in range(training_episodes):
    print("Episode ", i)
    if i >= training_episodes - n_render:
        single_training_step(file_path, iter=i, is_captured=True)
    else:
        single_training_step(file_path, iter=i)

    rewards_to_log_train0.clear()
    rewards_to_log_train1.clear()
    rewards_to_log_test0.clear()
    rewards_to_log_test1.clear()
    states_to_log.clear()
    actions_to_log.clear()


torch.set_printoptions(threshold=100_000)
print("Means, vars \n", obsnorm.get_means_vars())


save_agents(agents, q_save_path = q_save_path, target_q_save_path = target_q_save_path, policy_save_path = policy_save_path)


# End video writing
video.release()
cv2.destroyAllWindows()


end_time = time.time()
print("Elapsed time [s]: ", end_time - start_time)
