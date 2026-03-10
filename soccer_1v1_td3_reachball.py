from dm_control import suite
from dm_control import viewer
import numpy as np

import td3.networks as networks
import ddpg.replay as replay
import ddpg.ornsteinuhlenbecknoise as ounoise
import td3.td3 as td3
from td3.obsnormaliser import ObservationNormaliser

from tasks.reachball import load_environment

import os
import time
import numpy as np
import torch
import rendering.renderhelper as renderhelper
import datalogging.loggerhelper as loggerhelper
import cv2
import csv

"""
This script contains the training logic of a single agent playing soccer.
"""

def save_logged_stats(file_path, stats):
       filename = os.path.join(file_path, 'learning_stats.csv')
       
       with open(filename, 'a') as fd:
            cwriter = csv.writer(fd)

            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                  cwriter.writerow(['q_loss', 'policy_loss', 'mean_y', 'max_q1', 
                                    'std_y', 'mean_q1_target', 'std_q1_target', 'mean_reward', 'mean_action_magn'])
            
            for i in range(len(stats)):
                cwriter.writerow(stats[i])


def save_agent(agent, q_save_path, target_q_save_path, policy_save_path):
    torch.save(agent.q_net_1.state_dict(), q_save_path)
    torch.save(agent.target_q_net_1.state_dict(), target_q_save_path)
    torch.save(agent.policy_net.state_dict(), policy_save_path)



def collect_training_data(noise="none", is_captured=False, ou_scaling=1.0):
    
    time_step = env.reset()
    cum_reward = 0
    actions = []

    state = np.concatenate([v.flatten() for v in time_step.observation[0].values()
                            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
                            ])
    

    # TD3 for player 0, random actions for player 1
    while not time_step.last():
        actions.clear()

        for i in range(len(action_specs)):
            if i == 0:
                # TD3 for player 0

                with torch.no_grad():
                    current_state_torch = torch.from_numpy(state).to(device).float()
                    current_state_torch = obsnorm.normalise(current_state_torch)
                    action = policy_net(current_state_torch)
                    action = action.cpu().numpy()

                match noise:
                    case "none":
                        pass
                    case "ou":
                        action += action_noise.sample(scaling=ou_scaling)
                    case "gauss":
                        action += np.random.normal(scale=0.5, size=action.shape)
                    case "uniform":
                        action = np.random.uniform(low=-1.0, high=1.0, size=action.shape)
                    case _:
                        pass

                action = np.clip(action, -1.0, 1.0)
                action = np.reshape(action, action_specs[i].shape)
                actions.append(action)

            if i == 1:
                action = np.random.uniform(action_specs[i].minimum, action_specs[i].maximum, size=action_specs[i].shape)
                actions.append(action)

        # Step through env
        time_step = env.step(actions)

        # Render env output to video
        if is_captured:
            video.write(renderhelper.grabFrame(env, camera_id=6))
            actions_to_log.append(actions[0].tolist())
            states_to_log.append(state.tolist())

        # Get reward
        reward = time_step.reward[0]

        # Get next state
        state_t = np.concatenate([
                v.flatten() for v in time_step.observation[0].values()
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
                ])

        # Get terminal signal
        terminal = 0

        # Add sample to replay buffer
        replay_buffer.add_sample(state, actions[0], reward, state_t, terminal, env_info={})

        cum_reward += reward
        state = state_t

        rewards_to_log_train.append(float(cum_reward))
        
        # print("Reward: ", time_step.reward, " Discount: ", time_step.discount, "\nObservation: ", time_step.observation, "\n")

def eval_policy(is_captured=False):

    with torch.no_grad():
        time_step = env.reset()
        cum_reward = 0
        actions = []

        state = np.concatenate([v.flatten() for v in time_step.observation[0].values()
                                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
                                ])
            
        # TD3 for player 0, random actions for player 1
        while not time_step.last():
            actions.clear()

            for i in range(len(action_specs)):
                if i == 0:
                    # TD3 for player 0

                    current_state_torch = torch.from_numpy(state).to(device).float()
                    current_state_torch = obsnorm.normalise(current_state_torch)
                    action = policy_net(current_state_torch)
                    action = action.cpu().numpy()

                    action = np.clip(action, -1.0, 1.0)
                    # reshape into action_specs[i].shape
                    action = np.reshape(action, action_specs[i].shape)
                    actions.append(action)

                if i == 1:
                    action = np.random.uniform(action_specs[i].minimum, action_specs[i].maximum, size=action_specs[i].shape)
                    # action = np.ones(action_specs[i].shape)
                    actions.append(action)

            # Step through env
            time_step = env.step(actions)

            # Render env output to video
            if is_captured:
                video.write(renderhelper.grabFrame(env, camera_id=6))
                actions_to_log.append(actions[0].tolist())
                states_to_log.append(state.tolist())

            # Get reward
            reward = time_step.reward[0]

            # Get next state
            state_t = np.concatenate([
                    v.flatten() for v in time_step.observation[0].values()
                    if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
                    ])
            
            
            # Get terminal signal
            # TODO: add goal handling
            terminal = 0

            cum_reward += reward
            state = state_t

            rewards_to_log_test.append(float(cum_reward))




def single_training_step(file_path, iter, training_noise="none", is_captured=False):
    """
    This function collects first training data, then performs several
    training iterations and finally evaluates the current policy.
    """
    training_iters = 1000
    stats = []
    # Collect training data
    collect_training_data(is_captured=False, noise = training_noise)
    
    stats = agent.train(replay = replay_buffer, train_iters = training_iters, batch_size = 100)

    # Collect testing data
    eval_policy(is_captured = is_captured)

    # Save the cum. rewards achieved into a csv file
    loggerhelper.save_logged_data(file_path, epoch=iter, rewards_training=rewards_to_log_train, 
                     rewards_testing=rewards_to_log_test, states=states_to_log, actions=actions_to_log)
    save_logged_stats(file_path, stats)




start_time = time.time()
log_timestamp = time.ctime().replace(' ', '_')

device = 'cuda'

# Load task
env = load_environment(team_size = 1)

# Set up writing renders to video
video = renderhelper.setupVideoWriter(env)

# Get env specs
action_specs = env.action_spec()
time_step = env.reset() # step_type, reward, discount, observation
action_dim = action_specs[0].shape[0]
observation_dim = sum(v.size for v in time_step.observation[0].values())
                #  if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number))


# Initialise critic and actor networks and target networks
# hidden_sizes = [400, 300]
hidden_sizes_q = [512,256,128]
hidden_sizes_policy = [400,300]
q_net_1 = networks.QvalueNetwork(hidden_sizes=hidden_sizes_q, 
                                input_size = observation_dim + action_dim).to(device=device)
q_net_2 = networks.QvalueNetwork(hidden_sizes=hidden_sizes_q, 
                                input_size = observation_dim + action_dim).to(device=device)
target_q_net_1 = networks.QvalueNetwork(hidden_sizes=hidden_sizes_q, 
                                        input_size = observation_dim + action_dim).to(device=device)
target_q_net_2 = networks.QvalueNetwork(hidden_sizes=hidden_sizes_q, 
                                        input_size = observation_dim + action_dim).to(device=device)
policy_net = networks.PolicyNetwork(hidden_sizes=hidden_sizes_policy, 
                                    input_size = observation_dim, output_size=action_dim).to(device=device)
target_policy_net = networks.PolicyNetwork(hidden_sizes=hidden_sizes_policy, 
                                           input_size = observation_dim, output_size=action_dim).to(device=device)

# Target update rate
tau = 0.001

# replay buffer from rlkit
replay_buffer = replay.SimpleReplayBuffer(
            max_replay_buffer_size=500000,
            observation_dim=observation_dim,
            action_dim=action_dim,
            env_info_sizes={},)

# store cumulative rewards, states, actions
rewards_to_log_train = []
rewards_to_log_test = []
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
policy_save_path = model_save_folder + '/soccer_reachball_td3_policy.pt'
q_save_path = model_save_folder + '/soccer_reachball_td3_q1.pt'
target_q_save_path = model_save_folder + '/soccer_reachball_td3_q1_target.pt'

best_policy_save_path = model_save_folder + '/best_soccer_reachball_td3_policy.pt'
best_q_save_path = model_save_folder + '/best_soccer_reachball_td3_q1.pt'
best_target_q_save_path = model_save_folder + '/best_soccer_reachball_td3_q1_target.pt'


obsnorm = ObservationNormaliser(shape = observation_dim, device = device)

# Init DDPG agent
agent = td3.TD3(q_net_1, q_net_2, target_q_net_1, target_q_net_2, policy_net, target_policy_net, tau, obs_dim = observation_dim, obsnorm = obsnorm,
                policy_learning_rate=1e-4, q_learning_rate=1e-5, policy_delay = 4)
# agent = ddpg.ddpg.DDPG(q_net, target_q_net, policy_net, target_policy_net, tau, discount=0.986, 
#                        policy_learning_rate=0.0007, q_learning_rate=0.0007)

# Init noise process
# sigma and theta from original DDPG paper
action_noise = ounoise.OrnsteinUhlenbeckNoise(mean=np.zeros_like(action_dim), sigma=0.2, theta=0.15, 
                                                      dt=env.physics.timestep(), init_noise=np.zeros_like(action_dim))

# Train
training_episodes = 900
n_render = 5 # number of episodes to render
top_reward = 0

for _ in range(100):
    collect_training_data(noise="uniform")
for i in range(training_episodes):
    if i >= training_episodes - n_render:
        single_training_step(file_path, iter=i, training_noise="gauss", is_captured=True)
    else:
        single_training_step(file_path, iter=i, training_noise="gauss", is_captured=False)

    total_train_reward = rewards_to_log_train[-1]
    total_test_reward = rewards_to_log_test[-1]
    if total_test_reward > top_reward:
        top_reward = total_test_reward
        save_agent(agent, policy_save_path=best_policy_save_path, 
                   q_save_path=best_q_save_path, target_q_save_path=best_target_q_save_path)
    
    # print("Cumulative training reward: ", rewards_to_log_train[-1])
    # print("Cumulative test reward: ", rewards_to_log_test[-1])
    rewards_to_log_train.clear()
    rewards_to_log_test.clear()
    states_to_log.clear()
    actions_to_log.clear()


    # if i % 50 == 0:
    #     for _ in range(10):
    #         collect_training_data(noise="uniform")




save_agent(agent, policy_save_path=policy_save_path, q_save_path=q_save_path, target_q_save_path=target_q_save_path)

# End video writing
video.release()
cv2.destroyAllWindows()

end_time = time.time()
print("Elapsed time [s]: ", end_time - start_time)
