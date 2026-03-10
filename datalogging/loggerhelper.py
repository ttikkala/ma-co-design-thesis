import os
import csv

""" 
    This script contains helper functions for saving data to csv files.
"""

def save_logged_data(file_path, epoch, rewards_training, rewards_testing, states, actions):
    """ Saves logged rewards to a csv file."""
    
    with open(
        os.path.join(file_path,
            'rewards.csv'), 'a') as fd:
            cwriter = csv.writer(fd)
            cwriter.writerow([epoch])
            cwriter.writerow(rewards_training)
            cwriter.writerow(rewards_testing)
    with open(
        os.path.join(file_path,
            'statesactions.csv'), 'a') as fd:
            cwriter = csv.writer(fd)
            cwriter.writerow([epoch])
            cwriter.writerow(states)
            cwriter.writerow(actions)


def save_logged_stats(file_path, stats):
       filename = os.path.join(file_path, 'learning_stats.csv')
       
       with open(filename, 'a') as fd:
            cwriter = csv.writer(fd)

            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                  cwriter.writerow(['q_loss', 'policy_loss', 'mean_y', 'max_q_value', 
                                    'std_y', 'mean_q_target', 'std_q_target', 'mean_reward'])
            
            for i in range(len(stats)):
                cwriter.writerow(stats[i])


def save_logged_data_matd3(file_path, epoch, rewards_training0, rewards_training1, rewards_testing0, rewards_testing1, states, actions):
    """ Saves logged rewards to a csv file."""
    
    with open(
        os.path.join(file_path,
            'rewards.csv'), 'a') as fd:
            cwriter = csv.writer(fd)
            cwriter.writerow([epoch])
            cwriter.writerow([0])
            cwriter.writerow(rewards_training0)
            cwriter.writerow(rewards_testing0)
            cwriter.writerow([1])
            cwriter.writerow(rewards_training1)
            cwriter.writerow(rewards_testing1)
    with open(
        os.path.join(file_path,
            'statesactions.csv'), 'a') as fd:
            cwriter = csv.writer(fd)
            cwriter.writerow([epoch])
            cwriter.writerow(states)
            cwriter.writerow(actions)


def save_logged_stats_matd3(file_path, stats, iteration, n_agents=2):
    filename = os.path.join(file_path, 'learning_stats.csv')
    
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
