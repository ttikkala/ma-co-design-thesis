import torch
import numpy as np

""" 
    This script contains the MATD3 class which trains the actors and critics of a team of MATD3 agents.
"""



class MATD3():

    def __init__(self, q1_nets, q2_nets, target_q1_nets, target_q2_nets, policy_nets, target_policy_nets, 
                 tau, n_agents, obsnorm, discount=0.99, policy_learning_rate=1e-4, q_learning_rate=1e-3, policy_delay = 2, device='cuda'):
        
        self.q1_nets = q1_nets
        self.q2_nets = q2_nets
        self.target_q1_nets = target_q1_nets
        self.target_q2_nets = target_q2_nets
        self.policy_nets = policy_nets
        self.target_policy_nets = target_policy_nets
        self.target_update_tau = tau
        self.device = device
        self.n_agents = n_agents
        self.policy_delay = policy_delay

        self.discount = discount
        self.policy_learning_rate = policy_learning_rate
        self.q_learning_rate = q_learning_rate

        self.policy_optimizers, self.q_optimizers = get_optimizers(policies=self.policy_nets, q1s=self.q1_nets, q2s=self.q2_nets, 
                                                                   policy_lr=self.policy_learning_rate, q_lr=self.q_learning_rate)

        self.obsnorm = obsnorm

        for i in range(self.n_agents):
            soft_update(
                source=self.policy_nets[i],
                target=self.target_policy_nets[i],
                tau=1.0 # init as exact update of main nets
            )
            soft_update(
                source=self.q1_nets[i],
                target=self.target_q1_nets[i],
                tau=1.0
            )
            soft_update(
                source=self.q2_nets[i],
                target=self.target_q2_nets[i],
                tau=1.0
            )   

    def train(self, replay, train_iters, update_norm=True, batch_size = 64, obs_dim = 304):
        stats = []

        # self.obsnorm.reset()

        for iter in range(train_iters):
            iter_agent_stats = {}

            for i in range(self.n_agents):
                # initialise stuff
                batch = replay.random_batch(batch_size)
                batch = batch_to_torch(batch, device=self.device)
                rewards = batch['rewards']
                terminals = batch['terminals']
                states = batch['observations']
                actions = batch['actions']
                states_t = batch['next_observations']
                
                if update_norm:
                    self.obsnorm.update_batch(states)
                    self.obsnorm.update_batch(states_t)
                
                normed_states = self.obsnorm.normalise(states)
                normed_states_t = self.obsnorm.normalise(states_t)

                action_dim = 8 # hardcoded for now
                local_states_i = normed_states[:, (i * obs_dim):((i+1) * obs_dim)]

                #### Q update ####

                target_actions = []
                for j in range(self.n_agents):
                    action_noise = (torch.randn_like(actions[:,0:action_dim]) * 0.2).clamp(-0.5, 0.5)

                    local_states_j = normed_states[:, (j * obs_dim):((j+1) * obs_dim)]

                    target_action = torch.clamp(self.target_policy_nets[j](local_states_j) + action_noise, -1.0, 1.0)
                    target_actions.append(target_action)
                
                away_team_actions = actions[:, (self.n_agents * action_dim):]

                target_actions = torch.cat(target_actions + [away_team_actions], dim=1)


                # calculate value loss
                q_loss_fn = torch.nn.MSELoss()
                q1 = self.q1_nets[i](normed_states, actions)
                q2 = self.q2_nets[i](normed_states, actions)
                target_q1 = self.target_q1_nets[i](normed_states_t, target_actions)
                target_q2 = self.target_q2_nets[i](normed_states_t, target_actions)
                y = rewards[:,i].view(-1, 1) + self.discount * torch.min(target_q1, target_q2)
                q_loss = q_loss_fn(y, q1) + q_loss_fn(y, q2)

                # optimiser step
                self.q_optimizers[i].zero_grad()
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q1_nets[i].parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.q2_nets[i].parameters(), max_norm=1.0)
                self.q_optimizers[i].step()


                #### policy update ####
                if iter % self.policy_delay == 0:
                    # Get the actions s.t. agent i is using their policy and other agents 
                    # "do what they would typically do" (so get their actions from RB)
                    curr_actions = []
                    for j in range(self.n_agents):
                        if j == i:
                            curr_actions.append(self.policy_nets[i](local_states_i))
                        else:
                            curr_actions.append(actions[:, (j * action_dim):((j+1) * action_dim)])
                    
                    curr_actions = torch.cat(curr_actions + [away_team_actions], dim = 1)

                    # calculate policy loss
                    policy_loss = - (self.q1_nets[i](normed_states, curr_actions)).mean()

                    # optimiser step
                    self.policy_optimizers[i].zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), max_norm=1.0)
                    self.policy_optimizers[i].step()

                if iter % 400 == 0:
                    agent_stats = {
                        'q_loss': q_loss.cpu().detach().numpy(),
                        'policy_loss': policy_loss.cpu().detach().numpy(),
                        'mean_y': np.mean(y.cpu().detach().numpy()),
                        'max_q1': np.max(q1.cpu().detach().numpy()),
                        'std_y': np.std(y.cpu().detach().numpy()),
                        'mean_q1_target': np.mean(target_q1.cpu().detach().numpy()),
                        'std_q1_target': np.std(target_q1.cpu().detach().numpy()),
                        'mean_reward': np.mean(rewards.cpu().detach().numpy()),
                        'mean_action_magn': np.mean(np.abs(curr_actions.cpu().detach().numpy()))
                    }
                    
                    iter_agent_stats[f'agent_{i}'] = agent_stats

            if iter % self.policy_delay == 0:
                # update target networks
                for i in range(self.n_agents):
                    soft_update(self.policy_nets[i], self.target_policy_nets[i], self.target_update_tau)
                    soft_update(self.q1_nets[i], self.target_q1_nets[i], self.target_update_tau)
                    soft_update(self.q2_nets[i], self.target_q2_nets[i], self.target_update_tau)


            # TODO figure out which stats to track and how/when
            if iter % 400 == 0:
                stats.append(iter_agent_stats)

        return stats
        




def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
        target_param.data * (1.0 - tau) + param.data * tau
    )
        
def batch_to_torch(batch, device='cuda'):
    new_dict = {}
    for key,value in batch.items():
        new_dict[key] = torch.from_numpy(batch[key]).to(dtype=torch.float32, device=device)
    return new_dict

def get_optimizers(policies, q1s, q2s, policy_lr, q_lr):

    policy_optimizers = []
    value_optimizers = []

    for i in range(len(policies)):
        policy_optimizers.append(torch.optim.Adam(list(policies[i].parameters()), lr=policy_lr))
        value_optimizers.append(torch.optim.Adam(list(q1s[i].parameters()) + list(q2s[i].parameters()), lr=q_lr, weight_decay = 1e-3))

    return policy_optimizers, value_optimizers

