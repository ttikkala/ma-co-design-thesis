import torch
import numpy as np


""" 
    This script contains the DDPG class which trains the actor and critic of a DDPG agent.
"""

class DDPG():

    def __init__(self, q_net, target_q_net, policy_net, target_policy_net, tau, discount = 0.99, policy_learning_rate = 1e-4, q_learning_rate = 1e-3, device='cuda'):
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.policy_net = policy_net
        self.target_policy_net = target_policy_net
        self.target_update_tau = tau
        self.device = device

        self.discount = discount
        self.policy_learning_rate = policy_learning_rate
        self.q_learning_rate = q_learning_rate

        self.policy_optimizer, self.q_optimizer = get_optimizer(policy=self.policy_net, value=self.q_net, policy_lr=self.policy_learning_rate, q_lr=self.q_learning_rate)

        soft_update(
            source=self.policy_net ,
            target=self.target_policy_net,
            tau=1.0 # init as exact update of main nets
        )
        soft_update(
            source=self.q_net ,
            target=self.target_q_net,
            tau=1.0
        )   

    def train(self, replay, batch_size=256, log=False):
        # initialise stuff
        batch = replay.random_batch(batch_size)
        batch = batch_to_torch(batch, device=self.device)
        rewards = batch['rewards']
        terminals = batch['terminals']
        states = batch['observations']
        actions = batch['actions']
        states_t = batch['next_observations']


        # calculate value loss
        q_loss_fn = torch.nn.MSELoss()
        q = self.q_net(states, actions)
        policy_target_next = self.target_policy_net(states_t)
        q_target = self.target_q_net(states_t, policy_target_next) #* (1 - terminals)
        y = rewards + self.discount * q_target
        q_loss = q_loss_fn(y, q)

        # optimiser step
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()


        # calculate policy loss
        policy_loss = - (self.q_net(states, self.policy_net(states))).mean()

        # optimiser step
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()


        # update networks
        soft_update(self.policy_net, self.target_policy_net, self.target_update_tau)
        soft_update(self.q_net, self.target_q_net, self.target_update_tau)


        stats = []
        if log:
            q_loss = q_loss.cpu().detach().numpy()
            policy_loss = policy_loss.cpu().detach().numpy()
            mean_y = np.mean(y.cpu().detach().numpy())
            max_q = np.max(q.cpu().detach().numpy())
            std_y = np.std(y.cpu().detach().numpy())
            mean_q_target = np.mean(q_target.cpu().detach().numpy())
            std_q_target = np.std(q_target.cpu().detach().numpy())
            mean_reward = np.mean(rewards.cpu().detach().numpy())
            stats.extend([q_loss, policy_loss, mean_y, max_q, std_y, mean_q_target, std_q_target, mean_reward])

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

def get_optimizer(policy, value, policy_lr, q_lr):
    policy_optimizer = torch.optim.Adam(list(policy.parameters()), lr=policy_lr)
    value_optimizer = torch.optim.Adam(list(value.parameters()), lr=q_lr, weight_decay = 1e-3)

    return policy_optimizer, value_optimizer
