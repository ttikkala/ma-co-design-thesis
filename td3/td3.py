import torch
import numpy as np

""" 
    This script contains the TD3 class which trains the actor and critic of a TD3 agent.
"""



class TD3():

    def __init__(self, q_net_1, q_net_2, target_q_net_1, target_q_net_2, policy_net, target_policy_net, tau, obs_dim, obsnorm,
                 discount = 0.99, policy_learning_rate = 1e-4, q_learning_rate = 1e-3, policy_delay = 2, device='cuda'):
        self.q_net_1 = q_net_1
        self.q_net_2 = q_net_2
        self.target_q_net_1 = target_q_net_1
        self.target_q_net_2 = target_q_net_2
        self.policy_net = policy_net
        self.target_policy_net = target_policy_net
        self.target_update_tau = tau
        self.device = device
        self.policy_delay = policy_delay

        self.discount = discount
        self.policy_learning_rate = policy_learning_rate
        self.q_learning_rate = q_learning_rate

        self.policy_optimizer, self.q_optimizer = get_optimizer(policy=self.policy_net, q_1=self.q_net_1, q_2=self.q_net_2, policy_lr=self.policy_learning_rate, q_lr=self.q_learning_rate)

        self.obsnorm = obsnorm

        soft_update(
            source=self.policy_net ,
            target=self.target_policy_net,
            tau=1.0 # init as exact update of main nets
        )
        soft_update(
            source=self.q_net_1,
            target=self.target_q_net_1,
            tau=1.0
        )   
        soft_update(
            source=self.q_net_2,
            target=self.target_q_net_2,
            tau=1.0
        )

    def train(self, replay, train_iters, batch_size=256):
        stats = []

        self.obsnorm.reset()

        for i in range(train_iters):
            batch = replay.random_batch(batch_size)
            batch = batch_to_torch(batch, device=self.device)
            rewards = batch['rewards']
            terminals = batch['terminals']
            states = batch['observations']
            actions = batch['actions']
            states_t = batch['next_observations']

            self.obsnorm.update_batch(states)
            self.obsnorm.update_batch(states_t)

            normed_states = self.obsnorm.normalise(states)
            normed_states_t = self.obsnorm.normalise(states_t)

            action_noise = (torch.randn_like(actions) * 0.2).clamp(-0.5, 0.5)
            target_actions = torch.clamp(self.target_policy_net(normed_states_t) + action_noise, -1.0, 1.0)

            q_targ_1 = self.target_q_net_1(normed_states_t, target_actions)
            q_targ_2 = self.target_q_net_2(normed_states_t, target_actions)
            y = rewards + self.discount * torch.min(q_targ_1, q_targ_2) * (1 - terminals)

            q_1 = self.q_net_1(normed_states, actions)
            q_2 = self.q_net_2(normed_states, actions)

            q_loss_fn = torch.nn.MSELoss()
            q_loss = q_loss_fn(y, q_1) + q_loss_fn(y, q_2)

            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net_1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.q_net_2.parameters(), max_norm=1.0)
            self.q_optimizer.step()


            if i % self.policy_delay == 0:
                curr_actions = self.policy_net(normed_states)
                policy_loss = - (self.q_net_1(normed_states, curr_actions)).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.policy_optimizer.step()

                # soft updates
                soft_update(self.policy_net, self.target_policy_net, self.target_update_tau)
                soft_update(self.q_net_1, self.target_q_net_1, self.target_update_tau)
                soft_update(self.q_net_2, self.target_q_net_2, self.target_update_tau)


                if i % 400 == 0:
                    q_loss = q_loss.cpu().detach().numpy()
                    policy_loss = policy_loss.cpu().detach().numpy()
                    mean_y = np.mean(y.cpu().detach().numpy())
                    max_q1 = np.max(q_1.cpu().detach().numpy())
                    std_y = np.std(y.cpu().detach().numpy())
                    mean_q1_target = np.mean(q_targ_1.cpu().detach().numpy())
                    std_q1_target = np.std(q_targ_1.cpu().detach().numpy())
                    mean_reward = np.mean(rewards.cpu().detach().numpy())
                    mean_action_magn = np.mean(np.abs(curr_actions.cpu().detach().numpy()))
                    stats.append([q_loss, policy_loss, mean_y, max_q1, std_y, mean_q1_target, std_q1_target, mean_reward, mean_action_magn])

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

def get_optimizer(policy, q_1, q_2, policy_lr, q_lr):
    policy_optimizer = torch.optim.Adam(list(policy.parameters()), lr=policy_lr)
    value_optimizer = torch.optim.Adam(list(q_1.parameters()) + list(q_2.parameters()), lr=q_lr, weight_decay = 1e-3)

    return policy_optimizer, value_optimizer
