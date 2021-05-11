import os
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from util import update_single_target_network_parameters, weights_init_normal

def atanh(x):
    return 0.5*T.log((1+x)/(1-x))

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_samples = len(self.states)
        batch_start = np.arange(0, n_samples, self.batch_size)
        indices = np.arange(n_samples, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]   # list of list

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, act_lr, input_dims, n_actions, max_action, 
            fc1_dims=256, fc2_dims=256, init_w=3e-3, name='actor', chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ppo')
        self.max_action = max_action
        self.reparam_noise = 1e-6        

        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU()
        )
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.actor.apply(weights_init_normal)
        self.mu.apply(weights_init_normal)
        self.sigma.apply(weights_init_normal)

        self.optimizer = optim.Adam(self.parameters(), lr=act_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.actor(state)
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)   # sigma can not be negative
        
        return mu, sigma

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, crt_lr, input_dims, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ppo')

        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )
        self.critic.apply(weights_init_normal)

        self.optimizer = optim.Adam(self.parameters(), lr=crt_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        critic = self.critic(state)

        return critic

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, env, 
            act_lr=0.00003, crt_lr=0.0003, gamma=0.99, gae_lambda=0.95, max_action=1,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.input_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.max_action = max_action
        self.reparam_noise = 1e-6 

        self.actor = ActorNetwork(act_lr, self.input_dims, self.n_actions, self.max_action)
        self.critic = CriticNetwork(crt_lr, self.input_dims)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        value = self.critic(state)

        mu, sigma = self.actor(state)
        probabilities = Normal(mu, sigma)

        actions = probabilities.sample()   # NOT have grad_fn, cannot do actions.backward()
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.actor.device).float()  # 1. scale action to fit the environment
                                    # 2. action casted to float so that can be used by T.cat, otherwise it is double type

        log_probs = probabilities.log_prob(actions)             # to calculate the loss function
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)  # handle the scaling of action (as we use tanh to scale)
        log_probs = log_probs.sum(1, keepdim=True)   # 0-axis: batch, 1-axis: components of actions, summed over to get a scalar

        action = T.squeeze(action).detach().numpy()    # remove the dimension which equals 1
        probs = T.squeeze(log_probs).item()  
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr

            # Calculate advantage
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            # delta = reward_arr[:-1] + self.gamma*values[1:] - values[:-1]
            # discount_new = self.gamma*self.gae_lambda
            # for t in range(len(reward_arr)-2, -1, -1):
            #     advantage[t] = advantage[t+1]*discount_new + delta[t]
            
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                action = atanh(actions/T.tensor(self.max_action)).float()   # Inverse to the raw value domain

                mus, sigmas = self.actor(states)
                probabilities = Normal(mus, sigmas)

                new_probs = probabilities.log_prob(action)             # to calculate the loss function
                new_probs -= T.log(1-actions.pow(2)+self.reparam_noise)  # handle the scaling of action (as we use tanh to scale)
                new_probs = new_probs.sum(1, keepdim=True)   # 0-axis: batch, 1-axis: components of actions, summed over to get a scalar

                prob_ratio = (new_probs - old_probs).exp()  # new_probs.exp() / old_probs.exp()

                # weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                # actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                actor_loss = -weighted_clipped_probs.mean()

                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                returns = advantage[batch] + values[batch]
                critic_loss = F.mse_loss(returns, critic_value)

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

        return critic_loss.item(), actor_loss.item()