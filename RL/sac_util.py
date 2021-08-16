'''Soft Actor Critic utilizes maximum entropy framework. Add entropy to the reward function to 
encourage exploration, and to smooth out the effect of random seeds and episode to episode variation.
Maximizing the total rewards over time PLUS the randomness/stochasticity/entropy of how the agent behaves
'''

import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

from util import ReplayBuffer, update_single_target_network_parameters, initWB


class CriticNetwork(nn.Module):
    def __init__(self, crt_lr, input_dims, n_actions, fc_dims,
                 name='critic', chkpt_dir='tmp/sac', layerNorm=True):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc_dims = fc_dims
        self.model_dir = os.path.join(chkpt_dir, name)
        self.layerNorm=layerNorm

        # Fully connected layers
        self.fcs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.n_fcs = len(self.fc_dims)
        for fc_i in range(self.n_fcs):
            # Fully connected layer
            if fc_i == 0:
                fc = nn.Linear(self.input_dims+self.n_actions, self.fc_dims[0])
            else:
                fc = nn.Linear(self.fc_dims[fc_i-1], self.fc_dims[fc_i])
            initWB(fc)
            self.fcs.append(fc)
            # Layer Normalization over a mini-batch of inputs
            ln = nn.LayerNorm(self.fc_dims[fc_i])
            self.lns.append(ln)

        # Output layer
        self.q = nn.Linear(self.fc_dims[-1], 1)
        initWB(self.q)

        self.optimizer = optim.Adam(self.parameters(), lr=crt_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_action_value = T.cat([state, action], dim=1)
        for fc, ln in zip(self.fcs, self.lns):
            if self.layerNorm:
                state_action_value = F.relu(ln(fc(state_action_value)))
            else:
                state_action_value = F.relu(fc(state_action_value))

        q = self.q(state_action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.model_dir)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.model_dir))


class ValueNetwork(nn.Module):
    def __init__(self, crt_lr, input_dims, fc_dims,
                 name='value', chkpt_dir='tmp/sac', layerNorm=True):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.name = name
        self.model_dir = os.path.join(chkpt_dir, name)
        self.layerNorm = layerNorm

        # Fully connected layers
        self.fcs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.n_fcs = len(self.fc_dims)
        for fc_i in range(self.n_fcs):
            # Fully connected layer
            if fc_i == 0:
                fc = nn.Linear(self.input_dims, self.fc_dims[0])
            else:
                fc = nn.Linear(self.fc_dims[fc_i-1], self.fc_dims[fc_i])
            initWB(fc)
            self.fcs.append(fc)
            # Layer Normalization over a mini-batch of inputs
            ln = nn.LayerNorm(self.fc_dims[fc_i])
            self.lns.append(ln)

        # Output layer
        self.v = nn.Linear(self.fc_dims[-1], 1)
        initWB(self.v)

        self.optimizer = optim.Adam(self.parameters(), lr=crt_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        for fc, ln in zip(self.fcs, self.lns):
            if self.layerNorm:
                state = F.relu(ln(fc(state)))
            else:
                state = F.relu(fc(state))

        v = self.v(state)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.model_dir)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.model_dir))


class ActorNetwork(nn.Module):
    def __init__(self, act_lr, input_dims, n_actions, max_action, fc_dims, 
                name='actor', chkpt_dir='tmp/sac', layerNorm=True):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions
        self.model_dir = os.path.join(chkpt_dir, name)
        self.layerNorm = layerNorm
        self.max_action = max_action
        self.reparam_noise = 1e-6

        # Fully connected layers
        self.fcs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.n_fcs = len(self.fc_dims)
        for fc_i in range(self.n_fcs):
            # Fully connected layer
            if fc_i == 0:
                fc = nn.Linear(self.input_dims, self.fc_dims[0])
            else:
                fc = nn.Linear(self.fc_dims[fc_i-1], self.fc_dims[fc_i])
            initWB(fc)
            self.fcs.append(fc)
            # Layer Normalization over a mini-batch of inputs
            ln = nn.LayerNorm(self.fc_dims[fc_i])
            self.lns.append(ln)

        self.mu = nn.Linear(self.fc_dims[-1], self.n_actions)
        self.sigma = nn.Linear(self.fc_dims[-1], self.n_actions)
        initWB(self.mu)
        initWB(self.sigma)

        self.optimizer = optim.Adam(self.parameters(), lr=act_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        for fc, ln in zip(self.fcs, self.lns):
            if self.layerNorm:
                state = F.relu(ln(fc(state)))
            else:
                state = F.relu(fc(state))

        mu = self.mu(state)
        sigma = self.sigma(state)

        # sigma can not be negative
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()  # has grad_fn, can do actions.backward()
        else:
            actions = probabilities.sample()   # NOT have grad_fn, cannot do actions.backward()

        # 1. scale action to fit the environment
        action = T.tanh(actions) * \
            T.tensor(self.max_action).to(self.device).float()
        # 2. action casted to float so that can be used by T.cat, otherwise it is double type
        # to calculate the loss function
        log_probs = probabilities.log_prob(actions)
        # handle the scaling of action (as we use tanh to scale)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        # 0-axis: batch, 1-axis: components of actions, summed over to get a scalar
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self, modelName):
        T.save(self.state_dict(), self.checkpoint_file + modelName)

    def load_checkpoint(self, modelName):
        self.load_state_dict(T.load(self.checkpoint_file + modelName))


class Agent():
    def __init__(self, input_dims, n_actions, layer_sizes, 
                 act_lr=0.00003, crt_lr=0.0003, gamma=0.99, max_size=1000000, tau=0.005,
                 batch_size=64, reward_scale=1,
                 name='sac', chkpt_dir='tmp/ddpg', layerNorm=True,):
        '''Higher reward scale means higher weights given to rewards ratehr than entropy'''
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_actions = n_actions
        # The env action was scaled to [-1, 1]
        self.max_action = np.ones(self.n_actions)
        # Cannot use env.action_space.high, because env.action_space.high is not real action space
        self.layer_sizes = layer_sizes
        self.layerNorm = layerNorm

        self.memory = ReplayBuffer(max_size, self.input_dims, self.n_actions)

        self.actor = ActorNetwork(act_lr, self.input_dims, self.n_actions, 
                                self.max_action, fc_dims=self.layer_sizes, 
                                name='Actor_'+name, chkpt_dir=chkpt_dir, layerNorm=self.layerNorm)

        self.critic_1 = CriticNetwork(crt_lr, self.input_dims, self.n_actions, self.layer_sizes,
                                    name='critic1_'+name, chkpt_dir=chkpt_dir, layerNorm=self.layerNorm)
        self.critic_2 = CriticNetwork(crt_lr, self.input_dims, self.n_actions, self.layer_sizes,
                                    name='critic2_'+name, chkpt_dir=chkpt_dir, layerNorm=self.layerNorm)

        self.value = ValueNetwork(crt_lr, self.input_dims, self.layer_sizes, 
                                    name='value_'+name, chkpt_dir=chkpt_dir, layerNorm=self.layerNorm)
        self.target_value = ValueNetwork(crt_lr, self.input_dims, self.layer_sizes, 
                                    name='target_value_'+name, chkpt_dir=chkpt_dir, layerNorm=self.layerNorm)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        updated_value = update_single_target_network_parameters(
            self.value, self.target_value, tau
        )

        self.target_value.load_state_dict(updated_value)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
#        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
#        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # Update the value network
        self.value.optimizer.zero_grad()

        value = self.value.forward(state).view(-1)

        actions, log_probs = self.actor.sample_normal(
            state, reparameterize=False)
        log_probs = log_probs.view(-1)
        # Use the action from the current policy, rather than the one stored in the buffer
        q1_new_policy = self.critic_1.forward(state, actions).view(-1)
        q2_new_policy = self.critic_2.forward(state, actions).view(-1)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        value_target = critic_value - log_probs   # - log_probs is entropy

        value_loss = F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Update the critic network
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # action and state are from replay buffer generated by old policy
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)

        value_ = self.target_value.forward(state_).view(-1)
        # value_[done] = 0.0    # In building context, terminal state does not have 0 value
        q_hat = self.scale*reward + self.gamma*value_

        critic_1_loss = F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Update the actor network
        self.actor.optimizer.zero_grad()

        actions, log_probs = self.actor.sample_normal(
            state, reparameterize=True)
        log_probs = log_probs.view(-1)
        # Use the action from the current policy, rather than the one stored in the buffer
        q1_new_policy = self.critic_1.forward(state, actions).view(-1)
        q2_new_policy = self.critic_2.forward(state, actions).view(-1)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.update_network_parameters()

        return critic_loss.item(), actor_loss.item()
