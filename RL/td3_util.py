'''TD3 is short for Twin Delayed Deep Deterministic Policy Gradients 
Designed to mitigate the function approximation error in actor critic, including 
    over-estimation bias, variance and bias due to the accumulation of errors because
    of using a deep neural network to approximate the agent's policy and value 
    function
Modifications compared to DDPG:
1. delay the policy update by every other step
2. smooth the calculation of target action (by adding a random noise) when learning the critic
3. perform a double Q learning type modication of learning rule for a critic update

Paper of this method: 
http://proceedings.mlr.press/v80/fujimoto18a.html
'''

import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os

from util import ReplayBuffer, update_single_target_network_parameters, weights_init_normal


class CriticNetwork(nn.Module):
    def __init__(self, crt_lr, input_dims, n_actions,
                 fc1_dims=256, fc2_dims=256, init_w=3e-3,
                 name='critic', chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_')

        self.critic = nn.Sequential(
            nn.Linear(self.input_dims + n_actions, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1)
        )
        self.critic.apply(weights_init_normal)
        # self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.q = nn.Linear(self.fc2_dims, 1)
        # self.q.weight.data.uniform_(-init_w, init_w)
        # self.q.bias.data.uniform_(-init_w, init_w)

        self.optimizer = optim.Adam(self.parameters(), lr=crt_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = T.cat([state, action], dim=1)
        # q_action_value = self.fc1(action_value)
        # q_action_value = F.relu(q_action_value)
        # q_action_value = self.fc2(q_action_value)
        # q_action_value = F.relu(q_action_value)
        # q = self.q(q_action_value)

        q = self.critic(action_value)

        return q

    def save_checkpoint(self, modelName):
        T.save(self.state_dict(), self.checkpoint_file + modelName)

    def load_checkpoint(self, modelName):
        self.load_state_dict(T.load(self.checkpoint_file + modelName))


class ActorNetwork(nn.Module):
    def __init__(self, act_lr, input_dims, n_actions,
                 fc1_dims=256, fc2_dims=256, init_w=3e-3,
                 name='actor', chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_')

        self.actor = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions)
        )
        self.actor.apply(weights_init_normal)

        # self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # self.mu.weight.data.uniform_(-init_w, init_w)
        # self.mu.bias.data.uniform_(-init_w, init_w)

        self.optimizer = optim.Adam(self.parameters(), lr=act_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # prob = self.fc1(state)
        # prob = F.relu(prob)
        # prob = self.fc2(prob)
        # prob = F.relu(prob)
        # prob = self.mu(prob)

        prob = self.actor(state)

        mu = T.tanh(prob)

        return mu

    def save_checkpoint(self, modelName):
        T.save(self.state_dict(), self.checkpoint_file + modelName)

    def load_checkpoint(self, modelName):
        self.load_state_dict(T.load(self.checkpoint_file + modelName))


class Agent():
    def __init__(self, env, act_lr, crt_lr, tau,
                 gamma=0.99, max_size=1000000, batch_size=64,
                 layer1_size=256, layer2_size=256,
                 update_actor_interval=2,
                 noise=0.1, noise_targetAct=0.2):
        self.input_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.max_action = 1
        self.min_action = -1
        self.memory = ReplayBuffer(max_size, self.input_dims, self.n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(act_lr, self.input_dims, self.n_actions,
                                  layer1_size, layer2_size, name='actor')

        self.critic_1 = CriticNetwork(crt_lr, self.input_dims, self.n_actions,
                                      layer1_size, layer2_size, name='critic_1')
        self.critic_2 = CriticNetwork(crt_lr, self.input_dims, self.n_actions,
                                      layer1_size, layer2_size, name='critic_2')

        self.target_actor = ActorNetwork(act_lr, self.input_dims, self.n_actions,
                                         layer1_size, layer2_size, name='target_actor')
        self.target_critic_1 = CriticNetwork(crt_lr, self.input_dims, self.n_actions,
                                             layer1_size, layer2_size, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(crt_lr, self.input_dims, self.n_actions,
                                             layer1_size, layer2_size, name='target_critic_2')

        self.noise = noise
        self.noise_targetAct = noise_targetAct
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                 dtype=T.float).to(self.actor.device)

        mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        # done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
            T.clamp(T.tensor(np.random.normal(
                scale=self.noise_targetAct)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action,
                                 self.max_action)

        q1_ = self.target_critic_1.forward(state_, target_actions).view(-1)
        q2_ = self.target_critic_2.forward(state_, target_actions).view(-1)
        # q1_[done] = 0.0   # In building context, the terminal state does not have 0 value
        # q2_[done] = 0.0
        critic_value_ = T.min(q1_, q2_)
        target = reward + self.gamma*critic_value_

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1
        # if self.learn_step_cntr % self.update_actor_iter != 0:
        #     return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(
            state, self.actor.forward(state))  # can also use the mean
        # of actor_q1_loss and actor_q2_loss, but it would be slower and does not really matter
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return critic_loss.item(), actor_loss.item()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        updated_actor = update_single_target_network_parameters(
            self.actor, self.target_actor, tau
        )
        updated_critic_1 = update_single_target_network_parameters(
            self.critic_1, self.target_critic_1, tau
        )
        updated_critic_2 = update_single_target_network_parameters(
            self.critic_2, self.target_critic_2, tau
        )

        self.target_actor.load_state_dict(updated_actor)
        self.target_critic_1.load_state_dict(updated_critic_1)
        self.target_critic_2.load_state_dict(updated_critic_2)

    def save_models(self, modelName):
        print('.... saving models ....')
        self.actor.save_checkpoint(modelName)
        self.target_actor.save_checkpoint(modelName)
        self.critic_1.save_checkpoint(modelName)
        self.critic_2.save_checkpoint(modelName)
        self.target_critic_1.save_checkpoint(modelName)
        self.target_critic_2.save_checkpoint(modelName)

    def load_models(self, modelName):
        print('.... loading models ....')
        self.actor.load_checkpoint(modelName)
        self.target_actor.load_checkpoint(modelName)
        self.critic_1.load_checkpoint(modelName)
        self.critic_2.load_checkpoint(modelName)
        self.target_critic_1.load_checkpoint(modelName)
        self.target_critic_2.load_checkpoint(modelName)
