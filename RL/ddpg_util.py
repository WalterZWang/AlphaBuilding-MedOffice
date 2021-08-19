import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from util import ReplayBuffer, update_single_target_network_parameters, initWB


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)



class CriticNetwork(nn.Module):
    def __init__(self, crt_lr, input_dims, n_actions, fc_dims, 
                name='Critic', chkpt_dir='tmp/ddpg', layerNorm=True):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc_dims = fc_dims
        self.model_dir = os.path.join(chkpt_dir, name)       
        self.layerNorm = layerNorm

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
            if self.layerNorm:
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
        
        if self.layerNorm:
            for fc, ln in zip(self.fcs, self.lns):
                state_action_value = ln(F.relu(fc(state_action_value)))
        else:
            for fc in self.fcs:
                state_action_value = F.relu(fc(state_action_value))

        q = self.q(state_action_value)

        return q

    def save_checkpoint(self):
        checkpoint = {'input_size': self.input_dims,
                      'output_size': self.n_actions,
                      'hidden_layers': self.fc_dims,
                      'state_dict': self.state_dict()}
        T.save(checkpoint, self.model_dir)

    def load_checkpoint(self):
        checkpoint = T.load(self.model_dir)
        self.load_state_dict(checkpoint['state_dict'])


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc_dims, 
                name='Actor', chkpt_dir='tmp/ddpg', layerNorm=True):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc_dims = fc_dims
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

        self.mu = nn.Linear(self.fc_dims[-1], self.n_actions)
        initWB(self.mu)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        for fc, ln in zip(self.fcs, self.lns):
            if self.layerNorm:
                state = ln(F.relu(fc(state)))
            else:
                state = F.relu(fc(state))

        x = T.tanh(self.mu(state))

        return x

    def save_checkpoint(self):
        checkpoint = {'input_size': self.input_dims,
                      'output_size': self.n_actions,
                      'hidden_layers': self.fc_dims,
                      'state_dict': self.state_dict()}

        T.save(checkpoint, self.model_dir)

    def load_checkpoint(self):
        checkpoint = T.load(self.model_dir)      
        self.load_state_dict(checkpoint['state_dict'])

class Agent(object):
    def __init__(self, input_dims, n_actions, layer_sizes,
                 act_lr=0.00001, crt_lr=0.0001, tau=0.001, 
                 gamma=0.99, max_size=1000000, batch_size=64,
                 chkpt_dir='tmp/ddpg', name='ddpg', layerNorm=True):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.layer_sizes = layer_sizes
        self.layerNorm = layerNorm
        self.gamma = gamma   # discount factor
        self.tau = tau       # target network updating weight
        self.memory = ReplayBuffer(max_size, self.input_dims, self.n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(act_lr, self.input_dims, self.n_actions, self.layer_sizes, 
                                name='Actor_'+name, chkpt_dir=chkpt_dir, layerNorm=self.layerNorm)
        self.critic = CriticNetwork(crt_lr, self.input_dims, self.n_actions, self.layer_sizes, 
                                name='Critic_'+name, chkpt_dir=chkpt_dir, layerNorm=self.layerNorm)

        self.target_actor = ActorNetwork(act_lr, self.input_dims, self.n_actions, self.layer_sizes, 
                                name='TargetActor_'+name, chkpt_dir=chkpt_dir, layerNorm=self.layerNorm)
        self.target_critic = CriticNetwork(crt_lr, self.input_dims, self.n_actions, self.layer_sizes, 
                                name='TargetCritic_'+name, chkpt_dir=chkpt_dir, layerNorm=self.layerNorm)

        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(
            observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise()*0.05,
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        # done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        # calculate target
        self.target_actor.eval()
        self.target_critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(
            new_state, target_actions).view(-1)
        # critic_value_[done] = 0.0    # In building context, terminal state does not have value of 0
        target = reward + self.gamma*critic_value_

        # train critic
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_value = self.critic.forward(state, action).view(-1)
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # train actor
        self.critic.eval()
        self.actor.train()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
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
        updated_critic = update_single_target_network_parameters(
            self.critic, self.target_critic, tau
        )

        self.target_actor.load_state_dict(updated_actor)
        self.target_critic.load_state_dict(updated_critic)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        # self.target_actor.save_checkpoint(modelName)
        self.critic.save_checkpoint()
        # self.target_critic.save_checkpoint(modelName)

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        # self.target_actor.load_checkpoint(modelName)
        self.critic.load_checkpoint()
        # self.target_critic.load_checkpoint(modelName)
