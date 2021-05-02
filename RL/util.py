import numpy as np
import torch as T

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


## takes in a module and applies the weight initialization from normal distribution
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, y)
        # m.bias.data should be 0
        m.bias.data.fill_(0)

## ttakes in a module and applies the weight initialization from uniform distribution
def weights_init_uniform(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a uniform distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        # m.weight.data shoud be taken from a uniform distribution
        m.weight.data.uniform_(-y, y)
        # m.bias.data should be 0
        m.bias.data.fill_(0)


def update_single_target_network_parameters(network, target_network, tau):
    params = network.named_parameters()
    target_params = target_network.named_parameters()

    params_dict = dict(params)
    target_params_dict = dict(target_params)

    for p in params_dict:
        params_dict[p] = tau*params_dict[p].clone() + \
                    (1-tau)*target_params_dict[p].clone()
    
    return params_dict