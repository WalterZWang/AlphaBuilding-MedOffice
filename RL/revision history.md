### 2021-04-21
1. Add tanh to act network
2. Set act and crt to eval mode when needed because of BatchNorm1d/LayerNorm 

### 2021-04-22
1. Use Phil's implementation of DDPG
2. All time are considered as office hour to calculate reward 

### 2021-04-27
1. Implement SAC

### 2021-04-29
1. Revise the way to handle done state

Difference between DDPG and SAC
1. DDPG does need target_critic because target_critic was used to calculate the target of reward, which was used to update critic network;
   SAC does not need target_critic because the target of reward was calculated using target_value

### 2020-04-30
1. Implement and test TD3

### 2020-05-01
1. Revise TD3: train actor every time step (previously, only every second step) (line 223)
