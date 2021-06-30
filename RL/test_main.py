from gym_AlphaBuilding import medOff_env
import torch as T
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":

    env = medOff_env.MedOffEnv(building_path = 'gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                               sim_days = 365,
                               step_size = 900,
                               sim_year = 2015,
                               tz_name = 'America/Los_Angeles',
                               occupied_hour = (6, 20),
                               weight_reward = (0.2, 0.01))

    # Load the actor
    from ddpg_util import Agent
    agent = Agent(env, act_lr=0.000025, crt_lr=0.00025, tau=0.001, 
                  batch_size=64,  layer1_size=400, layer2_size=300)
    agent.actor.load_state_dict(T.load(agent.actor.checkpoint_file))
    agent.actor.eval()

    # Test its performance
    result_all = []
    obs = env.reset()
    done = False
    while not done:
        obs = T.tensor(obs, dtype=T.float).to(agent.actor.device)
        act = agent.actor.forward(obs).to(agent.actor.device).detach().numpy()
        new_state, reward, done, comments = env.step(act)
        result = np.append(new_state, act)
        result_all.append(result)

        obs = new_state
    
    result_all = pd.DataFrame(result_all)
    result_all.columns = env.states_time + env.states_amb + env.states_temp + env.action_names
    
    result_all.round(decimals=2)
    result_all.to_csv('log/ddpg_test_csv')