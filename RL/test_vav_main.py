from gym_AlphaBuilding import medOff_env
from dist_util import vav_obs
import torch as T
import pandas as pd
import numpy as np
import os


def test(test_algorithm, runName, actorName, env):

    input_dims = 6  # hour, dayOfWeek, outTemp, solar, \
    # dist from sp (inTemp - sp), ahu sat
    n_actions = 2   # flow rate, reheat
    vavs = []

    # Load the actor
    if test_algorithm == 'ddpg':
        from ddpg_util import Agent
        for vav_index in range(9):
            vav = Agent(input_dims, n_actions,
                        act_lr=0.000025, crt_lr=0.00025, tau=0.001,
                        batch_size=64,  layer1_size=32, layer2_size=16)
            actor_path = os.path.join(
                'tmp', test_algorithm, 'Actor_{}vav_{}'.format(actorName, vav_index))
            vav.actor.load_state_dict(T.load(actor_path))
            vav.actor.eval()
            vavs.append(vav)
    # elif test_algorithm == 'sac':
    #     from sac_util import Agent
    #     for _ in range(9):
    #     agent = Agent(env, reward_scale=3)
    # elif test_algorithm == 'td3':
    #     from td3_util import Agent
    #     agent = Agent(env, act_lr=0.000025, crt_lr=0.00025, tau=0.001,
    #                   batch_size=64,  layer1_size=256, layer2_size=256)

    # Test its performance
    result_all = []
    obs = env.reset()
    done = False
    ahuSAT = 12.75
    ahuSAT_scaled = 2.0 * (ahuSAT - env.action_space.low[0]) / \
        (env.action_space.high[0] - env.action_space.low[0]) - 1
    tempSP = 22
    while not done:
        if test_algorithm == 'ddpg':
            vavs_s, _ = vav_obs(obs, tempSP, ahuSAT, env)
            acts = [ahuSAT_scaled]
            for i, vav in enumerate(vavs):
                vav_s = vavs_s[i]
                vav_s = T.tensor(vav_s, dtype=T.float).to(vav.actor.device)
                vav_a = vav.actor.forward(vav_s).to(
                    vav.actor.device).detach().numpy()
                acts.extend(vav_a.tolist())
            acts = np.array(acts)
        # elif test_algorithm == 'sac':
        #     obs = T.Tensor([obs]).to(agent.actor.device)
        #     act, _ = agent.actor.sample_normal(obs, reparameterize=False)
        #     act = act.detach().numpy()[0]
        # elif test_algorithm == 'td3':
        #     obs = T.tensor(obs, dtype=T.float).to(agent.actor.device)
        #     act = agent.actor.forward(obs).to(
        #         agent.actor.device).detach().numpy()

        new_state, reward, done, comments = env.step(acts)
        new_state = env.rescale_state(new_state)
        acts = env.rescale_action(acts)

        result = np.concatenate(
            (new_state, acts, comments, np.array([reward])))

        result_all.append(result)

        obs = new_state

    result_all = pd.DataFrame(result_all)
    result_all.columns = env.states_time + env.states_amb + env.states_temp + env.action_names + \
        ['cost_energy', 'cost_comfort', 'temp_min', 'temp_max', 'UDH',
            'fanEnergy', 'coolEnergy', 'heatEnergy'] + ['reward']

    result_all.round(decimals=2)
    result_all.to_csv('{}_test.csv'.format(runName))


if __name__ == "__main__":

    env = medOff_env.MedOffEnv(building_path='gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                               sim_days=365,
                               step_size=900,
                               sim_year=2015)

    test_algorithm = 'ddpg'
    # actorName = test_algorithm+'_vav'
    actorName = 'dist_ddpg_run1'
    # actor_path = os.path.join(
    #     'tmp', test_algorithm, 'Actor_'+runName)
    runName = os.path.join('log', 'exp4-distVAV', actorName)

    test(test_algorithm, runName, actorName, env)
