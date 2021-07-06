from gym_AlphaBuilding import medOff_env
import torch as T
import pandas as pd
import numpy as np
import os


def test(test_algorithm, actor_path, save_path, env):
    # Load the actor
    if test_algorithm == 'ddpg':
        from ddpg_util import Agent
        agent = Agent(env, act_lr=0.000025, crt_lr=0.00025, tau=0.001,
                      batch_size=64,  layer1_size=400, layer2_size=300)
    elif test_algorithm == 'sac':
        from sac_util import Agent
        agent = Agent(env, reward_scale=3)
    elif test_algorithm == 'td3':
        from td3_util import Agent
        agent = Agent(env, act_lr=0.000025, crt_lr=0.00025, tau=0.001,
                      batch_size=64,  layer1_size=256, layer2_size=256)

    agent.actor.load_state_dict(T.load(actor_path))
    agent.actor.eval()

    # Test its performance
    result_all = []
    obs = env.reset()
    done = False
    while not done:
        if test_algorithm == 'ddpg':
            obs = T.tensor(obs, dtype=T.float).to(agent.actor.device)
            act = agent.actor.forward(obs).to(
                agent.actor.device).detach().numpy()
        elif test_algorithm == 'sac':
            obs = T.Tensor([obs]).to(agent.actor.device)
            act, _ = agent.actor.sample_normal(obs, reparameterize=False)
            act = act.detach().numpy()[0]
        elif test_algorithm == 'td3':
            obs = T.tensor(obs, dtype=T.float).to(agent.actor.device)
            act = agent.actor.forward(obs).to(
                agent.actor.device).detach().numpy()

        new_state, reward, done, comments = env.step(act)
        new_state = env.rescale_state(new_state)
        act = env.rescale_action(act)

        result = np.concatenate((new_state, act, comments, np.array([reward])))

        result_all.append(result)

        obs = new_state

    result_all = pd.DataFrame(result_all)
    result_all.columns = env.states_time + env.states_amb + env.states_temp + env.action_names + \
        ['cost_energy', 'cost_comfort', 'temp_min', 'temp_max', 'UDH',
            'fanEnergy', 'coolEnergy', 'heatEnergy'] + ['reward']

    result_all.round(decimals=2)
    result_all.to_csv('{}_test.csv'.format(save_path))


if __name__ == "__main__":

    env = medOff_env.MedOffEnv(building_path='gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                               sim_days=365,
                               step_size=900,
                               sim_year=2015)

    test_algorithm = 'ddpg'
    runName = test_algorithm+'_run2'
    actor_path = os.path.join(
        'tmp', test_algorithm, 'Actor_'+runName)
    save_path = os.path.join(
        'log', 'exp3-higherComfortWeight', runName)

    test(test_algorithm, actor_path, save_path, env)
