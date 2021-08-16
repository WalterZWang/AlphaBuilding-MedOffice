from gym_AlphaBuilding import medOff_env
import torch as T
import numpy as np
import pandas as pd
import os

import argparse

from tensorboardX import SummaryWriter


def train(agent, env, seed, result_save_path, exp_name):

    ResultName = '{}_run{}'.format(exp_name, seed)

    writer = SummaryWriter(comment=ResultName)
    np.random.seed(seed)

    with open(os.path.join(result_save_path, "run{}.log".format(seed)), "a") as f:
        # agent.load_models()

        best_score = np.NINF
        load_checkpoint = False    # load the model and test, not learn
        total_reward_history = []
        total_energy_history = []
        total_comfort_history = []
        total_uncDegHour_history = []
        total_crt_loss_history = []
        total_act_loss_history = []

        if load_checkpoint:
            agent.load_models()

        for episode in range(21):
            obs = env.reset()
            done = False
            total_reward = 0
            total_energy = 0
            total_comfort = 0
            total_uncDegHour = 0
            total_crt_loss = 0
            total_act_loss = 0
            while not done:
                act = agent.choose_action(obs)
                new_state, reward, done, comments = env.step(act)
                total_reward += reward
                agent.remember(obs, act, reward, new_state, int(done))

                if not load_checkpoint:
                    loss = agent.learn()
                    if loss:
                        crt_loss, act_loss = loss
                        # Book keeping
                        energy, comfort, temp_min, temp_max, uncDegHour, fanE, coolE, heatE = comments
                        total_energy += energy
                        total_comfort += comfort
                        total_uncDegHour += uncDegHour
                        total_crt_loss += crt_loss
                        total_act_loss += act_loss

                obs = new_state

            total_reward_history.append(total_reward)
            total_energy_history.append(total_energy)
            total_comfort_history.append(total_comfort)
            total_uncDegHour_history.append(total_uncDegHour)
            total_crt_loss_history.append(total_crt_loss)
            total_act_loss_history.append(total_act_loss)

            f.write("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (episode, total_reward, total_energy,
                                                            total_comfort, total_uncDegHour, total_crt_loss, total_act_loss))
            total_uncDegHour = total_uncDegHour/(9*8760*4*(5/7)*(14/24))
            writer.add_scalar("reward", total_reward, episode)
            writer.add_scalar("energy", total_energy, episode)
            writer.add_scalar("comfort", total_comfort, episode)
            writer.add_scalar("uncDegHour", total_uncDegHour, episode)
            writer.add_scalar("crt_loss", total_crt_loss, episode)
            writer.add_scalar("act_loss", total_act_loss, episode)

            # score is the total reward of 1 episode
            avg_score = np.mean(total_reward_history[-10:])
            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()

            print('episode ', episode, 'score %.2f' % total_reward,
                  'trailing 10 games avg %.3f' % avg_score)

def test(agent, env, seed, exp_name):

    agent.load_models()
    agent.actor.eval()

    # Test its performance
    result_all = []
    obs = env.reset()
    done = False
    while not done:
        if algorithm == 'ddpg':
            obs = T.tensor(obs, dtype=T.float).to(agent.actor.device)
            act = agent.actor.forward(obs).to(
                agent.actor.device).detach().numpy()
        elif algorithm == 'sac':
            obs = T.Tensor([obs]).to(agent.actor.device)
            act, _ = agent.actor.sample_normal(obs, reparameterize=False)
            act = act.detach().numpy()[0]
        elif algorithm == 'td3':
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
    result_all.to_csv('log/{}/run{}_test.csv'.format(exp_name,seed))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Main script to train or test the algoirthm')
    parser.add_argument('-m', '--mode', type=str, metavar='', required=True, help='Only two available: train or test')
    args = parser.parse_args()

    ## Parameters to tune ######################
    algorithm = 'ddpg'
    exp_name = "exp9-test_implementation_ddpg"
    layer_sizes = [400, 300]
    ############################################

    env = medOff_env.MedOffEnv(building_path='gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                               sim_days=365,
                               step_size=900,
                               sim_year=2015)
    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    seed = 1
    actorNetwork_save_path = os.path.join("tmp", algorithm, exp_name, 'run{}'.format(seed))
    if not os.path.exists(actorNetwork_save_path):
        os.makedirs(actorNetwork_save_path)
    result_save_path = os.path.join("log", exp_name)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    if algorithm == 'ddpg':
        from ddpg_util import Agent
        agent = Agent(input_dims, n_actions, layer_sizes,
                    act_lr=0.000025, crt_lr=0.00025, tau=0.001,
                    chkpt_dir=actorNetwork_save_path, name='hvac', layerNorm=True)
    elif algorithm == 'sac':
        from sac_util import Agent
        agent = Agent(input_dims, n_actions, layer_sizes,
                    act_lr=0.00003, crt_lr=0.0003, tau=0.005,
                    chkpt_dir=actorNetwork_save_path, name='hvac', layerNorm=True)
    elif algorithm == 'td3':
        from td3_util import Agent
        agent = Agent(input_dims, n_actions, layer_sizes,
                    act_lr=0.000025, crt_lr=0.00025, tau=0.001,
                    chkpt_dir=actorNetwork_save_path, name='hvac', layerNorm=True)

    if args.mode == 'train':
        train(agent, env, seed, result_save_path, exp_name)
    else:
        test(agent, env, seed, exp_name)