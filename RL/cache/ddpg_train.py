from gym_AlphaBuilding import medOff_env
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import time

from lib import models, utils

import torch
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

GAMMA = 0.99
BATCH_SIZE = 64
LR_crt = 1e-4
LR_act = LR_crt/2
REPLAY_SIZE = 500000
REPLAY_INITIAL = 2700   # start training with one month of data
ALPHA = 1-1e-4           # update rate of target network 

TEST_ITERS = 2 # compute test evaluation every 2 episodes

CUDA = False

RunName = "all_on_tau_0.1"

def test_net(act_net, env, episode=1, device="cpu"):
   
    rewards = 0.0
    energies = 0.0
    comforts = 0.0
    uncDegHours = 0.0
    tempMin = []
    tempMax = []
    steps = 0
    for _ in range(episode):
        obs = env.reset()

        while True:
            obs_v = utils.float32_preprocessor([obs]).to(device)  # delete [:-3]
            mu_v = act_net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)

            obs, reward, done, comments = env.step(action)

            energy, comfort, temp_min, temp_max, uncDegHour = comments

            rewards += reward
            energies += energy
            comforts += comfort
            uncDegHours += uncDegHour
            tempMin.append(temp_min)
            tempMax.append(temp_max)
            steps += 1

            if done:
                break
        tempMinNP = np.array(tempMin)
        tempMaxNP = np.array(tempMax)
    return rewards / episode, energies / episode, comforts / episode, uncDegHours/episode, tempMinNP, tempMaxNP

if __name__ == "__main__":
    device = torch.device("cuda" if CUDA else "cpu")

    save_path = os.path.join("log", "ddpg-" + RunName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    writer = SummaryWriter(comment="-ddpg_" + RunName)

    env = medOff_env.MedOffEnv(building_path = 'gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                                    sim_days = 365,
                                    step_size = 900,
                                    sim_year = 2015,
                                    tz_name = 'America/Los_Angeles')

    test_env = medOff_env.MedOffEnv(building_path = 'gym_AlphaBuilding/fmuModel/v1_fmu.fmu',
                                    sim_days = 365,
                                    step_size = 900,
                                    sim_year = 2015,
                                    tz_name = 'America/Los_Angeles')

    # The last 3 obs ('fanEnergy', 'coolEnergy', 'heatEnergy') are not states 
    act_net = models.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = models.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    tgt_act_net = utils.TargetNet(act_net)
    tgt_crt_net = utils.TargetNet(crt_net)

    # initiate optimizer for the act and crt network and the agent
    act_opt = optim.Adam(act_net.parameters(), lr=LR_act)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LR_crt)
    act_lr_sch = optim.lr_scheduler.ExponentialLR(act_opt, BETA)
    crt_lr_sch = optim.lr_scheduler.ExponentialLR(crt_opt, BETA)
    agent = models.AgentDDPG(act_net, device=device)


    exp_source = utils.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = utils.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)


    exp_idx = 0
    best_reward = None
    episode = 0

    with utils.RewardTracker(writer) as tracker:
        with utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                buffer.populate(1)
                # rewards_steps = exp_source.pop_rewards_steps()
                # if rewards_steps:
                #     rewards, steps = zip(*rewards_steps)
                #     # tb_tracker.track("episode_steps", steps[0], exp_idx)
                #     # tracker.reward("rewards", rewards, episodes)
                #     episodes += 1
                #     # print("episodes: %d" % (episodes))

                if len(buffer) < REPLAY_INITIAL:
                    continue

                exp_idx += 1

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch_ddqn(batch)

                # train critic
                crt_opt.zero_grad()
                print(type(states_v))
                print(actions_v)
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, exp_idx)
                tb_tracker.track("loss_critic_ref", q_ref_v.mean(), exp_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, exp_idx)

                tgt_act_net.alpha_sync(alpha=ALPHA)
                tgt_crt_net.alpha_sync(alpha=ALPHA)

                if exp_idx % ((env.n_steps -1) * TEST_ITERS) == 0:
                    episode += TEST_ITERS
                    ts = time.time()
                    rewards, energy, comfort, uncDegHour, tempMin, tempMax = test_net(act_net, test_env, 1, device=device)
                    print("Test done in %.2f sec, reward %.3f"% (
                        time.time() - ts, rewards))
                    writer.add_scalar("test_reward", rewards, episode)
                    writer.add_scalar("test_energy",energy, episode)
                    writer.add_scalar("test_comfort",comfort, episode)
                    writer.add_scalar("test_uncDegHour",uncDegHour, episode)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, exp_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

    pass
