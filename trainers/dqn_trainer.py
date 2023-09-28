import argparse
import os
import random
import time
from distutils.util import strtobool
from argparse import ArgumentParser

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from agents.dqn import QNetwork
from .trainer import BaseTrainer
from utils import notice


def get_dqn_config():
    parser = ArgumentParser()
    
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    
    return parser


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQNTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
        args = config['all_args']
        self.lr = args.learning_rate

        self.q_net = QNetwork(self.envs).to(self.device)
        self.targ_net = QNetwork(self.envs).to(self.device)
        self.targ_net.load_state_dict(self.q_net.state_dict())

        self.rb = ReplayBuffer(
            args.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )
    
    def train(self):
        optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        start_time = time.time()

        envs = self.envs
        args = self.all_args
        writer = self.writer
        
        obs, _ = envs.reset()
        
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            if random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                q_values = self.q_net(torch.Tensor(obs).to(self.device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if "episode" not in info:
                        continue
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(truncated):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                if global_step % args.train_frequency == 0:
                    data = self.rb.sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.targ_net(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    old_val = self.q_net(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        writer.add_scalar("losses/td_loss", loss, global_step)
                        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network
                if global_step % args.target_network_frequency == 0:
                    for self.targ_net_param, self.net_param in zip(self.targ_net.parameters(), self.q_net.parameters()):
                        self.targ_net_param.data.copy_(
                            args.tau * self.net_param.data + (1.0 - args.tau) * self.targ_net_param.data
                        )

    def take_actions(self, obs):
        q_values = self.q_net(torch.Tensor(obs).to(self.device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions

    def save(self, version=''):
        path = os.path.join(self.save_dir, f"dqn{version}.pt")
        notice(f"Saving model to {path}")
        torch.save(self.q_net.state_dict(), path)

    def load(self, version=''):
        path = os.path.join(self.model_dir, f"dqn{version}.pt")
        notice(f"Loading model from {path}")
        self.q_net.load_state_dict(torch.load(path))
        self.targ_net.load_state_dict(self.q_net.state_dict())