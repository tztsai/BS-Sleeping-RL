import torch
import numpy as np
import os.path as osp
from utils import *
from .nn.actor_critic import Actor, Critic
from config import *


class MappoPolicy:
    """
    MAPPO Policy class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space,
                 device=torch.device("cpu"), model_dir=None, model_version=""):
        self.obs_space = obs_space
        self.cent_obs_space = cent_obs_space
        self.act_space = act_space
        self.device = device

        self.actor = Actor(args, obs_space, act_space, device)
        self.critic = Critic(args, cent_obs_space, device)

        info(str(self.actor))
        info(str(self.critic))
        
        self._actor_rnn_state = None

        if EVAL and getattr(args, 'count_flops', False):
            from pthflops import count_ops
            self._count_flops = count_ops
            self._flops = 0
        else:
            self._count_flops = None

        if model_dir is not None:
            self.restore(model_dir, model_version)

    def get_actions(self, cent_obs, obs, actor_rnn_states, critic_rnn_states, masks, 
                    available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param actor_rnn_states: (np.ndarray) if actor is RNN, RNN states for actor.
        :param critic_rnn_states: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return actor_rnn_states: (torch.Tensor) updated actor network RNN states.
        :return critic_rnn_states: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, actor_rnn_states = self.actor(
            obs, actor_rnn_states, masks, available_actions, deterministic)
        values, critic_rnn_states = self.critic(cent_obs, critic_rnn_states, masks)
        return values, actions, action_log_probs, actor_rnn_states, critic_rnn_states

    def get_values(self, cent_obs, critic_rnn_states, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param critic_rnn_states: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, critic_rnn_states, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, actor_rnn_states, critic_rnn_states, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param actor_rnn_states: (np.ndarray) if actor is RNN, RNN states for actor.
        :param critic_rnn_states: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, actor_rnn_states, action, masks, available_actions, active_masks)

        values, _ = self.critic(cent_obs, critic_rnn_states, masks)
        
        return values, action_log_probs, dist_entropy

    def save(self, save_dir, version=""):
        notice("Saving models to {}".format(save_dir))
        torch.save(self.actor.state_dict(), osp.join(save_dir, "actor%s.pt" % version))
        torch.save(self.critic.state_dict(), osp.join(save_dir, "critic%s.pt" % version))

    def restore(self, model_dir, version=""):
        model_dir = Path(model_dir)
        actor_file = next(model_dir.glob(f'actor*{version}.pt'))
        critic_file = next(model_dir.glob(f'critic*{version}.pt'))
        notice("Restoring actor network from {}".format(actor_file))
        self.actor.load_state_dict(torch.load(str(actor_file)))
        self.critic.load_state_dict(torch.load(str(critic_file)))

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    @timeit
    @torch.no_grad()
    def act(self, obs, actor_rnn_state=0, masks=None, 
            available_actions=None, deterministic=True):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param actor_rnn_state: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        self.prep_rollout()
        if hasattr(self.actor, 'rnn') and actor_rnn_state is None:
            if self._actor_rnn_state is None:
                num_agents = len(obs)
                rnn_layers = self.actor.rnn.rnn.num_layers
                rnn_dim = self.actor.rnn.rnn.input_size
                actor_rnn_state = np.zeros((num_agents, rnn_layers, rnn_dim), dtype=np.float32)
            else:
                actor_rnn_state = self._actor_rnn_state
        if masks is None:
            masks = np.ones((1, 1), dtype=np.float32)
        actions, _, actor_rnn_state = self.actor(
            obs, actor_rnn_state, masks, available_actions, deterministic)
        if self._count_flops:
            # self.actor.act._deterministic = True
            self._flops += self._count_flops(self.actor, obs)
        self._actor_rnn_state = actor_rnn_state
        return actions.cpu().numpy()

    def get_actor_rnn_state(self):
        return self._actor_rnn_state
