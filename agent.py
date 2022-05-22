import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Any

from model import DuelingQNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learn (in theory) from the environment.

    NOTE: early training dynamics are different than later training dynamics
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        memory_buffer_size: int = int(1e5),
        batch_size: int = 64,
        discount_factor: float = 0.99,
        tau: float = 1e-3,
        lr: float = 5e-4,
        update_every: int = 4,
        p_weighted_init: float = 0.01,
        p_weighted_growth: float = 0.008,
    ):
        """Initialize an Agent object.

        with p_weighted_init=0.01 and p_weighted_growth=0.008, the p_thresh will saturate
        at 0.6 around 5 e in.

        Parameters
        ----------
        state_size : int
            dimension of each state
        action_size : int
            dimension of each action
        seed : int
            random seed
        memory_buffer_size : int, optional
            size of the replay memory buffer, by default int(1e5)
        batch_size : int, optional
            minibach size, by default 64
        discount_factor : float, optional
            discount factor (gamma), by default 0.99
        tau : float, optional
            soft update of target parameters, by default 1e-3
        lr : float, optional
            learning rate, by default 5e-4
        update_every : int, optional
            how often to perform update on the target network, by default 4
        p_weighted_init : float, optional
            initialization diff of p:, by default 0.01
        p_weighted_growth : float, optional
            growth value of p over time : by default 0.008
        """

        # environment
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # replay buffer
        self.memory_buffer_size = memory_buffer_size

        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.tau = tau
        self.lr = lr
        self.update_every = update_every

        # used for setting initial sampling p for any given action
        self.initial_diff = 0.01
        self.p_weighted_init = p_weighted_init
        self.p_weighted_growth = p_weighted_growth

        self.beta = 0.01

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.replaybuff = ReplayBuffer(
            action_size,
            self.memory_buffer_size,
            self.batch_size,
            self.initial_diff,
            self.p_weighted_init,
            self.p_weighted_growth,
            seed,
        )

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Record information and possibly learn"""

        # Save experience in replay memory
        self.replaybuff.add(state, action, reward, next_state, done)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replaybuff) > self.batch_size:
                experiences = self.replaybuff.sample()
                self.learn(experiences, self.discount_factor)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        state: array_like
            current state
        eps: float
            epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # obtain action estimates from q network
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # exploit: select best
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            # explore: perform random choice
            action = random.choice(np.arange(self.action_size))
        return action

    def learn(self, experiences: tuple, discount_factor: float):
        """Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences : Tuple[torch.Tensor]
            tuple of (s, a, r, s', done, w) tuples
        discount_factor : float
            discount factor (gamma)
        """

        states, actions, rewards, next_states, dones, indexes, w = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )

        # Compute Q targets for current states
        Q_targets = rewards + (discount_factor * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets, reduce=False)

        # beta = [0, 1], increase over time
        self.beta = self.beta + (self.beta * 0.01)
        self.beta = min(1.0, self.beta)
        if w is not None:
            rbmv = 1 / len(self.replaybuff.memory)
            # sum to 1
            w = w / w.sum(axis=0, keepdims=1)
            w += 0.00001  # avoid divide by zero
            inv_w = 1 / w
            mult = rbmv * inv_w
            powered = np.power(mult, self.beta)
            powered = powered.astype(np.float32)
            loss = torch.from_numpy(powered) * loss

        rloss = torch.mean(loss)

        mae = torch.abs(Q_expected - Q_targets)

        # minimize the loss
        self.optimizer.zero_grad()
        rloss.backward()
        self.optimizer.step()

        # updates weights to new diffs
        if indexes:
            new_diffs = mae.detach().numpy()
            # also experimented with the mean of the diffs, but did not find
            # useful in this particular experiment
            # > mean_diff = np.mean(new_diffs)
            for i, ind in enumerate(indexes):
                cur_nt = self.replaybuff.memory[ind]
                upated_nt = cur_nt._replace(diff=new_diffs[i][0])
                self.replaybuff.memory[ind] = upated_nt

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model: Any, target_model: Any, tau: float):
        """Soft update model parameters.

        0_target = t*0_local + (1 - t)*0_target

        Parameters
        ----------
        local_model : Any (PyTorch model)
            weights will be copied from
        target_model : Any (PyTorch model)
            weights will be copied to
        tau : float
            interpolation parameter
        """

        # pair params for easy access
        paired_params = zip(target_model.parameters(), local_model.parameters())

        # iterate params and update target params with tau regulated combination of local and target
        for target_param, local_param in paired_params:
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self,
        action_size: int,
        buffer_size: int,
        batch_size: int,
        initial_diff: float,
        p_weighted_init: float,
        p_weighted_growth: float,
        seed: int,
    ):
        """Initialize a ReplayBuffer object.

        Parameters
        ----------
        action_size : int
            dimension of each action
        buffer_size : int
            maximum size of buffer
        batch_size : int
            size of each training batch
        initial_diff : float
            initialization diff of p
        p_weighted_init : float
            initial value of p
        p_weighted_growth : float
            growth multiplier of p
        seed : int
            random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done", "diff"],
        )
        self.seed = random.seed(seed)
        self.initial_diff = initial_diff

        self.p_weighted_growth = p_weighted_growth
        self.p_weighted = p_weighted_init
        self.p_weighted_max = 0.4

        # probability of selecting episodes that were 'surprising'
        self.p_surprise = 0.5
        # self.p_good = 0.5  # surprised in a `good` way (positive outcome)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        note: the initial diff is set to some >0 value such that it is selected
        """
        e = self.experience(
            state, action, reward, next_state, done, diff=self.initial_diff
        )
        self.memory.append(e)

    def sample(self):
        """Sample a batch of experiences from memory.

        stack all entities in each var for batch training
        (e.g. rather than return a sample of tuples)
        """

        if random.random() < self.p_weighted:
            # select experiences from memory with either large or small differences between
            # target and expected

            # weights of each experience to bias sampling, weigh on difference between exp vs obs
            w = np.asarray([m.diff for m in self.memory if m is not None])

            # decide whether to bias sampling from 'surprising', 'unsurprising', or 'random' experiences
            if random.random() > self.p_surprise:
                # invert probability to be more likely to choose events with small differences
                # this is a similar idea to addressing the issue of never sampling 'unsurprising' experiences
                # (other solutions are adding a small value to differences near 0)
                w = w - np.max(w)
                w = np.abs(w)
            # else:
            #     # w is now weighted by positive rewards (will sample high reward actions)
            #     w = np.asarray([m.reward for m in self.memory if m is not None])
            #     # choose based on good/bad, reward
            #     if random.random() > self.p_good:
            #         # invert probability to be more likely to choose events with small differences
            #         # this is a similar idea to addressing the issue of never sampling 'unsurprising' experiences
            #         # (other solutions are adding a small value to differences near 0)
            #         w = w - np.max(w)
            #         w = np.abs(w)

            # return index and experiences, with weighted sampling
            # larger values will be sampled more frequently
            tups = random.choices(
                list(enumerate(self.memory)),
                weights=w,
                k=self.batch_size,
            )

            # the index of the experience and the experience named tuple
            indexes, experiences = zip(*tups)
        else:
            # uniform sample from all experiences
            tups = random.sample(list(enumerate(self.memory)), k=self.batch_size)
            indexes, experiences = zip(*tups)
            w = None

        # increase quadratically then clip to self.p_weighted_max
        self.p_weighted = min(
            self.p_weighted + (self.p_weighted * self.p_weighted_growth),
            self.p_weighted_max,
        )

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones, indexes, w)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
