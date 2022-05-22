import pickle
from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent


class Trainer:
    def __init__(
        self,
        n_episodes=1500,
        max_t=300,
        eps_init=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        target_score=15,
    ):
        """Create an environment and train an agent

        len(env_info.agents) == 1
        action_size = brain.vector_action_space_size == 4
        state = env_info.vector_observations[0] --> [1., 0, 0, 0.8, ..., 0., 0.]
        len(state) == 37

        Parameters
        ----------
        n_episodes : int, optional
            maximum number of training episodes, by default 1500
        max_t : int, optional
            maximum number of timesteps per episode, 300 by default for env, by default 300
        eps_init : float, optional
            initial value of epsilon, for epsilon-greedy action selection, by default 1.0
        eps_end : float, optional
            minimum value of epsilon, by default 0.01
        eps_decay : float, optional
            multiplicative factor (per episode) for decreasing epsilon, by default 0.995
        target_score : int, optional
            score threshold to terminate training, by default 15
        """

        self.n_episodes = n_episodes
        self.max_t = max_t

        self.target_score = target_score  # 13
        self.window_size = 100

        self.scores = []
        self.scores_window = deque(maxlen=self.window_size)  # last 100 scores

        self.eps = eps_init
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.save_path_fmt_score = "./params/best_params_{}.pth"

        # environment
        self.env = UnityEnvironment(file_name="./Banana.app")

        # get the default brain
        brain_name = self.env.brain_names[0]
        brain = self.env.brains[brain_name]

        # reset the environment
        env_info = self.env.reset(train_mode=True)[brain_name]

        # create agent
        self.agent = Agent(
            state_size=len(env_info.vector_observations[0]),  # 37
            action_size=brain.vector_action_space_size,  # 4
            seed=0,
        )

    def _check_done_save_params(self, e):
        if np.mean(self.scores_window) >= self.target_score:
            print(
                f"\nEnv solved in {e:d} episodes!\tAvg Score: {np.mean(self.scores_window):.2f}"
            )
            save_path = self.save_path_fmt_score.format(
                int(np.mean(self.scores_window))
            )
            torch.save(self.agent.qnetwork_local.state_dict(), save_path)
            print(f"params saved: {save_path}")
            return True
        return False

    def _terminal_monitor(self, e):
        print(
            f"\rEpisode {e}, \tAverage Score: {np.mean(self.scores_window) :.2f}",
            end="",
        )
        if e % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    e, np.mean(self.scores_window)
                )
            )

    def _unpack_env_info(self, env_info):
        cur_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return cur_state, reward, done

    def _update_score(self, score):
        self.scores_window.append(score)
        self.scores.append(score)

    def cleanup(self):
        # close environment
        self.env.close()
        print("env closed")

    def train(self):
        """train the agent for n_episodes and return the scores"""
        # run episode
        for e in range(1, self.n_episodes + 1):
            brain_name = self.env.brain_names[0]
            env_info = self.env.reset(train_mode=True)[brain_name]

            # set initial state
            state, _, _ = self._unpack_env_info(env_info)
            score = 0
            for _ in range(self.max_t):

                # use state to determine action
                action = self.agent.act(state, self.eps)

                # send the action to the environment
                env_info = self.env.step(action)[brain_name]

                # unpack reward and next state
                next_state, reward, done = self._unpack_env_info(env_info)

                # record step information to agent, possibly learn
                self.agent.step(state, action, reward, next_state, done)

                # update state
                state = next_state

                score += reward
                if done:
                    break

            # decrease epsilon
            self.eps = max(self.eps_end, self.eps_decay * self.eps)

            # update score, display in terminal
            self._update_score(score)
            self._terminal_monitor(e)

            # Maybe save params
            if self._check_done_save_params(e):
                break

        return self.scores


if __name__ == "__main__":
    t = Trainer()
    scores = t.train()
    print(f"done: {scores}")

    # perserve score log
    with open("scores.pkl", "wb") as f:
        pickle.dump(scores, f)
