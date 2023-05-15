import copy

import gym
import numpy as np
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from gym_art.quadrotor_multi.tests.plot_v_value_2d import plot_v_value_2d


class V_ValueMapWrapper(gym.Wrapper):
    def __init__(self, env, model, render_mode=None):
        """A wrapper that visualize V-value map at each time step"""
        gym.Wrapper.__init__(self, env)
        self._render_mode = render_mode
        self.curr_obs = None
        self.model = model

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.curr_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, info, terminated, truncated = self.env.step(action)
        self.curr_obs = obs
        return obs, reward, info, terminated, truncated

    def render(self):
        if self._render_mode == 'rgb_array':
            frame = self.env.render()
            if frame is not None:
                if len(frame.shape) == 4:
                    num_agents, width, height = frame.shape[0], frame.shape[1], frame.shape[2]
                    obs = dict(obs=np.array(self.curr_obs))
                    normalized_obs = prepare_and_normalize_obs(self.model, obs)
                    v_value_maps = self.get_v_value_map_2d_multi(normalized_obs, num_agents, width, height)
                    frame = np.concatenate((frame, v_value_maps), axis=2)
                    frame = frame.reshape((2, num_agents // 2,) + frame.shape[1:])
                    frame = np.transpose(frame, (0, 2, 1, 3, 4))
                    frame = frame.reshape(2 * frame.shape[1], -1, frame.shape[4])
                    # frame = np.concatenate((frame, v_value_maps), axis=1)
                    # frame = np.transpose(frame, (1, 0, 2, 3))
                    # frame = frame.reshape(frame.shape[0], -1, frame.shape[3])

                else:
                    width, height = frame.shape[0], frame.shape[1]
                    obs = dict(obs=np.array(self.curr_obs))
                    normalized_obs = prepare_and_normalize_obs(self.model, obs)
                    v_value_map_2d = self.get_v_value_map_2d(normalized_obs, width, height)
                    frame = np.concatenate((frame, v_value_map_2d), axis=1)
            return frame
        else:
            return self.env.render()

    def get_v_value_map_2d(self, obs, width=None, height=None):
        tmp_score = []
        idx = []
        idy = []
        rnn_states = None
        init_x, init_y = copy.deepcopy(obs['obs'][0][0]), copy.deepcopy(obs['obs'][0][1])
        for i in range(-10, 11):
            ti_score = []
            for j in range(-10, 11):
                obs['obs'][0][0] = init_x + i * 0.2
                obs['obs'][0][1] = init_y + j * 0.2

                # x = self.model.forward_head(self.curr_obs)
                # x, new_rnn_states = self.model.forward_core(x, rnn_states)
                # result = self.model.forward_tail(x, values_only=True, sample_actions=True)
                result = self.model.forward(obs, rnn_states, values_only=True)

                ti_score.append(result['values'].item())
                idx.append(i * 0.2)
                idy.append(j * 0.2)

            tmp_score.append(ti_score)

        idx, idy, tmp_score = np.array(idx), np.array(idy), np.array(tmp_score)
        v_value_map_2d = plot_v_value_2d(idx, idy, tmp_score, width=width, height=height)

        return v_value_map_2d

    def get_v_value_map_2d_multi(self, obs, num_agents, width, height):
        tmp_score = []
        idx = []
        idy = []
        rnn_states = None

        init_x, init_y = copy.deepcopy(obs['obs'][:, 0]), copy.deepcopy(obs['obs'][:, 1])
        for i in range(-10, 11):
            ti_score = []
            for j in range(-10, 11):
                obs['obs'][:, 0] = init_x + i * 0.2
                obs['obs'][:, 1] = init_y + j * 0.2

                result = self.model.forward(obs, rnn_states, values_only=True)

                ti_score.append(result['values'].cpu().numpy())
                idx.append(i * 0.2)
                idy.append(j * 0.2)

            tmp_score.append(ti_score)

        v_value_maps = []
        idx, idy, tmp_score = np.array(idx), np.array(idy), np.array(tmp_score)
        for i in range(num_agents):
            tmp_single_score = np.array(tmp_score[:, :, i])
            v_value_map_2d = plot_v_value_2d(idx, idy, tmp_single_score, width=width, height=height)
            v_value_maps.append(v_value_map_2d)

        return np.array(v_value_maps)

