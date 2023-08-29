import copy

import gymnasium as gym
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
                    v_value_maps = self.get_v_value_map_2d_multi(num_agents, width, height)
                    frame = np.concatenate((frame, v_value_maps), axis=2)
                    frame = frame.reshape((2, num_agents // 2,) + frame.shape[1:])
                    frame = np.transpose(frame, (0, 2, 1, 3, 4))
                    frame = frame.reshape(2 * frame.shape[1], -1, frame.shape[4])
                    # frame = np.concatenate((frame, v_value_maps), axis=1)
                    # frame = np.transpose(frame, (1, 0, 2, 3))
                    # frame = frame.reshape(frame.shape[0], -1, frame.shape[3])

                else:
                    width, height = frame.shape[0], frame.shape[1]
                    v_value_map_2d = self.get_v_value_map_2d(width, height)
                    frame = np.concatenate((frame, v_value_map_2d), axis=1)
            return frame
        else:
            return self.env.render()

    def get_obs(self):
        obs = []
        for i in range(self.env.num_agents):
            obs.append(self.env.envs[i].state_vector(self.env.envs[i]))
        if self.env.num_use_neighbor_obs > 0:
            obs = self.env.add_neighborhood_obs(obs)
        for i in range(self.env.num_agents):
            self.pos[i, :] = self.env.envs[i].dynamics.pos
        if self.env.use_obstacles:
            obs = self.env.obstacles.step(obs=obs, quads_pos=self.pos)
        return obs

    def get_v_value_map_2d(self, width=None, height=None):
        tmp_score = []
        idx = []
        idy = []
        rnn_states = None
        init_x, init_y = self.env.envs[0].dynamics.pos[0], self.env.envs[0].dynamics.pos[1]
        for i in range(-10, 11):
            ti_score = []
            for j in range(-10, 11):
                self.env.envs[0].dynamics.pos[0] = init_x + i * 0.2
                self.env.envs[0].dynamics.pos[1] = init_y + j * 0.2

                # x = self.model.forward_head(self.curr_obs)
                # x, new_rnn_states = self.model.forward_core(x, rnn_states)
                # result = self.model.forward_tail(x, values_only=True, sample_actions=True)
                curr_obs = self.get_obs()
                obs = dict(obs=np.array(curr_obs))
                normalized_obs = prepare_and_normalize_obs(self.model, obs)
                result = self.model.forward(normalized_obs, rnn_states, values_only=True)

                ti_score.append(result['values'].item())
                idx.append(i * 0.2)
                idy.append(j * 0.2)

            tmp_score.append(ti_score)

        self.env.envs[0].dynamics.pos[0] = init_x
        self.env.envs[0].dynamics.pos[1] = init_y
        idx, idy, tmp_score = np.array(idx), np.array(idy), np.array(tmp_score)
        v_value_map_2d = plot_v_value_2d(idx, idy, tmp_score, width=width, height=height)

        return v_value_map_2d

    def get_v_value_map_2d_multi(self, num_agents, width, height):
        tmp_score = []
        idx = []
        idy = []
        temp_maps = []
        rnn_states = None
        for drone in range(num_agents):
            init_x, init_y = self.env.envs[drone].dynamics.pos[0], self.env.envs[drone].dynamics.pos[1]
            for i in range(-10, 11):
                ti_score = []
                for j in range(-10, 11):
                    self.env.envs[drone].dynamics.pos[0] = init_x + i * 0.2
                    self.env.envs[drone].dynamics.pos[1] = init_y + j * 0.2

                    # x = self.model.forward_head(self.curr_obs)
                    # x, new_rnn_states = self.model.forward_core(x, rnn_states)
                    # result = self.model.forward_tail(x, values_only=True, sample_actions=True)
                    curr_obs = self.get_obs()
                    obs = dict(obs=np.array(curr_obs))
                    normalized_obs = prepare_and_normalize_obs(self.model, obs)
                    result = self.model.forward(normalized_obs, rnn_states, values_only=True)

                    ti_score.append(result['values'].cpu().numpy()[drone])
                    idx.append(i * 0.2)
                    idy.append(j * 0.2)

                tmp_score.append(ti_score)
            self.env.envs[drone].dynamics.pos[0] = init_x
            self.env.envs[drone].dynamics.pos[1] = init_y
            temp_maps.append(tmp_score)
            if drone < num_agents-1:
                idx = []
                idy = []
            ti_score = []
            tmp_score = []

        v_value_maps = []
        idx, idy = np.array(idx), np.array(idy)
        for drone in range(num_agents):
            v_value_map_2d = plot_v_value_2d(idx, idy, np.array(temp_maps[drone]), width=width, height=height)
            v_value_maps.append(v_value_map_2d)

        return np.array(v_value_maps)

