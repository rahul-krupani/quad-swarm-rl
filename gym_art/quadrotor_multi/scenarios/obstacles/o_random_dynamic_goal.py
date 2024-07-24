
import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base
from gym_art.quadrotor_multi.quadrotor_traj_gen import QuadTrajGen
from gym_art.quadrotor_multi.quadrotor_planner import traj_eval


class Scenario_o_random_dynamic_goal(Scenario_o_base):
    """ This scenario implements a 13 dim goal that tracks a smooth polynomial trajectory. 
        Each goal point is evaluated through the polynomial generated per reset."""
        
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approch_goal_metric = 0.5

        self.goal_generator = [QuadTrajGen(poly_degree=7) for i in range(num_agents)]
        self.start_point = [np.zeros(3) for i in range(num_agents)]
        self.end_point = [np.zeros(3) for i in range(num_agents)]

        
    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        self.update_formation_and_relate_param()

        tick = self.envs[0].tick
        
        time = self.envs[0].sim_steps*tick*(self.envs[0].dt) #  Current time in seconds.
        
        for i in range(self.num_agents):
            next_goal = self.goal_generator[i].piecewise_eval(time)
    
            self.end_point[i] = next_goal.as_nparray()
            # print(time, self.end_point[i])
            self.goals = copy.deepcopy(self.end_point)
            
        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]
            
        return

    def reset(self, obst_map, cell_centers): 
  
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        for i in range(self.num_agents):
            self.start_point[i] = self.generate_pos_obst_map()
            
            initial_state = traj_eval()
            initial_state.set_initial_pos(self.start_point[i])
            
            final_goal = self.generate_pos_obst_map()
            
            # Fix the goal height at 0.65 m
            final_goal[2] = 0.65
            
            dist = np.linalg.norm(self.start_point[i] - final_goal)

            traj_duration = np.random.uniform(low=dist / 1.5, high=self.envs[0].ep_time-1)

            goal_yaw = np.random.uniform(low=0, high=3.14/2)

            # Generate trajectory with random time from (2, ep_time)
            self.goal_generator[i].plan_go_to_from(initial_state=initial_state, desired_state=np.append(final_goal, goal_yaw), 
                                                   duration=traj_duration, current_time=0)
            self.end_point[i] = self.goal_generator[i].piecewise_eval(0).as_nparray()

        self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        
        self.goals = copy.deepcopy(self.end_point)
        
        
        