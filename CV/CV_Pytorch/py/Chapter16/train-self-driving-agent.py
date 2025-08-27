#!/usr/bin/env python
# coding: utf-8

# In[ ]:


%load_ext autoreload
%autoreload 2
from model import DQNetworkImageSensor
from actor import Actor
from torch_snippets import *
from collections import deque

# In[ ]:


import gym
import gym_carla
import carla
params = {
    'number_of_vehicles': 10,
    'number_of_walkers': 0,
    'display_size': 384,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0, 3],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
}

# Set gym-carla environment
env = gym.make('carla-v0', params=params)
preprocess = lambda im: im.transpose(2,0,1) / 255. # torch.Tensor(im).permute(1,2,0) / 255.


# In[ ]:


load_path = 'fast-car-v2.pth'
save_path = 'fast-car-v2.1.pth'

actor = Actor()
if load_path is not None:
    actor.qnetwork_local.load_state_dict(torch.load(load_path))
    actor.qnetwork_target.load_state_dict(torch.load(load_path))
else:
    pass

n_episodes = 1000
log = Report(n_episodes)
def dqn(n_episodes=n_episodes, max_t=1000, eps_start=0.1, eps_end=0.01, eps_decay=0.995):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    # initialsize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        image, lidar, sensor = state['camera'], state['lidar'], state['state']
        image, lidar = preprocess(image), preprocess(lidar)
        state_dict = {'image': image, 'lidar': lidar, 'sensor': sensor}
        score = 0
        for t in range(max_t):
            action = actor.act(state_dict, eps)
            # _action = torch.argmax(action[0].cpu().detach())
            next_state, reward, done, _  = env.step(action)
            image, lidar, sensor = next_state['camera'], next_state['lidar'], next_state['state']
            image, lidar = preprocess(image), preprocess(lidar)
            next_state_dict = {'image': image, 'lidar': lidar, 'sensor': sensor}
            actor.step(state_dict, action, reward, next_state_dict, done)
            state_dict = next_state_dict
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        log.record(i_episode, score=score, end='\r')
        if i_episode % 100 == 0:
            log.record(i_episode, mean_score=np.mean(scores_window))
            torch.save(actor.qnetwork_local.state_dict(), save_path)
    return scores

dqn()

# In[ ]:



