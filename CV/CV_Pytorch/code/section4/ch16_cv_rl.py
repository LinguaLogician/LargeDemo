# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: ch16_cv_rl.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/27 10:00
# https://chat.deepseek.com/a/chat/s/f77ff107-6359-49af-95eb-213a043d89a9
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import carla
import gym_carla
from torch_snippets import *

# ============================== Hyperparameters ==============================
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 1e-2
LR = 5e-4
UPDATE_EVERY = 50
ACTION_SIZE = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================== Model Definition ==============================
class DQNetworkImageSensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 32, (8, 8), stride=4),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.Conv2d(64, 128, (3, 3), stride=1),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.Linear(512, 9)
        )
        self.lidar_branch = nn.Sequential(
            nn.Conv2d(3, 32, (8, 8), stride=4),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.Conv2d(64, 128, (3, 3), stride=1),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.Linear(512, 9)
        )
        self.sensor_branch = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 9)
        )

    def forward(self, image, lidar=None, sensor=None):
        x = self.image_branch(image)
        if lidar is None:
            y = 0
        else:
            y = self.lidar_branch(lidar)
        z = self.sensor_branch(sensor)
        return x + y + z


# ============================== Replay Buffer ==============================
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        images = torch.from_numpy(np.vstack([e.state['image'][None] for e in experiences if e is not None])).float().to(
            device)
        lidars = torch.from_numpy(np.vstack([e.state['lidar'][None] for e in experiences if e is not None])).float().to(
            device)
        sensors = torch.from_numpy(np.vstack([e.state['sensor'] for e in experiences if e is not None])).float().to(
            device)
        states = [images, lidars, sensors]
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_images = torch.from_numpy(
            np.vstack([e.next_state['image'][None] for e in experiences if e is not None])).float().to(device)
        next_lidars = torch.from_numpy(
            np.vstack([e.next_state['lidar'][None] for e in experiences if e is not None])).float().to(device)
        next_sensors = torch.from_numpy(
            np.vstack([e.next_state['sensor'] for e in experiences if e is not None])).float().to(device)
        next_states = [next_images, next_lidars, next_sensors]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# ============================== Actor (Agent) ==============================
class Actor():
    def __init__(self):
        self.qnetwork_local = DQNetworkImageSensor().to(device)
        self.qnetwork_target = DQNetworkImageSensor().to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, 10)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        images, lidars, sensors = state['image'], state['lidar'], state['sensor']
        images = torch.from_numpy(images).float().unsqueeze(0).to(device)
        lidars = torch.from_numpy(lidars).float().unsqueeze(0).to(device)
        sensors = torch.from_numpy(sensors).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(images, lidar=lidars, sensor=sensors)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(9))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        images, lidars, sensors = states
        next_images, next_lidars, next_sensors = next_states
        Q_targets_next = self.qnetwork_target(next_images, lidar=next_lidars, sensor=next_sensors).detach().max(1)[
            0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(images, lidar=lidars, sensor=sensors).gather(1, actions.long())
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# ============================== Helper Functions for Different Tasks ==============================
def frozen_lake_q_learning():
    env = gym.make('FrozenLake-v0', is_slippery=False)
    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))

    # First phase: random exploration
    episode_rewards = []
    for i in range(10000):
        state = env.reset()
        total_rewards = 0
        for step in range(50):
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            qtable[state, action] += 0.1 * (reward + 0.9 * np.max(qtable[new_state, :]) - qtable[state, action])
            state = new_state
            total_rewards += reward
        episode_rewards.append(total_rewards)
    print("Q-table after random exploration:\n", qtable)

    # Second phase: epsilon-greedy
    episode_rewards = []
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay_rate = 0.005
    for episode in range(1000):
        state = env.reset()
        total_rewards = 0
        for step in range(50):
            exp_exp_tradeoff = random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state, :])
            else:
                action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            qtable[state, action] += 0.9 * (reward + 0.9 * np.max(qtable[new_state, :]) - qtable[state, action])
            state = new_state
            total_rewards += reward
        episode_rewards.append(total_rewards)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    print("Q-table after epsilon-greedy:\n", qtable)

    # Test the trained agent
    env.reset()
    for episode in range(1):
        state = env.reset()
        step = 0
        done = False
        print("-----------------------")
        print("Episode", episode)
        for step in range(50):
            env.render()
            action = np.argmax(qtable[state, :])
            print(action)
            new_state, reward, done, info = env.step(action)
            if done:
                print("Number of Steps", step + 1)
                break
            state = new_state
    env.close()


def cart_pole_dqn():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    class DQNetwork(nn.Module):
        def __init__(self, state_size, action_size):
            super(DQNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, 24)
            self.fc2 = nn.Linear(24, 24)
            self.fc3 = nn.Linear(24, action_size)

        def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class Agent():
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.seed = random.seed(0)
            self.buffer_size = 2000
            self.batch_size = 64
            self.gamma = 0.99
            self.lr = 0.0025
            self.update_every = 4
            self.local = DQNetwork(state_size, action_size).to(device)
            self.optimizer = optim.Adam(self.local.parameters(), lr=self.lr)
            self.memory = deque(maxlen=self.buffer_size)
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
            self.t_step = 0

        def step(self, state, action, reward, next_state, done):
            self.memory.append(self.experience(state, action, reward, next_state, done))
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                if len(self.memory) > self.batch_size:
                    experiences = self.sample_experiences()
                    self.learn(experiences, self.gamma)

        def act(self, state, eps=0.):
            if random.random() > eps:
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                self.local.eval()
                with torch.no_grad():
                    action_values = self.local(state)
                self.local.train()
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

        def learn(self, experiences, gamma):
            states, actions, rewards, next_states, dones = experiences
            Q_expected = self.local(states).gather(1, actions)
            Q_targets_next = self.local(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            loss = F.mse_loss(Q_expected, Q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def sample_experiences(self):
            experiences = random.sample(self.memory, k=self.batch_size)
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
                device)
            dones = torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
            return (states, actions, rewards, next_states, dones)

    agent = Agent(state_size, action_size)
    scores = []
    scores_window = deque(maxlen=100)
    n_episodes = 5000
    max_t = 5000
    eps_start = 1.0
    eps_end = 0.001
    eps_decay = 0.9995
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        state_size = env.observation_space.shape[0]
        state = np.reshape(state, [1, state_size])
        score = 0
        for i in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -10
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\tReward {} \tAverage Score: {:.2f} \tEpsilon: {}'.format(i_episode, score,
                                                                                     np.mean(scores_window), eps),
              end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} \tEpsilon: {}'.format(i_episode, np.mean(scores_window), eps))
        if i_episode > 10 and np.mean(scores[-10:]) > 450:
            break
    env.close()
    plt.plot(scores)
    plt.title('Scores over increasing episodes')
    plt.show()


def pong_dqn():
    env = gym.make('PongDeterministic-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    def preprocess_frame(frame):
        bkg_color = np.array([144, 72, 17])
        img = np.mean(frame[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
        return img

    def stack_frames(stacked_frames, state, is_new_episode):
        frame = preprocess_frame(state)
        stack_size = 4
        if is_new_episode:
            stacked_frames = deque([np.zeros((80, 80), dtype=np.uint8) for i in range(stack_size)], maxlen=4)
            for i in range(stack_size):
                stacked_frames.append(frame)
            stacked_state = np.stack(stacked_frames, axis=2).transpose(2, 0, 1)
        else:
            stacked_frames.append(frame)
            stacked_state = np.stack(stacked_frames, axis=2).transpose(2, 0, 1)
        return stacked_state, stacked_frames

    class DQNetwork(nn.Module):
        def __init__(self, states, action_size):
            super(DQNetwork, self).__init__()
            self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=4)
            self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=2)
            self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=1)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(2304, 512)
            self.fc2 = nn.Linear(512, action_size)

        def forward(self, state):
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    class Agent():
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.seed = random.seed(0)
            self.buffer_size = 10000
            self.batch_size = 32
            self.gamma = 0.99
            self.lr = 0.0001
            self.update_every = 4
            self.update_every_target = 1000
            self.learn_every_target_counter = 0
            self.local = DQNetwork(state_size, action_size).to(device)
            self.target = DQNetwork(state_size, action_size).to(device)
            self.optimizer = optim.Adam(self.local.parameters(), lr=self.lr)
            self.memory = deque(maxlen=self.buffer_size)
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
            self.t_step = 0

        def step(self, state, action, reward, next_state, done):
            self.memory.append(self.experience(state[None], action, reward, next_state[None], done))
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                if len(self.memory) > self.batch_size:
                    experiences = self.sample_experiences()
                    self.learn(experiences, self.gamma)

        def act(self, state, eps=0.):
            if random.random() > eps:
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                self.local.eval()
                with torch.no_grad():
                    action_values = self.local(state)
                self.local.train()
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

        def learn(self, experiences, gamma):
            self.learn_every_target_counter += 1
            states, actions, rewards, next_states, dones = experiences
            Q_expected = self.local(states).gather(1, actions)
            Q_targets_next = self.target(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            loss = F.mse_loss(Q_expected, Q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.learn_every_target_counter % 1000 == 0:
                self.target_update()

        def target_update(self):
            print('target updating')
            self.target.load_state_dict(self.local.state_dict())

        def sample_experiences(self):
            experiences = random.sample(self.memory, k=self.batch_size)
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
                device)
            dones = torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
            return (states, actions, rewards, next_states, dones)

    agent = Agent(state_size, action_size)
    n_episodes = 5000
    max_t = 5000
    eps_start = 1.0
    eps_end = 0.02
    eps_decay = 0.995
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    stack_size = 4
    stacked_frames = deque([np.zeros((80, 80), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        state, frames = stack_frames(stacked_frames, state, True)
        score = 0
        for i in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state, frames = stack_frames(frames, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\tReward {} \tAverage Score: {:.2f} \tEpsilon: {}'.format(i_episode, score,
                                                                                     np.mean(scores_window), eps),
              end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} \tEpsilon: {}'.format(i_episode, np.mean(scores_window), eps))
    env.close()


def train_self_driving_agent():
    params = {
        'number_of_vehicles': 10,
        'number_of_walkers': 0,
        'display_size': 384,
        'max_past_step': 1,
        'dt': 0.1,
        'discrete': True,
        'discrete_acc': [-3.0, 0, 3],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-0.3, 0.3],
        'ego_vehicle_filter': 'vehicle.lincoln*',
        'port': 2000,
        'town': 'Town03',
        'task_mode': 'random',
        'max_time_episode': 1000,
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 2.0,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'display_route': True,
        'pixor_size': 64,
        'pixor': False,
    }
    env = gym.make('carla-v0', params=params)
    preprocess = lambda im: im.transpose(2, 0, 1) / 255.

    load_path = 'fast-car-v2.pth'
    save_path = 'fast-car-v2.1.pth'

    actor = Actor()
    if load_path is not None:
        actor.qnetwork_local.load_state_dict(torch.load(load_path))
        actor.qnetwork_target.load_state_dict(torch.load(load_path))

    n_episodes = 1000
    log = Report(n_episodes)

    def dqn(n_episodes=n_episodes, max_t=1000, eps_start=0.1, eps_end=0.01, eps_decay=0.995):
        scores = []
        scores_window = deque(maxlen=100)
        eps = eps_start
        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            image, lidar, sensor = state['camera'], state['lidar'], state['state']
            image, lidar = preprocess(image), preprocess(lidar)
            state_dict = {'image': image, 'lidar': lidar, 'sensor': sensor}
            score = 0
            for t in range(max_t):
                action = actor.act(state_dict, eps)
                next_state, reward, done, _ = env.step(action)
                image, lidar, sensor = next_state['camera'], next_state['lidar'], next_state['state']
                image, lidar = preprocess(image), preprocess(lidar)
                next_state_dict = {'image': image, 'lidar': lidar, 'sensor': sensor}
                actor.step(state_dict, action, reward, next_state_dict, done)
                state_dict = next_state_dict
                score += reward
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay * eps)
            log.record(i_episode, score=score, end='\r')
            if i_episode % 100 == 0:
                log.record(i_episode, mean_score=np.mean(scores_window))
                torch.save(actor.qnetwork_local.state_dict(), save_path)
        return scores

    dqn()
    env.close()


def understand_gym_environment():
    from gym import envs
    print(envs.registry.all())
    env = gym.make('FrozenLake-v0', is_slippery=False)
    env.render()
    print("Observation space:", env.observation_space.n)
    print("Action space:", env.action_space.n)
    print("Sample action:", env.action_space.sample())
    state = env.reset()
    print("Initial state:", state)
    next_state, reward, done, info = env.step(env.action_space.sample())
    print("Next state:", next_state, "Reward:", reward, "Done:", done, "Info:", info)
    env.close()


# ============================== Main Execution ==============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run specific RL tasks')
    parser.add_argument('--task', type=str, choices=['frozen_lake', 'cart_pole', 'pong', 'self_driving', 'gym_env'],
                        help='Task to run')
    args = parser.parse_args()

    if args.task == 'frozen_lake':
        frozen_lake_q_learning()
    elif args.task == 'cart_pole':
        cart_pole_dqn()
    elif args.task == 'pong':
        pong_dqn()
    elif args.task == 'self_driving':
        train_self_driving_agent()
    elif args.task == 'gym_env':
        understand_gym_environment()
    else:
        print("Please specify a task with --task [frozen_lake|cart_pole|pong|self_driving|gym_env]")
