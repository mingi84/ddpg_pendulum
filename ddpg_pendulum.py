import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# --------------------
# 하이퍼파라미터 설정
# --------------------
ENV_NAME = 'Pendulum-v1'
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
BUFFER_SIZE = 100000
BATCH_SIZE = 64
MAX_EPISODES = 200
MAX_STEPS = 200
EXPLORATION_NOISE = 0.1
SEED = 0

env = gym.make(ENV_NAME)  # 학습용 환경 (렌더링 X)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.max_action * x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.memory = ReplayBuffer()

    def select_action(self, state, noise=True):
        state_tensor = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state_tensor).detach().cpu().numpy().flatten()
        if noise:
            action = action + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
        return np.clip(action, -max_action, max_action)

    def train(self):
        if self.memory.size() < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * GAMMA * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


agent = DDPGAgent()

rewards_list = []

for episode in range(MAX_EPISODES):
    (state, info) = env.reset(seed=SEED)
    episode_reward = 0
    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        terminated = done or truncated

        agent.memory.add(state, action, reward, next_state, float(terminated))

        state = next_state
        episode_reward += reward

        agent.train()

        if terminated:
            break

    rewards_list.append(episode_reward)
    print(f"Episode {episode}, Reward: {episode_reward:.2f}")

env.close()

# --------------------------
# 학습 완료 후 시각화 단계
# --------------------------
# 이제 render_mode='human'으로 별도의 환경을 만들어 학습된 정책 시각화
test_env = gym.make(ENV_NAME, render_mode='human')
test_env.reset(seed=SEED)

for i in range(5):  # 5 에피소드 정도 시연
    (state, info) = test_env.reset()
    for step in range(200):
        test_env.render()
        # 테스트 시에는 noise 없이 결정적 정책으로 행동
        action = agent.select_action(state, noise=False)
        next_state, reward, done, truncated, info = test_env.step(action)
        state = next_state
        if done or truncated:
            break

test_env.close()
