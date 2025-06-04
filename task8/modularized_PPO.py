import os
import time
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque  # --- MODIFIED ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DiscretizedCarRacing(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = [
            np.array([0.0, 1.0, 0.0]),
            np.array([-1.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.8]),
            np.array([-0.5, 1.0, 0.0]),
            np.array([0.5, 1.0, 0.0]),
        ]
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action_idx):
        return self.actions[action_idx]

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        obs, info = self.env.reset()
        preprocessed = preprocess_single_frame(obs)
        for _ in range(self.k):
            self.frames.append(preprocessed)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        preprocessed = preprocess_single_frame(obs)
        self.frames.append(preprocessed)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)  # shape: (k, 64, 64)

def preprocess_single_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)     # Convert to grayscale
    resized = cv2.resize(gray, (64, 64))                # Resize to 64x64
    normalized = resized / 255.0                        # Normalize pixel values to [0, 1]
    return normalized.astype(np.float32)                # Ensure float32 type for PyTorch

def preprocess_state(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (64, 64))
    state = np.expand_dims(state, axis=0)
    state = state / 255.0
    return torch.tensor(state, dtype=torch.float32, device=device)

class PPOPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            dummy_out = self.shared_cnn(dummy)
            self.flattened_size = dummy_out.view(1, -1).shape[1]

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.shared_cnn(x)
        x = self.shared_fc(x)
        return self.actor(x), self.critic(x)

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.values, self.log_probs = [], [], []

    def add(self, state, action, reward, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def get_tensors(self, returns, advantages):
        return (
            torch.stack(self.states),
            torch.tensor(self.actions, dtype=torch.long, device=device),
            torch.stack(self.log_probs),
            returns,
            advantages,
        )

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        values = torch.stack(self.values)
        values = torch.cat([values, torch.tensor([last_value], device=device)])

        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])

        returns = torch.stack(returns)
        advantages = returns - values[:-1]
        return returns, advantages

class PPO:
    def __init__(self, env, writer, **kwargs):
        self.env = env
        self.writer = writer
        self.policy = PPOPolicy(action_dim=env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=kwargs.get('learning_rate', 3e-4))
        self.rollout_buffer = RolloutBuffer()

        self.n_steps = kwargs.get('n_steps', 4096)
        self.batch_size = kwargs.get('batch_size', 64)
        self.n_epochs = kwargs.get('n_epochs', 10)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_range = kwargs.get('clip_range', 0.2)
        self.value_coef = kwargs.get('value_coef', 0.5)
        self.entropy_coef = kwargs.get('entropy_coef', 0.02)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)

        self.episode_counter = 0
        self.reward_history = deque(maxlen=100)
        self.prev_avg_reward = 0
        self.entropy_history = deque(maxlen=100)  # ⬅️ This line is essential

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: max(0.1, 1 - step / 1000.0)
        )


        self.episode_counter = 0  # --- MODIFIED ---
        self.reward_history = deque(maxlen=100)  # --- MODIFIED ---
        self.prev_avg_reward = 0  # --- MODIFIED ---
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: max(0.1, 1 - step / 1000.0)  # decay to 10% over time
        )


    def collect_rollouts(self):
        steps = 0
        state, _ = self.env.reset()
        done = False
        episode_rewards = []
        episode_reward = 0
        episode_count = 0

        while steps < self.n_steps:
            state_tensor = preprocess_state(state).unsqueeze(0)
            with torch.no_grad():
                logits, value = self.policy(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            reward = np.clip(reward, -1, 1)

            self.rollout_buffer.add(state_tensor.squeeze(0), action.item(), reward, done, value.squeeze(), log_prob)

            state = next_state
            steps += 1
            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                episode_count += 1
                state, _ = self.env.reset()

        with torch.no_grad():
            state_tensor = preprocess_state(state).unsqueeze(0)
            _, last_value = self.policy(state_tensor)
            last_value = last_value.item() if not done else 0

        returns, advantages = self.rollout_buffer.compute_returns_and_advantages(last_value)
        avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0
        return returns, advantages, avg_episode_reward, episode_count  # --- MODIFIED ---

    def update_policy(self, returns, advantages):
        states, actions, old_log_probs, returns, advantages = self.rollout_buffer.get_tensors(returns, advantages)
        std = advantages.std()
        if std < 1e-8:
            std = 1e-8
        advantages = (advantages - advantages.mean()) / std

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.n_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = batch
                logits, values = self.policy(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (batch_returns - values.squeeze()).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.rollout_buffer.clear()
        return policy_loss.item(), value_loss.item(), entropy.item()

    def learn(self):
        iteration = 0
        while self.episode_counter < 1000:  # --- MODIFIED ---
            returns, advantages, avg_ep_reward, num_eps = self.collect_rollouts()
            p_loss, v_loss, ent = self.update_policy(returns, advantages)

            self.episode_counter += num_eps
            self.reward_history.append(avg_ep_reward)
            avg_reward = np.mean(self.reward_history)
            reward_increase = avg_reward - self.prev_avg_reward
            self.prev_avg_reward = avg_reward

            iteration += 1

            self.writer.add_scalar("loss/policy", p_loss, self.episode_counter)
            self.writer.add_scalar("loss/value", v_loss, self.episode_counter)
            self.writer.add_scalar("loss/entropy", ent, self.episode_counter)
            self.writer.add_scalar("charts/avg_episode_reward", avg_ep_reward, self.episode_counter)
            self.writer.add_scalar("charts/avg_100ep_reward", avg_reward, self.episode_counter)
            self.writer.add_scalar("charts/reward_increase", reward_increase, self.episode_counter)

            # New: log entropy moving average
            self.entropy_history.append(ent)
            avg_entropy = np.mean(self.entropy_history)
            self.writer.add_scalar("charts/avg_entropy", avg_entropy, self.episode_counter)


            print(f"Ep {self.episode_counter} | AvgReward (this rollout): {avg_ep_reward:.2f} | Trend: {reward_increase:+.2f}")
            print(f"Losses => Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Entropy: {ent:.4f}")
            print("-" * 50)

            self.writer.flush()

            # Learning rate schedule step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("charts/learning_rate", current_lr, self.episode_counter)

            if iteration % 10 == 0:
                torch.save(self.policy.state_dict(), f"ppo_model_iter{iteration}.pth")


# --- Main ---
if __name__ == "__main__":
    run_name = f"PPO_CarRacing_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    env = DiscretizedCarRacing(env)

    agent = PPO(env, writer, n_steps=4096, batch_size=64, n_epochs=10)
    agent.learn()
