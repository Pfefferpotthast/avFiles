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
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess_single_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    return normalized.astype(np.float32)


def preprocess_state(state_stack):
    return torch.tensor(state_stack, dtype=torch.float32, device=device)


def reward_memory():
    count = 0
    length = 100
    history = np.zeros(length)

    def memory(reward):
        nonlocal count
        history[count] = reward
        count = (count + 1) % length
        return np.mean(history)

    return memory


def compute_shaped_reward(rgb_obs, base_reward, terminated, av_r_fn=None, enable_early_termination=True):
    reward = base_reward

    if terminated:
        reward += 100.0

    if np.mean(rgb_obs[:, :, 1]) > 185.0:
        reward -= 0.01 

    early_done = False
    if enable_early_termination and av_r_fn is not None:
        early_done = av_r_fn(reward) <= -1.0 

    return reward, early_done


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.last_rgb = None

    def reset(self):
        obs, info = self.env.reset()
        self.last_rgb = obs
        preprocessed = preprocess_single_frame(obs)
        for _ in range(self.k):
            self.frames.append(preprocessed)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_rgb = obs
        preprocessed = preprocess_single_frame(obs)
        self.frames.append(preprocessed)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)

    def get_latest_rgb(self):
        return self.last_rgb


class DiscretizedCarRacing(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = [
            np.array([0.0, 1.0, 0.0]),      # Forward
            np.array([-1.0, 1.0, 0.0]),    # Left + Forward
            np.array([1.0, 1.0, 0.0]),     # Right + Forward
            np.array([0.0, 0.0, 0.8]),     # Brake
            np.array([-0.5, 1.0, 0.0]),    # Slight Left + Forward
            np.array([0.5, 1.0, 0.0]),     # Slight Right + Forward
        ]
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action_idx):
        return self.actions[action_idx]


class PPOPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 64, 64)
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
        self.optimizer = optim.Adam(self.policy.parameters(), lr=kwargs.get('learning_rate', 2.5e-4))  
        self.rollout_buffer = RolloutBuffer()

        self.n_steps = kwargs.get('n_steps', 2048) 
        self.batch_size = kwargs.get('batch_size', 64)
        self.n_epochs = kwargs.get('n_epochs', 4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_range = kwargs.get('clip_range', 0.2)
        self.value_coef = kwargs.get('value_coef', 0.5)
        self.entropy_coef = kwargs.get('entropy_coef', 0.01) 
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)

        self.episode_counter = 0
        self.reward_history = deque(maxlen=100)
        self.prev_avg_reward = 0
        self.entropy_history = deque(maxlen=100)

        self.early_termination_enabled = kwargs.get("early_termination", True)
        self.av_r = reward_memory() if self.early_termination_enabled else None

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

            next_state, base_reward, terminated, truncated, _ = self.env.step(action.item())

            if hasattr(self.env, "get_latest_rgb"):
                rgb_frame = self.env.get_latest_rgb()
            else:
                raise RuntimeError("env must support get_latest_rgb() for reward shaping")

            reward, early_done = compute_shaped_reward(
                rgb_obs=rgb_frame,
                base_reward=base_reward,
                terminated=terminated,
                av_r_fn=self.av_r,
                enable_early_termination=self.early_termination_enabled
            )

            done = terminated or truncated or early_done
            

            reward = np.clip(reward / 50.0, -1.0, 1.0) 

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
        return returns, advantages, avg_episode_reward, episode_count

    def update_policy(self, returns, advantages):
        states, actions, old_log_probs, returns, advantages = self.rollout_buffer.get_tensors(returns, advantages)
        
        std = advantages.std()
        advantages = (advantages - advantages.mean()) / (std + 1e-8)

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        policy_losses, value_losses, entropies = [], [], []

        for epoch in range(self.n_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = batch
                logits, values = self.policy(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                if ratio.max() > 3.0:
                    break
                    
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

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        self.rollout_buffer.clear()
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)

    def learn(self):
        iteration = 0
        best_reward = -float('inf')
        
        while self.episode_counter < 1000:
            returns, advantages, avg_ep_reward, num_eps = self.collect_rollouts()
            p_loss, v_loss, ent = self.update_policy(returns, advantages)

            self.episode_counter += num_eps
            self.reward_history.append(avg_ep_reward)
            avg_reward = np.mean(self.reward_history)
            reward_increase = avg_reward - self.prev_avg_reward
            self.prev_avg_reward = avg_reward

            self.entropy_history.append(ent)
            avg_entropy = np.mean(self.entropy_history)

            iteration += 1

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(self.policy.state_dict(), "best_ppo_model.pth")

            self.writer.add_scalar("loss/policy", p_loss, self.episode_counter)
            self.writer.add_scalar("loss/value", v_loss, self.episode_counter)
            self.writer.add_scalar("loss/entropy", ent, self.episode_counter)
            self.writer.add_scalar("charts/avg_episode_reward", avg_ep_reward, self.episode_counter)
            self.writer.add_scalar("charts/avg_100ep_reward", avg_reward, self.episode_counter)
            self.writer.add_scalar("charts/reward_increase", reward_increase, self.episode_counter)
            self.writer.add_scalar("charts/avg_entropy", avg_entropy, self.episode_counter)

            print(f"Ep {self.episode_counter} | AvgReward (this rollout): {avg_ep_reward:.2f} | Trend: {reward_increase:+.2f}")
            print(f"Losses => Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Entropy: {ent:.4f}")
            print(f"Best Reward: {best_reward:.2f}")
            print("-" * 50)

            self.writer.flush()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("charts/learning_rate", current_lr, self.episode_counter)

            if iteration % 10 == 0:
                torch.save(self.policy.state_dict(), f"ppo_model_iter{iteration}.pth")


if __name__ == "__main__":
    run_name = f"PPO_CarRacing_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    base_env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    discrete_env = DiscretizedCarRacing(base_env)
    env = FrameStackWrapper(discrete_env, k=4)

  
    agent = PPO(
        env, 
        writer, 
        n_steps=2048,          
        batch_size=64, 
        n_epochs=4,             
        learning_rate=2.5e-4,   
        entropy_coef=0.01,     
        early_termination=True
    )
    agent.learn()