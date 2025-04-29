# rl.py

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """
    Simple feed-forward network for Q-value approximation.
    """
    def __init__(self, state_size, action_size, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        input_dim = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    DQN Agent with both one-step updates and optional replay buffer.
    """
    def __init__(
        self,
        state_size,
        action_size,
        memory_size=2000,
        batch_size=64,
        gamma=0.95,
        lr=1e-3,
        eps_start=1.0,
        eps_min=0.01,
        eps_decay=0.995,
        device=None
    ):
        self.state_size = state_size
        self.action_size = action_size

        # optional replay memory (not used if you do purely one-step updates)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        """
        Epsilon-greedy action selection with torch.no_grad for speed.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_v = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            qvals = self.model(state_v).cpu().numpy()
        return int(np.argmax(qvals))

    def learn(self, state, action, reward, next_state, done):
        """
        Perform one gradient update from a single (s, a, r, s', done) tuple.
        """
        # prepare tensors
        s_v      = torch.FloatTensor(state).to(self.device).unsqueeze(0)   # [1 x state_dim]
        nxt_v    = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
        r_v      = torch.FloatTensor([reward]).to(self.device)            # [1]
        a_v      = torch.LongTensor([action]).to(self.device).unsqueeze(1) # [1 x 1]
        done_v   = torch.FloatTensor([done]).to(self.device)              # [1]

        # Q(s,a)
        q_val    = self.model(s_v).gather(1, a_v).squeeze(1)               # [1]
        # target = r + γ * max_a' Q(s',a') * (1 - done)
        next_max = self.model(nxt_v).max(1)[0]                            # [1]
        target   = r_v + self.gamma * next_max * (1 - done_v)

        loss = self.loss_fn(q_val, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay ε
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

    def memorize(self, state, action, reward, next_state, done):
        """
        Store transition for optional replay.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Sample a batch and learn from it (classic DQN update).
        """
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_v  = torch.FloatTensor(np.vstack(states)).to(self.device)
        next_v    = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        rewards_v = torch.FloatTensor(rewards).to(self.device)
        actions_v = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        dones_v   = torch.FloatTensor(dones).to(self.device)

        q_vals    = self.model(states_v).gather(1, actions_v).squeeze(1)
        next_max  = self.model(next_v).max(1)[0]
        expected  = rewards_v + self.gamma * next_max * (1 - dones_v)

        loss = self.loss_fn(q_vals, expected.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

    def save(self, path):
        torch.save({
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon':         self.epsilon,
        }, path)

    def load(self, path):
        chk = torch.load(path, map_location=self.device)
        self.model.load_state_dict(chk['model_state'])
        self.optimizer.load_state_dict(chk['optimizer_state'])
        self.epsilon = chk.get('epsilon', self.epsilon)


def rl(env, RLpoints, agent=None):
    """
    Trains (or continues training) a DQN agent using one-step updates.

    Args:
      env       : OpenAI Gym env (must have .reset() and .step()).
      RLpoints  : int, number of episodes to train for.
      agent     : optional DQNAgent; if None, a fresh one is created.

    Returns:
      DQNAgent  : your trained (or further trained) agent.
    """
    # figure out state & action dimensions
    sample = env.reset()
    if isinstance(sample, tuple):
        sample = sample[0]
    state_size  = np.prod(sample.shape)
    action_size = env.action_space.n

    if agent is None:
        agent = DQNAgent(state_size, action_size)

    for ep in range(1, RLpoints + 1):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        state = obs.reshape(-1)

        done = False
        while not done:
            action = agent.act(state)
            nxt, reward, done, _ = env.step(action)
            if isinstance(nxt, tuple):
                nxt = nxt[0]
            nxt_state = nxt.reshape(-1)

            agent.learn(state, action, reward, nxt_state, done)
            state = nxt_state

    return agent


def predict(agent, state):
    """
    Quick inference: pick best action for a single state.
    """
    state_v = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        qvals = agent.model(state_v).cpu().numpy()
    return int(np.argmax(qvals))
