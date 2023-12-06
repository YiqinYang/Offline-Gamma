import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

torch.set_num_threads(16)
torch.set_num_interop_threads(16)

plt.rcParams['pdf.use14corefonts'] = True
plt.rc('text', usetex=True)

GAMMA = 0.95


class RandomMdp():
    def __init__(self, random=False):
        self.N_S = 30
        self.N_A = 10
        self.initial_state_dist = np.ones(self.N_S)
        self.initial_state_dist /= self.initial_state_dist.sum()  # uniform reset
        if random:
            # random env
            self.P = np.random.rand(self.N_S, self.N_A, self.N_S)
            self.P /= self.P.sum(-1, keepdims=True)
        else:
            self.P = np.random.rand(self.N_S, self.N_A, self.N_S)
            self.P[self.P == self.P.max(-1, keepdims=True)] = 1
            self.P = np.floor(self.P)
            assert np.all(self.P.sum(-1) == 1), self.P
            # deterministic env
        self.r = np.random.rand(self.N_S, self.N_A)  # r(s, a)
        self.max_steps = 100

    def reset(self):
        self.t = 0
        self.state = np.random.choice(self.N_S, p=self.initial_state_dist)
        return self.state

    def step(self, action):
        self.t += 1
        reward = self.r[self.state, action]
        self.state = np.random.choice(self.N_S, p=self.P[self.state, action])
        return self.state, reward, self.t >= self.max_steps


class RandomPolicy():
    def __init__(self):
        self.N_S = 5
        self.N_A = 3
        self.policy = np.random.rand(self.N_S, self.N_A)
        self.policy /= self.policy.sum(-1, keepdims=True)

    def sample(self, state):
        return np.random.choice(self.N_A, p=self.policy[state])

    def set_policy(self, mu):
        self.policy = mu


def generate_traj(mdp, policy):
    traj = []
    state = mdp.reset()
    done = False
    while not done:
        action = policy.sample(state)
        next_state, reward, done = mdp.step(action)
        traj.append((state, action, reward, next_state))
        state = next_state
    return traj


def get_fixed_point(operator, Q, mdp, *args, eps=1e-6):
    Q_old = - 1 / (1 - GAMMA)  # arbitray init
    count = 0
    while np.max(np.abs(Q_old - Q)) > eps and count < 1000:
        Q_old = Q
        Q = operator(Q, mdp, *args)
        count += 1
    return Q


def optimal_operator(Q, mdp, *args):
    """
    Optimal Bellman Operator: Q(s, a) = r(s, a) + gamma * max_{a'} Q(s', a')
    """
    return mdp.r + GAMMA * np.dot(mdp.P, Q.max(-1))


def eval_operator(Q, mdp, mu, *args):
    """
    Bellman Operator: Q(s, a) = r(s, a) + γ * E_μ Q(s', a')
    """
    return mdp.r + GAMMA * np.dot(mdp.P, np.sum(mu * Q, axis=-1))

def softmax(x, alpha=1):
    y = x - x.max(axis=-1, keepdims=True)
    y = np.exp(y / alpha)
    return y / np.sum(y, axis=-1, keepdims=True)