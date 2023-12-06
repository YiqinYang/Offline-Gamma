from os import access
import numpy as np
import torch
import d4rl
import gym


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(3e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.gamma = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device(device if torch.cuda.is_available() else "cpu")
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# torch.cuda.set_device(5)


	def add(self, state, action, next_state, reward, done, gamma):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.gamma[self.ptr] = gamma

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.gamma[ind]).to(self.device)
		)

	def convert_D4RL(self, dataset, env_name, datanumber, discount):
		env = gym.make(env_name)
		dataset = d4rl.sequence_dataset(env)
		j = 0
		for seq in dataset:
			j += 1
			observations, actions, dones, rewards, truly_dones = seq['observations'], seq['actions'], seq[
					'timeouts'], seq['rewards'], seq['terminals']
			next_observations = seq['next_observations']
			length = len(observations)
			for i in range(length-1):
				self.add(observations[i], actions[i], next_observations[i], rewards[i], truly_dones[i], discount)    
			if j == datanumber:
				break
		print(env_name, ' load: ', j)

		if 'walker2d' in env_name:
			env_name_2 = 'walker2d-random-v2'
		elif 'hopper' in env_name:
			env_name_2 = 'hopper-random-v2'
		elif 'halfcheetah' in env_name:
			env_name_2 = 'halfcheetah-random-v2'	
		env = gym.make(env_name_2)
		dataset = d4rl.sequence_dataset(env)
		k = 0
		for seq in dataset:
			k += 1
			observations, actions, dones, rewards, truly_dones = seq['observations'], seq['actions'], seq[
					'timeouts'], seq['rewards'], seq['terminals']
			next_observations = seq['next_observations']
			length = len(observations)
			for i in range(length-1):
				self.add(observations[i], actions[i], next_observations[i], rewards[i], truly_dones[i], discount)    
			if k == datanumber:
				break
		print(env_name_2, ' load: ', k)

	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std