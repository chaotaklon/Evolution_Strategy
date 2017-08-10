import sys
import os
import copy
import random

import gym
import numpy as np

OUTPUT_DIR = '/home/alanmain/Reinforcement_Learning/out/ES_human/'
RENDER = True
RECORD = True
HIDEEN_SIZE = 600
POPULATION_SIZE = 100
NET_INIT_SD = 0.1
NET_TRIAL_SD = 0.01

np.random.seed(1234)

env = gym.make('Humanoid-v1')
if RECORD:
	env = gym.wrappers.Monitor(env, OUTPUT_DIR, force=True)
	
ACTION_SCALE = env.action_space.high[0]
if ACTION_SCALE != -env.action_space.low[0]:
	print "action space is asym"
	sys.exit(0)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

W1 = np.random.normal(0, NET_INIT_SD, [obs_dim, HIDEEN_SIZE])
b1 = np.random.normal(0, NET_INIT_SD, HIDEEN_SIZE)
W2 = np.random.normal(0, NET_INIT_SD, [HIDEEN_SIZE, action_dim])
b2 = np.random.normal(0, NET_INIT_SD, action_dim)

generation = -1
while True:
	generation += 1
	W1_trial = np.random.normal(0, NET_TRIAL_SD, [POPULATION_SIZE, obs_dim, HIDEEN_SIZE])
	b1_trial = np.random.normal(0, NET_TRIAL_SD, [POPULATION_SIZE, HIDEEN_SIZE])
	W2_trial = np.random.normal(0, NET_TRIAL_SD, [POPULATION_SIZE, HIDEEN_SIZE, action_dim])
	b2_trial = np.random.normal(0, NET_TRIAL_SD, [POPULATION_SIZE, action_dim])
	reward_array = np.zeros(POPULATION_SIZE, np.float32)
	for p in range(POPULATION_SIZE):
		observation = env.reset()
		for t in range(env.spec.timestep_limit):
			if RENDER:
				env.render()
				
			h1 = np.tanh( np.matmul(observation, W1+W1_trial[p]) + (b1+b1_trial[p]) )
			action = ACTION_SCALE * np.tanh( np.matmul(h1, W2+W2_trial[p]) + (b2+b2_trial[p]) )
			
			observation, reward, done, info = env.step(action)
			if done and t < env.spec.timestep_limit - 1:
				reward = -1
			elif done and t == env.spec.timestep_limit - 1:
				reward = 1
				
			reward_array[p] += reward
			
			if done:
				print "generation: ", generation, " population: ", p, " finished at: ", t, " timestep"
				break
				
	print "generation: ", generation, " max total reward: ", np.amax(reward_array)
	reward_array /= np.sum(np.absolute(reward_array))
	for p in range(POPULATION_SIZE):
		W1 += W1_trial[p] * reward_array[p]
		b1 += b1_trial[p] * reward_array[p]
		W2 += W2_trial[p] * reward_array[p]
		b2 += b2_trial[p] * reward_array[p]
		
	if generation % 1000 == 0 and generation != 0:
		np.save(OUTPUT_DIR + "W1.npy", W1)
		np.save(OUTPUT_DIR + "b1.npy", b1)
		np.save(OUTPUT_DIR + "W2.npy", W2)
		np.save(OUTPUT_DIR + "b2.npy", b2)
		print "generation: ", generation, " save weight"

