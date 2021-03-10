#!/usr/bin/env python2
import gym
import gym_psketch

for env_name in gym_psketch.env_list:
    env = gym.make(env_name)
    env.reset()
