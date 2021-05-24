import torch
import pybullet as p
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch as T
from environment import Env
from ddpg_agent import Agent
from networks import ActorNetwork
import math
import time

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

physicsClientId = p.connect(p.GUI)  # connect to bullet  
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) 
# physicsClientId = p.connect(p.DIRECT)  # to launch without GUI
# p.setRealTimeSimulation(1)

env = Env(maxIterations = 250, timeStep = 0.02)

agent = Agent(alpha=1e-4,
              beta=1e-3,
              tau=1e-3,
              gamma=0.99,
              input_dims=(10,),
              n_actions=3,
              max_size=int(1e5),
              batch_size=128,
              fc1_dims=400,
              fc2_dims=300,
              action_limit=1)

# load optimized policy
agent.load_models()
agent.actor.to(device)
agent.actor.eval()

episodes = 10
for e in range(episodes):
    state = env.reset()  # get initial episode observations
    done = False  # reset episode flag
    rewardPerEpisode = 0  # reset the accumulative reward
    counter = 0
    # for i in range(50):
    while not done:
        counter +=1
        p.stepSimulation()
        # time.sleep(0.1)
        state = T.tensor([state], dtype=T.float).to(agent.actor.device)  # get states/observations
        # print(state)
        action = agent.actor.forward(state).to(agent.actor.device)  # predict action
        action = action.cpu().detach().numpy()[0]  
        state, reward, done, distance_trav = env.step(action)  # execute action
        rewardPerEpisode += reward
        if done:
            break
    print('episode:', e, 'reward:', f'{rewardPerEpisode:.2f}','iteration:', f'{counter:.0f}')
env.close()
