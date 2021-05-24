"""
Script to perform training 

Train agent untill Agent's policy is optimized
"""
import time
import pybullet as p
import math
import random
from scipy.spatial import distance
import numpy as np
from ddpg_agent import Agent
import matplotlib.pyplot as plt
from environment import Env
import torch as T
# To send message to phone to inform that training has been completed
# ++++++++++++++++++++++++++++
from twilio.rest import Client
account_sid = 'AC7dec2d89fc51a0c03724d46da863973e'
auth_token = '2fa2e929001ec795a400b30f71834cde'
client = Client(account_sid, auth_token)
# ++++++++++++++++++++++++++++

# set type of interface with PyBullet physic engine
physicsClientId = p.connect(p.GUI)  # connect to bullet
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# physicsClientId = p.connect(p.DIRECT)  # to launch without GUI
# p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
# p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
# p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    env = Env(maxIterations=250, timeStep=0.02)

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
                  action_limit=1)  # define max and min limit for action output

    # load optimized policy
    agent.load_models()
    agent.actor.to(device)
    agent.critic.to(device)
    agent.actor.eval()
    agent.critic.eval()

    best_return = -1000000
    n_episodes = 200  # number of episodes for training
    rewardPerEpisode_history = []
    start = time.time()  # time training

    for i in range(n_episodes):

        observation = env.reset()  # get initial episode observations
        rewardPerEpisode = 0  # reset the accumulative reward
        done = False  # reset episode flag
        counter = 0
        while not done:
            counter += 1
            # Select action accordin to current policy and exploration noise
            action = agent.choose_action(observation)

            # Execute action
            observation_, reward, done, distance_trav = env.step(action)
            # Store transition
            agent.remember(observation, action, reward, observation_, done)
            # Sample a random minibatch and update networks
            agent.learn()

            observation = observation_

            rewardPerEpisode += reward
            print('\rEpisode: {}\tstep reward:{:.2f}'.format(
                i+1, reward), end="")
            if done:
                break

        rewardPerEpisode_history.append(rewardPerEpisode)
        avg_reward = np.mean(rewardPerEpisode_history[-10:])

        if avg_reward > best_return:
            best_return = avg_reward
            agent.save_models()

        print('\r', end="")
        print('\rEpisode:', i+1, '\treward:%.2f' % rewardPerEpisode,
              '\tavg reward:%.2f' % avg_reward,'\tdist_trav:%.2f' % distance_trav,"cm\t",'Iteration:%.2f' % counter, end='')

    env.close()
    end = time.time()
    time_taken = round((end - start)/60, 2)
    print(
        f"++++++++++ Training time for {n_episodes} episodes is: {time_taken} minutes ++++++++++")
    # sending message to phone
    message = client.messages.create(
        body='Training has completed in:' + str(time_taken) + 'minutes with average reward of: ' +
        str(round(avg_reward, 2)),
        from_='+12185161141', to='+4407411914423')
    x = [i+1 for i in range(n_episodes)]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, rewardPerEpisode_history)
    plt.ylabel('Reward')
    plt.xlabel('Episode #')
    plt.show()
