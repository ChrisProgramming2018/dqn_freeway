import os
import sys 
import cv2
import gym
import time
import torch 
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime import datetime
from replay_buffer import ReplayBuffer
from utils import time_format

def train_agent(env, config):
    """
    Args:
    """
    
    # create CNN convert the [1,3,84,84] to [1, 200]
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    #pathname = str(args.locexp) + "/" + str(args.env_name) + '_agent_' + str(args.policy)
    #pathname += "_batch_size_" + str(args.batch_size) + "_lr_act_" + str(args.lr_actor) 
    #pathname += "_lr_critc_" + str(args.lr_critic) + "_lr_decoder_"
    pathname = "" 
    tensorboard_name = str(config["locexp"]) + '/runs/' + pathname 
    writer = SummaryWriter(tensorboard_name)
    eps = config["eps_start"]
    eps_end = config["eps_end"]
    eps_decay = config["eps_decay"]
    scores_window = deque(maxlen=100)
    scores = [] 
    t0 = time.time()
    for i_episode in range(config["train_episodes"]):
        state = env.reset()
        score = 0
        for t in range(config["max_t"]):
            #action = agent.act(state, eps)
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent scor
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} time: {}'.format(i_episode, np.mean(scores_window), time_format(time.time() - t0)), end="")
