# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 01:27:16 2022

@author: hibad
"""

import sys
import numpy as np
sys.path.append('envs')
from tetris import TetrisEnv
import matplotlib.pyplot as plt 
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle 
from player import Net, Player

net=Net(220).to(device)
# net=pickle.load( open( "Q_net_checkpoint.p", "rb" ) )
# with open( "player_memory.p", "rb" ) as file:
#     dat=pickle.load(file)
#     lt_memory=dat["lt_memory"]
#     st_memory=dat["lt_memory"]
max_turn=[]
losses=[]
cleared_row=[]
#episode=[]
#%%
gamma=0.9

player=Player(gamma, net=net)
# player.lt_memory=lt_memory
# player.st_memory=st_memory
#%%
render=False
for i in range(10000):
    print("epsiode:",len(max_turn))
    greedy=0 #np.random.rand()
    player.greedy=greedy
    print("greedy", greedy)
    env = TetrisEnv()
    state=env.reset()
    done=False
    batch=[]
    while not done:
        actions = env.get_actions()
        action = player.policy( state, actions)
        state_prev = state.copy()
        state, reward, done, _ = env.step(action)
        actions_prime = env.get_actions()
        batch.append((state_prev,action,reward,state,actions_prime))
        if done:
            loss=player.update(batch.copy())
            if loss:
                losses.append(loss)
                print("loess", loss)

            break
            
        if render:
            env.render()
            
    max_turn.append(state.turn)
    cleared_row.append(state.cleared)
 #   episode.append(episode[-1]+1)

    print("turn", state.turn)
    print("cleared", state.cleared)
    if not i%10:
        plt.figure()
        render=True
        plt.plot(range(len(max_turn)),max_turn, ".", color="blue", alpha=0.1)
        average=[np.mean(max_turn[j:j+100]) for j in range(len(max_turn)-100)]
        plt.plot(np.array(range(len(average)))+50, average, "--", color="black")
        plt.title("Survived Turn vs Episode")
        plt.xlabel("Episode")
        plt.ylabel("Turn")
        
        plt.figure()
        plt.plot(range(len(losses))[-100:],losses[-100:], ".", color="red", alpha=0.1)
        average=[np.mean(losses[j:j+25]) for j in range(len(losses)-25)]
        plt.plot((np.array(range(len(average)))+12)[-100:], average[-100:], "--", color="black")
        plt.title("MSE Loss")

        plt.figure() 
        plt.plot(range(len(cleared_row)),cleared_row, ".", color="blue", alpha=0.1)
        average=[np.mean(cleared_row[j:j+100]) for j in range(len(cleared_row)-100)]
        plt.plot(np.array(range(len(average)))+50, average, "--", color="black")
        plt.title("Row Cleared vs Episode")
        plt.xlabel("Episode")
        plt.ylabel("Row Cleared")
        
        plt.pause(0.05)

        net=player.Q_net
        lt_memory=player.lt_memory
        st_memory=player.st_memory
        if len(average):
            if average[-1]>=max(average):
                pickle.dump( net, open( "Q_net_checkpoint.p", "wb" ) )
                pickle.dump({"lt": player.lt_memory, "st": player.st_memory},  open( "player_memory.p", "wb" ) )
    else:
        render=False