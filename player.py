# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:53:17 2022

@author: hibad
"""
import sys
import numpy as np
sys.path.append('envs')
from tetris import TetrisEnv
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle 

class Net(nn.Module):

    def __init__(self, n):
        super(Net, self).__init__()
        self.cov11= nn.Conv2d(1,1,(2,2))
        self.cov21= nn.Conv2d(1,1,(2,2))

        self.cov12= nn.Conv2d(1,1,(3,3))

        self.fc1 = nn.Linear(86,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,100)
        self.fc4 = nn.Linear(100,1)

    def forward(self, x):
        x1= x[0].view(1,x[0].shape[0],x[0].shape[1])
        x11=F.leaky_relu(self.cov11(x1))
        x21=F.leaky_relu(self.cov21(x11))
        
        x12=F.leaky_relu(self.cov12(x1))
        
        
        

        x21=F.max_pool2d(x21, kernel_size=((2,2)),stride=(2,2))
        x12=F.max_pool2d(x12, kernel_size=((2,2)),stride=(2,2))


        x21=x21.view(1,x21.shape[1]*x21.shape[2])
     

        x12=x12.view(1,x12.shape[1]*x12.shape[2])
        x3=x[1].view(1, x[1].shape[0])
        
        x=torch.cat((x21,x12,x3),dim=1)

        x =F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x= F.leaky_relu(self.fc3(x))
        x= self.fc4(x)
        x=torch.clamp(x, min=-9000, max=4000)
        return x
#%%
class Player:
    def __init__(self, gamma, net):
        self.gamma=gamma
      #  self.learning_rate=learning_rate
        self.Q_net=net
        self.optimizer=optim.Adam(self.Q_net.parameters())
        self.criterion=nn.MSELoss()
        self.st_memory=[]
        self.lt_memory=[]
        self.greedy=0
        
    def get_reward(self, env):
        """
        reward function
        """
        if env.state.lost:
            reward=-9000
        elif env.cleared_current_turn:
            reward=env.cleared_current_turn*100
        else:         
            reward=1
        return reward
    
    def get_state_vector(self, state):
        # one_hot=np.zeros(7)
        # one_hot[state.next_piece]=1
        field=state.field>0
        holes=sum([sum(state.field[0:top, i]==0) for i,top in enumerate(state.top)])
        top=max(state.top)
        std=np.std(state.top)

        return  (torch.Tensor(field).to(device), 
                  torch.from_numpy(np.concatenate((state.top, [top,holes,std,1])).astype(np.float32)).to(device))

    def policy(self,state, actions):
        Q=np.zeros(len(actions))
        for i, action in enumerate(actions):
            model=TetrisEnv()
            model.set_state(state)
            state_prime, _, done, _ = model.step(action)
            reward=self.get_reward(model)
            if done:
                Q[i]=-np.inf
            else:
                Q[i]=self.greedy*reward+(1-self.greedy)*self.Q_function(state, action)
            
        if np.random.randint(100):

            action=actions[np.argmax(Q)]
        else:
            candidate=min(3, len(actions))
            ind = np.argpartition(Q, -candidate)[-candidate:]
            action=actions[ind[np.random.randint(candidate)]]
        return action
    
    def Q_function(self, x,a):
        sim = TetrisEnv()
        sim.set_state(x)
        x_prime, _, done , _ = sim.step(a)
        r=self.get_reward(sim)
        if done:
            return r
        else:
            h=self.Q_net.forward(self.get_state_vector(x_prime))
        return r+self.gamma*h
    
    def learn(self):
        #collect batches
       
        training_batch=[]
        losses=[]
        for item in self.st_memory:
            training_batch+=item.copy()
        if len(self.lt_memory)>0:
            for _ in range(1):
                training_batch+=(self.lt_memory[np.random.randint(0,len(self.lt_memory))]).copy()
        
        if len(training_batch)<=5000:
            return 0
        print("learning")

        # ds_index=np.random.choice(len(training_batch), 5000, replace=False)
        # batch=[training_batch[i] for i in ds_index]
        np.random.shuffle(training_batch)
        batch=training_batch
        #process batch
        for i in range(int(np.floor(len(batch)/500))):
            target=[]
            inputs=[]
            for  x, a, r, x_prime,actions_prime in batch[500*i:min(500*i+500,len(batch) )]:
                #compute target 
                if not x_prime.lost:
                    Q_prime=np.zeros(len(actions_prime))
                    for i, action in enumerate(actions_prime):
                        Q_prime[i]=self.Q_function(x_prime, action)
                    
                    target.append(max(Q_prime))
                else:
                    target.append(0)
                    
                inputs.append(self.Q_net(self.get_state_vector(x_prime)).float())
            inputs=torch.cat(inputs, dim=1)                
            target=torch.from_numpy(np.array(target, dtype=np.float32)).to(device).detach()

            inputs=inputs.view(1, 500)
            target=target.view(1, 500)
            
            self.optimizer.zero_grad()
            loss=self.criterion(inputs,target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        self.st_memory=[batch[-int(len(batch)%500):]]
        return np.mean(losses)
        
    def update(self, batch):
        self.st_memory.append(batch.copy())
        self.lt_memory.append(batch.copy())
        self.lt_memory=self.lt_memory[-1000:]

        if len(self.st_memory)>=10:       
           loss=self.learn()
           return loss
        else:
            return 0

if __name__ == "__main__":
    net=pickle.load( open( "Q_net_checkpoint.p", "rb" ) )
    player=Player(0.9, net=net)
    cleared=[]
    for _ in range(20):
        env = TetrisEnv()
        state=env.reset()
        done=False
        while not done:
            actions = env.get_actions()
            action = player.policy(state, actions)
            state_prev=np.copy(state)
            state, _, done, _ = env.step(action)
            actions_prime = env.get_actions()
            if done:
                break
            env.render()
        cleared.append(state.cleared)
    print("Average Line cleared: ", np.mean(cleared), "+-", np.std(cleared))
    print("Max Line cleared: ", np.max(cleared))
