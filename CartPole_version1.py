#original code for cartpole
import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop,Adam
from keras import optimizers
from collections import deque
import copy
import matplotlib.pyplot as plt

nb_games=100000
#learning_rate=0.002
learning_rate=0.0001
batch_size=64
#rate=1.0
rate=0.7
gamma=0.95


class my_agent:
    def __init__(self, input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.model=self.myNN()

        
    def myNN(self):
        model=Sequential()
        # 4 inputs representings the observations
        model.add(Dense(30,input_dim=self.input_size,activation='relu'))
        model.add(Dense(30,activation='relu'))
        #model.add(Dense(20,activation='relu'))
        # 2 outputs representing representing 0 and 1 (left, right)
        model.add(Dense(self.output_size,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=learning_rate))
        return model
    
    def agent_action(self, state):
        predict_val=self.model.predict(state)
        return np.argmax(predict_val[0])


    
    def replay(self):
        minibatch = random.sample(my_memory,batch_size)
        
        for state, action, reward, futur_state, done in minibatch:
            target = reward
            if not done:
                target = reward + gamma * \
                       np.amax(self.model.predict(futur_state)[0])   
            target_f = self.model.predict(state)
            target_f[0][action] = target
            #the neural network will predict the reward given a certain state
            # approximate the output based on the input
            self.model.fit(state, target_f, epochs=1, verbose=0)

            

        
if __name__=="__main__":
    my_memory=deque(maxlen=100000)
    decision_memory=deque(maxlen=100)
    data=[]
    data2=[]
    env = gym.make('CartPole-v1')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    agent=my_agent(input_size,output_size)
    done=False
    enough_data=0
    reward=0
    

    for i in range(nb_games):
        # Obtain an initial observation of the environment
        state=env.reset()
        state=np.reshape(state,[1,input_size])
        reward_sum=0
        reward_avg=0
        #reward=0
        
        for t in range(700):
            #env.render()
            if(i<80) or np.random.rand()<rate:
                action=random.randrange(output_size)
            else:
                action=agent.agent_action(state)
            #rate = rate * 0.8   
            

            #action represent either 0 or 1, when we pass it to the env which represents the game environment, 
            #it emits the following 4 outputs
            futur_state,reward,done,info=env.step(action)
            futur_state=np.reshape(futur_state,[1,input_size])
            if t>batch_size:
                enough_data=1 
            reward=(reward -5) if done else (reward+1)
            my_memory.append((state,action,reward,futur_state,done))
            state=futur_state
            
            
            if done:
                print("Game number : {}/{},score: {}" .format(i,nb_games,t))
                decision_memory.append(t)
                data.append(t)
                rate = rate * 0.8 
                break
        
        if i>100: 
            copy_mem=copy.deepcopy(decision_memory)
            for j in range(99):
                reward_sum = reward_sum + copy_mem.pop()
                
                    
            reward_avg=reward_sum/100
            data2.append(reward_avg)

            
            if reward_avg >= 195.0:
                print("\n Problem solved, average reward :", reward_avg)
                #break
                
        
        if enough_data==1:
            agent.replay()
        
        
        if i==(10000):
           
            plt.plot(data2,'g')
            plt.xlabel('Game number +100')
            plt.ylabel('Average game score')
            plt.title('score variation')
            plt.show()
            break





    
