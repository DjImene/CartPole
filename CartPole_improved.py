#New version of the code for cartpole "improved"

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

#The number of games must be high enough to observe the variation and see if there is a permanent state
nb_games=100000
#The best learning rate is 0.0001
#I've tried others values and the result isn't as good as this value
#learning_rate=0.002
learning_rate=0.0001
#I've tried diffrent batch sizes as 32, 64, 128,... and I've found out that 64 is giving the best results
batch_size=64
rate=1.0
#rate=0.7
gamma=0.95

#We define an agent that learn from the environment by interacting with it
#we get observations after each of it's actions on the environment
#we store those experiences so our agent could learn from its mistakes
class my_agent:
    #The init funcion is an initialization function that calls the "myNN" function which builds our neural network
    #it also defines two constant values, the input and output sizes of our neural network
    def __init__(self, input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.model=self.myNN()

    #Defining the neural network, fixed number of neurones for the input and output layers
    #For different number of neurons and different number of hidden layer, the results are not as good as the ones
    #provided with this specific modal, others activations functions as well as optimizers and losses have been tested
    #better results have been found with the modal bellow
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
    
    #The action can be either 0 or 1
    def agent_action(self, state):
        predict_val=self.model.predict(state)
        return np.argmax(predict_val[0])


    #This function is used to train the neural network bases on the memory(the previous experiencies or actions it had)
    def replay(self):
        minibatch = random.sample(my_memory,batch_size)
        
        for state, action, reward, futur_state, done in minibatch:
            target = reward
            if not done:
                #target=best_reward
                target = reward + gamma * np.amax(self.model.predict(futur_state)[0])   
            target_f = self.model.predict(state)
            target_f[0][action] = target
            #the neural network will predict the reward given a certain state
            # approximate the output based on the input
            self.model.fit(state, target_f, epochs=1, verbose=0)

            

        
if __name__=="__main__":
    #my_memory is the memory in which we store the previous experiences
    #the size of the memory is fixed to 1000 in this case so the first experiencies are removed sequentially
    #we get better results when we remove some of the previous experiences and keep the most recent ones
    my_memory=deque(maxlen=1000)
    #the decision memory is the memory in which we store the last 100 results so we can decide if the average of those 100
    #exepriences is good enough to consider that our agent has learned
    decision_memory=deque(maxlen=100)
    #data memories are memories in which we store the data to be used for our plots
    data=[]
    data2=[]
    #we decide which environment is to be used, in this case it's CartPole
    env = gym.make('CartPole-v1')
    #the input_size is the number of observations we get from our environment, in this case it's 4
    #cart position, cart velocity, pole angle, pole velocity
    input_size = env.observation_space.shape[0]
    #the output size is the number of actions we could make to interact with the environment,
    # in this case we have 2 actions, moving right or moving left
    output_size = env.action_space.n
    #we initialize our agent with the input and output sizes
    agent=my_agent(input_size,output_size)
    #we initialize done with false, it becomes true when the game ends
    done=False
    #we initialize enough_data with 0, it becomes 1 when we store enough data (data>batch_size) to begin the training
    enough_data=0
    best_reward=0
    

    for i in range(nb_games):
        # Obtain an initial observation of the environment
        state=env.reset()
        state=np.reshape(state,[1,input_size])
        reward_sum=0
        reward_avg=0
        #reward=0
        
        for t in range(1000):
            reward=0
            #env.render()
            #at first our agent act randomly then when it start to learn it acts randomly occasionaly
            if(i<80) or np.random.rand()<rate:
                action=random.randrange(output_size)
            else:
                #we send the current state (4 observation) to our neural network and it predicts the action to be made
                action=agent.agent_action(state)
            rate = rate * 0.8   
            
            reward=t
            
            #action represent either 0 or 1, when we pass it to the env which represents the game environment, 
            #it emits the following 4 outputs
            futur_state,reward,done,info=env.step(action)
            futur_state=np.reshape(futur_state,[1,input_size])
            if t>batch_size:
                enough_data=1 
            

            #reward=(reward -5) if done else (reward+1)
            #All the actions are stored into a memory, to be used afterwise for the training step
            my_memory.append((state,action,reward,futur_state,done))
            state=futur_state
            
            
            if done:
                reward=t-10
                print("Game number : {}/{},score: {},reward:{},best reward:{}" .format(i,nb_games,t,reward,best_reward))
                decision_memory.append(t)
                data.append(t)
                #rate = rate * 0.8 
                if reward>best_reward:
                    best_reward=reward
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
                
        #my_memory.remove((reward=-1))
                
        #if i==300:
        #      for l in range(200):
        #      my_memory.popleft()
        #if i>400 and (i%300)==0:
        #    for l in range(100):
         #       my_memory.popleft()
        
        
        #If we have enough data we can start the training
        if enough_data==1:
            agent.replay()
        





plt.plot(data2,'g')
plt.xlabel('Game number +100')
plt.ylabel('Average game score')
plt.title('score variation')
plt.show()


plt.plot(data,'r')
plt.xlabel('Game number')
plt.ylabel('Game score')
plt.title('score variation')
plt.show()

