# CartPole
This project will demonstrate how deep reinforcement learning can be implemented and applied to play a CartPole game using Keras and Gym.

#### The state in CartPole is composed by 4 observations :
- Cart position
- Cart velocity
- Pole angle
- Pole velocity at tip

#### Actions : 
- Left
- Right

#### Reward :
- +1 for every t step 
- -10 in the end of a game

#### Episode termination :
- Pole Angle is more than ±20.9°
- Cart Position is more than ±2.4 


#### Game solved :
Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

#### The neural network:
- Input layer: 4 nodes receiving the state (4 observations)
- N hidden layers with M nodes 
- Output layer: 2 nodes (2 actions possible)


### Reinforcement learning
Q-function: used to approximate the reward based on a state

Q(s,a) :Calculates the expected future value from state "s" and action "a"

#### Problem in reinforcement learning = unstable / diverge when neural networks are used to represent the action-value function.
Causes:
- Correlation present in the sequence of observations
- small updates to Q may change the data distribution
- correlations between the action-values (Q) and the target value (r+ gamma * Q(s,a))

#### Solution:
- use a replay function that randomizes over the data => removing correlation in the observation sequence and smoothing changes
- use an iterative update that adjusts the action-values (Q) toward target values that are only periodically adjusted => reducing correlations with the target

#### How to use theses solutions?
- We need to store the agent's experiences (memory) 
- During learning, we apply Q-learning updates on minibatches
- The target at iteration "i" only updates with the Q-network parameters of iteration "i-1"
- Calculate average score-per-episode + average predicted 




### References
Human-level control through deep reinforcement learning,


### Installation Dependences
Python
TensorFlow
Keras
Gym
