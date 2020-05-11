# Project 1 - Udacity Banana Collector
## Challenge
The environment to solve involves training a banana collector agent to 
navigate a square world and pick up yellow bananas whilst avoiding blue 
bananas.

A reward of +1 is given for every yellow banana collected and -1 for every 
blue banana collected. The goal is to pick up as many yellow bananas as 
possible. The task is episodic and the environment is considered solved 
when an average score of +13 is achieved over 100 consecutive episodes.

The state space has 37 dimensions. It contains the agent's velocity along 
with a ray-based perception of objects around the agent's forward 
direction in a continuous observation space.

For the action space, there are 4 discrete actions which correspond to: 
0 (forwards), 1 (backwards), 2 (left) and 3 (right). The agent can choose 
any of these at each timestep.

## Development environment
+ This agent has been trained using __Python 3.6.8__ on __Windows 10__
+ The __requirements.txt__ file contains all required Python packages
+ To install these packages, navigate to your Python virtual 
environment, activate it and use: 
     - pip install -r requirements.txt 

The Banana Collector environment is a modified version of the one 
provided by [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). 
However it is not the same and for this project the modified version 
provided by Udacity has been used for training.

__Ensure__ the location of the Udacity modified Banana Collector 
environment folder is in the same folder as the banana_collector.py 
file. This will make sure the code finds the environment.

## Running code
banana_collector.py contains the entry point to the code, whether for 
training a new agent or for viewing a previously trained agent.
#### Training
To run the code and train a new agent, run banana_collector.py. This script 
will instantiate a Unity Environment and an Agent class and pass these as 
arguments to the "ddqnper" function. This function loops through episodes 
and passes the states, actions and rewards to the agent for training. 
Make sure the Unity Environment is given the correct argument for the 
file_name parameter, which should be the the file path, file name and 
extension of the Unity Environment.

To run (for training):
* Check file_name parameter for Unity Environment
* Check the hyperparameters given to the Agent class
* Call the "ddqnper" method with the Agent and the Unity environment as 
arguments
#### Demonstrating
To run the code and view the trained agent in action in its environment, 
make an edit to banana_collector.py at lines 132 - 133, to comment out the 
"ddqnper" function and to call the "demonstrate_agent" function. Again, 
make sure the Unity Environment has the correct file_name argument. The 
other parameter to demonstrate the agent is the location including 
filename and extension of the saved model weights. The file with the best 
weights has been provided as part of this submission under 
checkpoint_final.pth.

To run (for demonstration):
* Check file_name parameter for Unity Environment
* Find the checkpoint file to demonstrate
* Call the "demonstrate_agent" method with the Unity environment and the 
chosen checkpoint file containing the model weights file as arguments
