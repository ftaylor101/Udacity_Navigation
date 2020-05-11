import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import time

from collections import deque
from unityagents import UnityEnvironment

from project_one import Agent, QNetwork


def ddqnper(agent: Agent, env: UnityEnvironment, n_episodes: int = 2000, max_t: int = 1000, eps_start: float = 1.0,
            eps_end: float = 0.01, eps_decay: float = 0.955):
    """
    Double Deep Q-Learning algorithm with Prioritised Experience Replay, based on and adapted from Udacity code for
    the Lunar Lander.

    :param agent: an agent with a RL learning algorithm
    :param env: an environment in which the agent can explore
    :param n_episodes: maximum number of training episodes
    :param max_t: maximum number of timesteps per episode
    :param eps_start: starting value of epsilon, for epsilon-greedy action selection
    :param eps_end: minimum value of epsilon
    :param eps_decay: multiplicative factor (per episode) for decreasing epsilon
    """
    brain_name = env.brain_names[0]  # used to get latest environment information
    scores = []  # list containing scores from each episode
    average_score = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    # loop through episodes
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)
            next_state = env_info[brain_name].vector_observations[0]
            reward = env_info[brain_name].rewards[0]
            done = env_info[brain_name].local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        average_score.append(np.mean(scores_window))
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    if np.mean(scores_window) < 13.0:
        print("Environment not solved")
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_no_soln.pth')

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(scores)), scores, c='b', label='Score')
    ax.plot(np.arange(len(scores)), average_score, c='g', label='Average score')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    return scores


def demonstrate_agent(env: UnityEnvironment,
                      model_path: str = "C:/Users/Frederic/PycharmProjects/projectone/project_one/checkpoint.pth"):
    """
    A function to demonstrate the trained agent in action.

    :param env: The Unity Environment
    :param model_path: The location of the saved parameters of the Torch model
    """
    # load in torch network
    model = QNetwork(state_size=37, action_size=4, seed=0)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    brain_name = env.brain_names[0]
    print("Brain name:", brain_name)
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]

    score = 0
    scores = []
    for i in range(1000):
        state = torch.from_numpy(state).float().unsqueeze(0)
        actions = model(state)
        action = np.int32(np.argmax(actions.cpu().data.numpy()))
        env_info = env.step(action)

        next_state = env_info[brain_name].vector_observations[0]
        reward = env_info[brain_name].rewards[0]
        done = env_info[brain_name].local_done[0]
        state = next_state
        score += reward
        scores.append(score)
        print('\rStep {}\tScore: {}'.format(i, score), end="")
        time.sleep(0.25)
        if done:
            break
    # plot the scores of a trained agent
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    return scores


if __name__ == "__main__":
    # set up environment
    curr_path = os.getcwd()
    my_env = UnityEnvironment(
        file_name=os.path.join(curr_path, "Banana_Windows_x86_64/Banana.exe"))

    my_agent_gamma = Agent(gamma=0.99, tau=1e-3)

    my_scores = ddqnper(agent=my_agent_gamma, env=my_env, n_episodes=2000)
    # scores = demonstrate_agent(env=my_env, model_path=os.path.join(curr_path, "checkpoint.pth"))

    my_env.close()
