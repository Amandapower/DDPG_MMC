# from mlagents import UnityEnvironment # unityagents pkg changed to mlagents pkg
# import mlagents
# import mlagents_envs
from mlagents_envs.environment import UnityEnvironment
import numpy as np
from ddpg_agent import Agent 
from collections import deque
import torch
import torch.nn.functional as F
# import torch.optim as optim
import time
from workspace_utils import active_session
# import os

env_name = "C:\\Users\\Amanda\\Documents\\MouseNet Research\\Roller_ball_test\\Roller_ball_executable\\Roller_ball_test.exe"
# env_name = "C:\\Users\\Amanda\\Documents\\MouseNet Research\\robotics_reaching_environment\\hand_agent_executable\\robotics_reaching_environment.exe" # Path to unvity environment binary to launch

try: 
    # Launch unity environment
    env = UnityEnvironment(file_name=env_name,seed=1, side_channels=[])

    # Start the environment 
    env.reset()

    # Get behaviour names 
    behaviour_names = env.behavior_specs.keys()

    # Check that behaviour names have been retrieved from the environment
    if not behaviour_names:
        print("No behaviours found. Ensure that the unity environment has agents with behaviours")
    else:
        behaviour_name = list(env.behavior_specs.keys())[0]
        print(f"Behaviour name: {behaviour_name}")

        # Get the number of agents 
        decisionSteps, terminalSteps = env.get_steps(behavior_name=behaviour_name)
        num_agents = len(decisionSteps) + len(terminalSteps)
        print(f"Number of agents: {num_agents}")

except Exception as e:
    print(f"Error initializing environment: {e}")


agent = Agent(state_size=8, action_size=2, random_seed=2) # Altered from origional to fit new environment

# Code runs until this point 

def ddpg(n_episodes=2000, max_t=1000):
    
    print("Enter ddpg...\n")
    scores_deque = deque(maxlen=100)
    scores = []
    best_score = 0
    best_average_score = 0
    for i_episode in range(1, n_episodes+1):
        
        avg_score = 0

        # reset the environment
        env.reset()

        #get the decision and terminal steps
        decisionSteps, terminalSteps = env.get_steps(behavior_name=behaviour_name)

        # get number of agents
        num_agents = len(decisionSteps) + len(terminalSteps)
        print(f"Number of agents: {num_agents}")

        # get the states vectory
        stateVector = decisionSteps.obs[0]

        #init score agents
        scores_agents = np.zeros(num_agents)
        score = 0
        agent.reset()
        for t in range(max_t):

            #choose actions
            actions = agent.act(stateVector)

            # set the actions for the behaviour and step the environment
            env.set_actions(behavior_name=behaviour_name, action=actions)

            # Step the environment to get the next states 
            env.step()

            # get the next states
            # get the decision and terminal steps
            decisionSteps, terminalSteps = env.get_steps(behavior_name=behaviour_name)

            # extract the next states vector from the decision steps 
            next_state_vector = decisionSteps.obs[0]
            print("Next state vector: ", next_state_vector)

            # get the rewards
            rewards = decisionSteps.reward
            print("Rewards: ", rewards)

            # rewards = env_info.rewards
            episode_finished = len(terminalSteps) > 0
            print("Episode finished: ", episode_finished)

            # see if episode has finished
            agent.step(stateVector, actions, rewards, next_state_vector, episode_finished)
            stateVector = next_state_vector
            scores_agents += rewards
            if np.any(episode_finished):
                break

        # mean score of 20 agents in this episode
        score = np.mean(scores_agents)
        scores_deque.append(score)
        avg_score = np.mean(scores_deque)
        scores.append(score)

        #refresh the best agent score
        if score > best_score:
            best_score = score

        #refresh the best average score    
        if avg_score > best_average_score:
            best_average_score = avg_score
        
        #print current episode
        print("Episode:{}, Score:{:.2f}, Best Score:{:.2f}, Average Score:{:.2f}, Best Avg Score:{:.2f}".format(
            i_episode, score, best_score, avg_score, best_average_score))
        if (avg_score >= 32):
            torch.save(agent.actor_local.state_dict(), 'actor_solved.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_solved.pth')
            break
                
        
    env.close() 
    return scores

start = time.time()
with active_session():
    scores = ddpg()
end = time.time()
print('\nTotal training time = {:.1f} min'.format((end-start)/60))