from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from ddpg_agent import Agent 
from collections import deque
import torch
import torch.nn.functional as F
import time
from workspace_utils import active_session
import os

# env_name = "/root/DDPG/DDPG_MMC/robotics_reaching_executable_linux/robotics_reaching_exe_linux.x86_64" # Path to robotics reaching exe
env_name = "/root/DDPG/DDPG_MMC/robotics_reaching_executable_linux_no_log/robotics_reaching_exe_linux.x86_64" # Path to robotics reaching exe wihtout script debugging

# Ensure the executable has the necessary permissions 
os.chmod(env_name, 0o755)

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

        # Get what actions the environment expects and the required shape
        behaviour_spec = env.behavior_specs[behaviour_name]
        print(f"Behaviour specifications: {behaviour_spec.action_spec}")

        # Get the number of agents 
        decisionSteps, terminalSteps = env.get_steps(behavior_name=behaviour_name)
        num_agents = len(decisionSteps) + len(terminalSteps)
        print(f"Number of agents: {num_agents}")

except Exception as e:
    print(f"Error initializing environment: {e}")
    behavior_name = None


agent = Agent(state_size=8, action_size=2, random_seed=2) # Altered from origional to fit new environment

# Code runs until this point 

def ddpg(n_episodes=5, max_t=1000):
    
    print("Enter ddpg...\n")
    scores_deque = deque(maxlen=100)
    scores = []
    actions = []
    best_score = 0
    best_average_score = 0
    try:
        for i_episode in range(1, n_episodes+1):

            print(f"Episode number: {i_episode}")
            avg_score = 0

            # reset the environment
            env.reset()

            #get the decision and terminal steps
            decisionSteps, terminalSteps = env.get_steps(behavior_name=behaviour_name)
            print("Printing decisionSteps: " )
            print(decisionSteps)
            print(type(decisionSteps))

            # get number of agents
            num_agents = len(decisionSteps)
            print(f"Number of agents: {num_agents}")

            # get number of continuous actions
            num_continuous_actions = env.behavior_specs[behaviour_name].action_spec.continuous_size

            # create 2D numpy array of continuous actions 
            continuous_actions = np.random.rand(num_agents, num_continuous_actions).astype(np.float32)

            # create actiontuple
            action_tuple = ActionTuple(continuous=continuous_actions)

            # get the states vector
            stateVector = decisionSteps.obs[0]

            #init score agents
            scores_agents = np.zeros(num_agents)
            print("scores_agents type: ", type(scores_agents))
            print("scores_agents shape: ", scores_agents.shape)

            score = 0
            agent.reset()

            for t in range(max_t):

                try:
                    # Checkpoint to ensure it's not getting stuck
                    if t % 100 == 0: 
                        print(f"Progressing at step {t}")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

                # set the actions for the behaviour and step the environment
                env.set_actions(behavior_name=behaviour_name, action=action_tuple)

                # Step the environment to get the next states 
                env.step()

                # get the next states
                decisionSteps, terminalSteps = env.get_steps(behavior_name=behaviour_name)
                print(f"Step {t}, Decision steps: {len(decisionSteps)}, Terminal steps: {len(terminalSteps)}")

                # Check if all agents are in terminal state
                if len(decisionSteps)==0 and len(terminalSteps)>0:
                    print(f"All agents are in terminal states at step {t}. Ending episode early.")
                    break

                # extract the next states vector from the decision steps 
                next_state_vector = decisionSteps.obs[0]
                print("Next state vector: ", next_state_vector)

                # get the rewards
                rewards = decisionSteps.reward
                print("rewards type: ", type(rewards))
                print("rewards shape: ", rewards.shape)
                print("rewards: ", rewards)

                episode_finished = np.array([len(terminalSteps) > 0] * num_agents) # episode_fiished values must be passed into agent.step function as an array
                print("Episode finished: ", episode_finished)

                # see if episode has finished
                if next_state_vector is not None: 
                    agent.step(stateVector, actions, rewards, next_state_vector, episode_finished)
                    stateVector = next_state_vector
                #Check if scores-agents and rewards are compatible for addition
                scores_agents = np.add(scores_agents, rewards)
                # scores_agents += rewards
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
    finally:
        env.close() # Ensure env.close() is alwyas called    
    return scores

start = time.time()
with active_session():
    scores = ddpg()
end = time.time()
print('\nTotal training time = {:.1f} min'.format((end-start)/60))