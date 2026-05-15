import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn
import torch.optim as optim

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device= "cpu"


class Agent:

    def __init__(self, param_set):
        self.param_set = param_set

        with open("paramters.yaml") as f:
            all_params_set = yaml.safe_load("f")
            print(all_params_set)
            params = all_params_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]


    def run(self, is_training=True, render=False) :
        
        env = gym.make("FlappyBird-v0", render_mode= "human" if render else None, use_lidar=False)

        num_states = env.observation_space.shape[0]  # input dimensions or sample
        num_actions = env.action_space.n            # output dimensions or sample
        policy_dqn = DQN(num_states, num_actions).to(device)


        if is_training:
            memory = ReplayMemory(10000)

        for episode in itertools.count():

            episode_rewards = 0
            terminated = False

            state, _ = env.reset()

            # One Episode
            while not terminated:
                # Each one step   
                action = env.acion_space.sample()  # independent of the old state
                next_state, reward, terminated, _, info = env.step(action)   # Processing: terminated is done with episode

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                
                episode_rewards += reward
                state = new_state 

            
            print(f"Episode = {episode} has total rewards = {episode_rewards}")


                

        # env.close  # manuall stop 