import flappy_bird_gymnasium
import gymnasium as gym
env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)

state, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.acion_space. sample()

    # Processing:
    next_state, reward, terminated, _, info = env.step(action)

    # Checking if the player is still alive
    if terminated:
        break

env.close