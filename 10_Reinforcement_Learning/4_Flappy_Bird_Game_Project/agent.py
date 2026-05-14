import flappy_bird_gymnasium
import gymnasium as gym

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device= "cpu"


def run(self, is_training=True, render=False) :
    
    env = gym.make("FlappyBird-v0", render_mode= "human" if render else None, use_lidar=False)

    state, _ = env.reset()

    while True:
        # Next action:
        # (feed the observation to your agent here)
        action = env.acion_space.sample()

        # Processing:
        next_state, reward, terminated, _, info = env.step(action)

        # Checking if the player is still alive
        if terminated:
            break

    env.close