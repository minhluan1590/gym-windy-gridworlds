import gymnasium as gym
import gym_windy_gridworlds
import collections
import numpy as np

"""
In this file, we will implement the SARSA algorithm to solve the Windy Gridworld problem, described in Example 6.5 of the book.
"""

# We will use the WindyGridWorld gym environment, implemented at: https://github.com/ibrahim-elshar/gym-windy-gridworlds
ENV_NAME = "WindyGridWorld-v0"

# The discount factor
GAMMA = 1.0

# The number of episodes to test the agent
TEST_EPISODES = 200

# Define the Agent class
class Agent:
    def __init__(self):
        # Create the environment
        self.env = gym.make(ENV_NAME)

        # Initialize the state
        self.state = self.env.reset()[0]

        # Initialize the values table
        self.values = collections.defaultdict(float)

        # Initialize the learning rate
        self.alpha = 0.5

        # Initialize the epsilon value
        self.epsilon = 0.1

    def play_n_random_steps(self, count):
        # Play n random steps to initialize the values table
        for _ in range(count):
            # Select a random action
            action = self.env.action_space.sample()

            # Perform the action
            new_state, reward, terminated, truncated, _ = self.env.step(action)

            # Update the values table
            self.values[(self.state, action)] = reward

            # Update the state
            self.state = self.env.reset()[0] if (terminated or truncated) else new_state

    def select_action(self, state):
        # Select an action using epsilon-greedy policy
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            # Get the value of the current state-action pair
            value = self.values[(state, action)]

            # Update the best action and value
            if best_value is None or best_value < value:
                best_value = value
                best_action = action

        # Select an action using epsilon-greedy policy
        if self.epsilon > 0.0:
            # Select a random action
            if self.epsilon > np.random.random():
                return self.env.action_space.sample()
            # Select the best action
            else:
                return best_action
        # Select the best action
        else:
            return best_action

    def play_episode(self, env):
        # Play an episode
        total_reward = 0.0
        state = env.reset()[0]
        while True:
            # Select an action
            action = self.select_action(state)

            # Perform the action
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Update the values table
            self.values[(state, action)] += self.alpha * (
                reward
                + GAMMA * self.values[(new_state, self.select_action(new_state))]
                - self.values[(state, action)]
            )

            # Update the total reward
            total_reward += reward

            # Update the state
            state = env.reset()[0] if (terminated or truncated) else new_state

            # Break if the episode is terminated
            if terminated:
                break

        return total_reward


if __name__ == "__main__":
    # Create the agent
    agent = Agent()

    # Play n random steps to initialize the values table
    agent.play_n_random_steps(100)

    # Play 200 episodes using the learned policy
    for episode in range(TEST_EPISODES):
        # Play an episode
        reward = agent.play_episode(agent.env)

        # Print the reward
        print("Episode: {}, Reward: {}".format(episode, reward))


