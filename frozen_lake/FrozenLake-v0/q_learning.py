import gym
import numpy as np

env = gym.make('FrozenLake-v0')
q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))

learning_rate = 0.8
discount_factor = 0.95
num_episodes = 2000

reward_list = []
for run_number in range(num_episodes):
    state = env.reset() # returns the initial observation space
    total_reward = 0
    task_complete = False
    actions_taken = 0
    while actions_taken < 99:
        actions_taken += 1
        action = np.argmax(q_table[state, :] + np.random.randn(1, env.action_space.n) / (run_number + 1))
        next_state, reward, task_complete, _ = env.step(action)
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
        total_reward += reward
        state = next_state
        if task_complete:
            break
    reward_list.append(total_reward)
print(q_table)
