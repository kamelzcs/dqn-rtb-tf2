import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

###TESTING---------------------------------------------------------------------
from ddpg.agent import DDPG_Agent
# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"

epsilon_max = 1
epsilon_min = 0.01
epsilon_decay_rate = 0.000025
discount_factor = 0.99
batch_size = 50
memory_cap = 50000
update_frequency = 200
random_n = 25000
episodes_n = 3000

tf.compat.v1.reset_default_graph()


env = gym.make('MountainCar-v0')
env.seed(0)

agent = DDPG_Agent(gamma=0.99, num_states=env.observation_space.shape[0],
                   num_actions=env.action_space.n,
                   critic_lr=1e-3, actor_lr=1e-3)
random_training_counter = 0
state = env.reset()
termination = False
for i in range(random_n):
    action = np.random.randint(env.action_space.n)
    next_state, reward, termination = env.step(action)[:3]

    memory_sample = (action, state, reward, next_state, termination)
    agent.buffer.record(memory_sample)

    random_training_counter += 1
    # print(random_training_counter)

    if (termination == True):
        state = env.reset()
    else:
        state = next_state

episode_counter = 1
global_step_counter = 0
while (episode_counter < episodes_n):
    if (episode_counter % 10 == 0):
        print('Episode {} of {}'.format(episode_counter, episodes_n))

    state = env.reset()
    termination = False
    action_sum = 0
    while not termination:
        action, _ = agent.policy(state)
        action_sum += action
        if action_sum > 10:
            print(action_sum)
        next_state, reward, termination = env.step(action)[:3]

        agent.reward_episode += reward
        memory_sample = (action, state, reward, next_state, termination)
        agent.buffer.record(memory_sample)

        global_step_counter += 1
        agent.buffer.learn()
        # if (global_step_counter % update_frequency == 0):
        agent.update_target(agent.target_actor.variables, agent.actor_model.variables)
        agent.update_target(agent.target_critic.variables, agent.critic_model.variables)


        # agent.e_greedy_policy.epsilon_update(global_step_counter)
        state = next_state
        if (global_step_counter > 2 * 100000):
            import platform
            if platform.system() != 'Linux':
                env.render()

    print("Step {},Episode reward: {}"\
          .format(global_step_counter, agent.reward_episode))
    agent.reward_list.append(agent.reward_episode)
    agent.reward_episode = 0
    episode_counter += 1
        
env.close()
plt.plot(agent.reward_list)
