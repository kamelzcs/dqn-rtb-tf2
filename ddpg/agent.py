import numpy as np
import tensorflow as tf

from ddpg.OUActionNoise import OUActionNoise
from ddpg.buffer import Buffer
from ddpg.ddpg_network import DDPG_Network


class DDPG_Agent:
    def __init__(self, episode_length=96, upper_bound=0.1, lower_bound=-0.1,
                 critic_lr=1e-6, actor_lr=1e-6, gamma=1.0, num_states=5, num_actions=1):

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        ddpg = DDPG_Network(num_states=num_states, num_actions=num_actions)

        self.actor_model = ddpg.get_actor()
        self.critic_model = ddpg.get_critic()

        self.target_actor = ddpg.get_actor()
        self.target_critic = ddpg.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # for MountainCar
        self.reward_episode = 0
        self.reward_list = []


        # Learning rate for actor-critic models
        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        total_episodes = 100
        # Discount factor for future rewards
        # gamma = 1.0

        # Used to update target networks
        self.tau = 0.05

        self.buffer = Buffer(gamma=gamma, target_actor=self.target_actor, target_critic=self.target_critic,
                             critic_model=self.critic_model, critic_optimizer=critic_optimizer,
                             actor_model=self.actor_model, actor_optimizer=actor_optimizer,
                             num_states=num_states, num_actions=num_actions)

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(np.expand_dims(state, axis=0)))
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + self.ou_noise()

        # We make sure action is within bounds
        # legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        legal_action = sampled_actions

        squeeze_legal_action = np.squeeze(legal_action)
        # print(f'legal_action: {legal_action}, state:{state}')
        values = self.critic_model([np.expand_dims(state, axis=0), np.expand_dims(legal_action, axis=0)], training=False)
        return np.argmax(values), np.squeeze(values)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
