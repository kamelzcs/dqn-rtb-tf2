import numpy as np
import tensorflow as tf

from ddpg.OUActionNoise import OUActionNoise
from ddpg.buffer import Buffer
from ddpg.ddpg_network import DDPG_Network


class DDPG_Agent:
    def __init__(self):
        self.upper_bound = 0.1
        self.lower_bound = -0.1

        std_dev = 0.2
        num_states = 96 + 5
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        ddpg = DDPG_Network(num_states=num_states)

        self.actor_model = ddpg.get_actor()
        self.critic_model = ddpg.get_critic()

        self.target_actor = ddpg.get_actor()
        self.target_critic = ddpg.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        critic_lr = 1e-5
        actor_lr = 1e-5

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        total_episodes = 100
        # Discount factor for future rewards
        gamma = 1.0
        # Used to update target networks
        self.tau = 1e-4

        self.buffer = Buffer(gamma=gamma, target_actor=self.target_actor, target_critic=self.target_critic,
                             critic_model=self.critic_model, critic_optimizer=critic_optimizer,
                             actor_model=self.actor_model, actor_optimizer=actor_optimizer,
                             num_states=num_states
                             )

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(np.expand_dims(state, axis=0)))
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + self.ou_noise()

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)], np.squeeze(self.critic_model([np.expand_dims(state, axis=0), legal_action], training=False))

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
