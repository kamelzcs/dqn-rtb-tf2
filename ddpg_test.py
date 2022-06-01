
import numpy as np

from rtb_environment import RTB_environment
from solver.optimal_solver import solve


def ddpg_test(test_file_dict, budget, initial_Lambda, agent, episode_length, step_length):
    """
    This function tests a bidding agent on a number of auctions from
    a single campaign and outputs the results.
    :param agent: this is the trained DQN-based bidding agent
    :param test_file_dict: a dictionary containing testing data (bids,
    ctr estimations, clicks), budget, and so on from a single campaign.
    :param budget_scaling: a scaling parameter for the budget
    :return:
    """
    test_environment = RTB_environment(test_file_dict, episode_length, step_length)
    budget_list = []
    Lambda_list = []
    action_list = []
    action_value_list = []
    episode_budget = 0
    optimal_reward = 0

    while test_environment.data_count > 0:
        episode_budget = min(episode_length * step_length, test_environment.data_count)\
                         / test_file_dict['imp'] * budget + episode_budget
        state, reward, termination = test_environment.reset(episode_budget, initial_Lambda)
        optimal_reward += solve(test_environment.episode_ctr_estimations, test_environment.episode_winning_bids, test_environment.budget)
        while not termination:
            action, action_value = agent.policy(state)
            next_state, reward, termination = test_environment.step(action[0])
            state = next_state

            budget_list.append(test_environment.budget)
            Lambda_list.append(test_environment.Lambda)
            action_value_list.append(action_value)
            action_list.append(action[0])
        episode_budget = test_environment.budget
    impressions, click, cost, win_rate, ecpc, ecpi = test_environment.result()

    return impressions, click, cost, win_rate, ecpc, ecpi, optimal_reward, \
           [np.array(budget_list).tolist(), np.array(Lambda_list).tolist(),
            np.array(action_value_list).tolist(), np.array(action_list).tolist()]