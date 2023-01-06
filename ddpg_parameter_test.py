import numpy as np
import tensorflow as tf

from agent import agent
from ddpg.agent import DDPG_Agent
from ddpg_test import ddpg_test
from rtb_environment import RTB_environment, get_data
from drlb_test import drlb_test
from lin_bid_test import lin_bidding_test
from rand_bid_test import rand_bidding_test
from test_result.model.CampResult import CampResult
from test_result.model.Result import Result


# parameter_list = [camp_id, epsilon_decay_rate, budget_scaling, budget_init_variance, initial_Lambda]

def parameter_camp_test(parameter_list):
    """
    This function should take a camp ID, train an agent for that specific campaign
    and then test the agent for that campaign. We start by defining the hyper-parameters.
    It (currently) takes the whole campaign as an episode.
    """

    update_frequency = 100

    camp_id = parameter_list[0]
    budget_scaling = parameter_list[1]
    initial_Lambda = parameter_list[2]
    budget_init_var = parameter_list[4] * budget_scaling
    step_length = parameter_list[5]
    episode_length = parameter_list[6]

    rtb_agent = DDPG_Agent(episode_length)

    camp_n = ['1458', '2259', '2997', '2821', '3358', '2261', '3386', '3427', '3476']
    train_file_dict, test_file_dict = get_data(camp_n)
    test_file_dict = test_file_dict[camp_id]
    total_budget = 0
    total_impressions = 0
    global_step_counter = 0

    # for i in camp_n:
    test_camp_n = ['2997']
    for j in range(1):
        for i in camp_n:
            rtb_environment = RTB_environment(train_file_dict[i], episode_length, step_length)
            total_budget += train_file_dict[i]['budget']
            total_impressions += train_file_dict[i]['imp']
            while rtb_environment.data_count > 0:
                episode_size = min(episode_length * step_length, rtb_environment.data_count)
                budget = train_file_dict[i]['budget'] * min(rtb_environment.data_count, episode_size) \
                         / train_file_dict[i]['imp'] * budget_scaling
                budget = np.random.normal(budget, budget_init_var)

                cur_lambda = rtb_environment.Lambda if rtb_environment.Lambda != 1 else initial_Lambda
                state, reward, termination = rtb_environment.reset(budget, cur_lambda)
                while not termination:
                    action, _ = rtb_agent.policy(state)
                    next_state, reward_until_episode_end, termination = rtb_environment.step(action[0])

                    memory_sample = (action, state, (rtb_environment.episode_cur_reward + reward_until_episode_end) / rtb_environment.episode_optimal_reward, next_state, termination)
                    rtb_agent.buffer.record(memory_sample)
                    rtb_agent.buffer.learn()
                    # if global_step_counter % update_frequency == 0:
                    rtb_agent.update_target(rtb_agent.target_actor.variables, rtb_agent.actor_model.variables)
                    rtb_agent.update_target(rtb_agent.target_critic.variables, rtb_agent.critic_model.variables)
                    #
                    # rtb_agent.e_greedy_policy.epsilon_update(global_step_counter)
                    state = next_state
                    global_step_counter += 1

    # epsilon = rtb_agent.e_greedy_policy.epsilon
    budget = total_budget / total_impressions * test_file_dict['imp'] * budget_scaling
    imp, click, cost, wr, ecpc, ecpi, optimal_reward, camp_info = ddpg_test(test_file_dict, budget, initial_Lambda, rtb_agent,
                                                            episode_length, step_length)
    # sess.close()
    lin_bid_result = lin_bidding_test(train_file_dict[camp_id], test_file_dict, budget, 'historical')
    rand_bid_result = rand_bidding_test(train_file_dict[camp_id], test_file_dict, budget, 'uniform')

    # result_dict = {'camp_id': camp_id, 'parameters': parameter_list[1:], 'total budget': budget,
    #                'auctions': test_file_dict['imp'], 'optimal_reward': optimal_reward,
    #                'camp_result': np.array([imp, click, cost, wr, ecpc, ecpi]).tolist(), 'budget': camp_info[0],
    #                'lambda': camp_info[1], 'action values': camp_info[2], 'actions': camp_info[3],
    #                'lin_bid_result': lin_bid_result, 'rand_bid_result': rand_bid_result}

    result: Result = Result(camp_id=camp_id, parameters=parameter_list, total_budget=total_budget,
                            auctions=test_file_dict['imp'], optimal_reward=optimal_reward,
                            camp_result=CampResult(imp=imp, click=click, cost=cost, wr=wr, ecpc=ecpc, ecpi=ecpi),
                            budget=camp_info[0], lambda_value=camp_info[1], action_values=camp_info[2],
                            actions=camp_info[3], lin_bid_test=lin_bid_result,
                            rand_bid_test=rand_bid_result
                            )
    return result
