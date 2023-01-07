import numpy as np
import tensorflow as tf

from agent import agent
from rtb_environment import RTB_environment, get_data
from drlb_test import drlb_test
from lin_bid_test import lin_bidding_test
from rand_bid_test import rand_bidding_test
from test_result.model.CampResult import CampResult
from test_result.model.Result import Result
from test_result.model.hyper_parameters import Parameters


# parameter_list = [camp_id, epsilon_decay_rate, budget_scaling, budget_init_variance, initial_Lambda]

def parameter_camp_test(parameters: Parameters):
    """
    This function should take a camp ID, train an agent for that specific campaign
    and then test the agent for that campaign. We start by defining the hyper-parameters.
    It (currently) takes the whole campaign as an episode.
    """

    epsilon_max = 0.9
    epsilon_min = 0.05
    discount_factor = 1
    batch_size = 32
    memory_cap = 100000
    update_frequency = 100

    camp_id = parameters.camp_id
    budget_scaling = parameters.budget_scaling
    initial_lambda = parameters.initial_Lambda
    epsilon_decay_rate = parameters.epsilon_decay_rate
    budget_init_var = parameters.budget_init * budget_scaling
    step_length = parameters.step_length
    learning_rate = parameters.learning_rate
    seed = parameters.seed
    episode_length = parameters.episode_length

    action_size = 7
    state_size = 5
    tf.compat.v1.reset_default_graph()
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    sess = tf.compat.v1.Session()
    rtb_agent = agent(epsilon_max, epsilon_min, epsilon_decay_rate,
                      discount_factor, batch_size, memory_cap,
                      state_size, action_size, learning_rate, sess)

    camp_n = ['1458', '2259', '2997', '2821', '3358', '2261', '3386', '3427', '3476']
    train_file_dict, test_file_dict = get_data(camp_n)
    test_file_dict = test_file_dict[camp_id]
    total_budget = 0
    total_impressions = 0
    global_step_counter = 0

    # for i in camp_n:
    test_camp_n = ['2997']
    for i in camp_n:
        rtb_environment = RTB_environment(train_file_dict[i], episode_length, step_length)
        total_budget += train_file_dict[i]['budget']
        total_impressions += train_file_dict[i]['imp']
        while rtb_environment.data_count > 0:
            episode_size = min(episode_length * step_length, rtb_environment.data_count)
            budget = train_file_dict[i]['budget'] * min(rtb_environment.data_count, episode_size) \
                     / train_file_dict[i]['imp'] * budget_scaling
            budget = np.random.normal(budget, budget_init_var)

            cur_lambda = rtb_environment.Lambda if rtb_environment.Lambda != 1 else initial_lambda
            state, reward, termination = rtb_environment.reset(budget, cur_lambda)
            while not termination:
                action, _, _ = rtb_agent.action(state)
                next_state, reward_until_episode_end, termination = rtb_environment.step(rtb_environment.actions[action])

                reward_ratio = (rtb_environment.episode_cur_reward + reward_until_episode_end) / rtb_environment.episode_optimal_reward
                # print(f'episode_cur_reward:{rtb_environment.episode_cur_reward}, reward_until_episode_end:{reward_until_episode_end}, reward_ratio:{reward_ratio}')
                memory_sample = (action, state, reward_ratio, next_state, termination)
                rtb_agent.replay_memory.store_sample(memory_sample)
                rtb_agent.q_learning()
                if global_step_counter % update_frequency == 0:
                    rtb_agent.target_network_update()

                rtb_agent.e_greedy_policy.epsilon_update(global_step_counter)
                state = next_state
                global_step_counter += 1

    epsilon = rtb_agent.e_greedy_policy.epsilon
    budget = total_budget / total_impressions * test_file_dict['imp'] * budget_scaling
    imp, click, cost, wr, ecpc, ecpi, optimal_reward, camp_info = drlb_test(test_file_dict, budget, initial_lambda, rtb_agent,
                                                            episode_length, step_length)
    sess.close()
    lin_bid_result = lin_bidding_test(train_file_dict[camp_id], test_file_dict, budget, 'historical')
    rand_bid_result = rand_bidding_test(train_file_dict[camp_id], test_file_dict, budget, 'uniform')
    result: Result = Result(camp_id=camp_id, parameters=parameters, epsilon=epsilon, total_budget=total_budget,
                            auctions=test_file_dict['imp'], optimal_reward=optimal_reward,
                            camp_result=CampResult(imp=imp, click=click, cost=cost, wr=wr, ecpc=ecpc, ecpi=ecpi),
                            budget=camp_info[0], lambda_value=camp_info[1], unimod=camp_info[2],
                            action_nested_values=camp_info[3], lin_bid_test=lin_bid_result,
                            rand_bid_test=rand_bid_result
                            )


    # result_dict = {'camp_id': camp_id, 'parameters': parameter_list[1:], 'epsilon': epsilon, 'total_budget': budget,
    #                'auctions': test_file_dict['imp'], 'optimal_reward': optimal_reward,
    #                'camp_result': {'imp': imp, 'click': click, 'cost': cost, 'wr': wr, 'ecpc': ecpc, 'ecpi': ecpi},
    #                'budget': camp_info[0],
    #                'lambda_value': camp_info[1], 'unimod': camp_info[2], 'action_values': camp_info[3],
    #                'lin_bid_result': lin_bid_result, 'rand_bid_result': rand_bid_result}
    return result
