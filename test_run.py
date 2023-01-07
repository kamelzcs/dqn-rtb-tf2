import random

import ddpg_parameter_test
from parameter_test import parameter_camp_test
import json
from multiprocessing import Pool, Process

from test_result.model.Result import Results
from test_result.model.hyper_parameters import Parameters


standard_camp_id = '1458'
test_camp_id = '2997'
camp_id = standard_camp_id
def main():
    # faulthandler.enable()
    # faulthandler.register(signal.SIGINT.value)
    with open('test_result/time_split_dqn_multi_optimal', 'w') as dnq_multi, \
            open('test_result/time_split_ddpg_multi_optimal', 'w') as ddpg_multi:
        json.dump(json.loads(ddpg_test_multi().json()), ddpg_multi, indent=2)
        json.dump(json.loads(dqn_test_multi().json()), dnq_multi, indent=2)

    # with open('test_result/time_split_ddpg_multi_optimal', 'w') as ddpg_multi:
    #     json.dump(json.loads(ddpg_test_multi().json()), ddpg_multi, indent=2)


def dqn_test_multi():
    concurrency = 20
    # parameter_test = ['1458', 1.0 / 32] + [1e-4] + [0.0001, 2500, 500, 1e-4, random.randint(0, 100), 96]
    parameter_test = Parameters(camp_id=camp_id, budget_scaling=1.0 / 32, initial_Lambda=1e-4,
                                epsilon_decay_rate=0.0001, budget_init=2500,
                                step_length=500,
                                learning_rate=1e-4, seed=random.randint(0, 100),
                                episode_length=96)
    with Pool(processes=concurrency) as pool:
        results = pool.map(parameter_camp_test, [parameter_test for _ in range(concurrency)])
        return Results(__root__=results)


def dnq_test():
    imp_threshold = 50000
    click_threshold = 400
    results = []
    for i in range(3):
        # parameter_test = ['1458', 1.0 / 32] + [1e-4] + [0.0001, 2500, 500, 1e-4, random.randint(0, 100), 96]
        parameter_test = Parameters(camp_id='1458', budget_scaling=1.0 / 32, initial_Lambda=1e-4,
                                    epsilon_decay_rate=0.0001, budget_init=2500,
                                    learning_rate=1e-4, seed=random.randint(0, 100),
                                    step_length=500, episode_length=96)
        result = parameter_camp_test(parameter_test)
        camp_result = result['camp_result']
        results.append(result)
        print(json.dumps(result))
        if camp_result[0] >= imp_threshold and camp_result[1] >= click_threshold:
            break
    return results


def ddpg_test():
    imp_threshold = 50000
    click_threshold = 400
    results = []
    for i in range(3):
        # parameter_test = ['1458', 1.0 / 32] + [1e-4] + [0.0001, 2500, 500, 96]
        parameter_test = Parameters(camp_id='1458', budget_scaling=1.0 / 32, initial_Lambda=1e-4,
                                    budget_init=2500, step_length=500, episode_length=96)
        result = ddpg_parameter_test.parameter_camp_test(parameter_test)
        camp_result = result['camp_result']
        results.append(result)
        print(json.dumps(result))
        if camp_result[0] >= imp_threshold and camp_result[1] >= click_threshold:
            break
    return results


def ddpg_test_multi():
    concurrency = 20
    parameter_test = Parameters(camp_id=camp_id, budget_scaling=1.0 / 32, initial_Lambda=1e-4,
                                budget_init=2500, step_length=500, episode_length=96)
    # parameter_test = [test_camp_id, 1.0 / 32] + [1e-4] + [0.0001, 2500, 500, 96]
    with Pool(processes=concurrency) as pool:
        results = pool.map(ddpg_parameter_test.parameter_camp_test, [parameter_test for i in range(concurrency)])
        return Results(__root__=results)


if __name__ == "__main__":
    main()
