import random

import ddpg_parameter_test
from parameter_test import parameter_camp_test
import json
import faulthandler
import signal
import multiprocessing
from multiprocessing import Pool


def main():
    # faulthandler.enable()
    # faulthandler.register(signal.SIGINT.value)
    # with open('test_result/time_split_dnq_multi_optimal', 'w') as dnq_multi, \
    #         open('test_result/time_split_ddpg_multi_optimal', 'w') as ddpg_multi:
    #     json.dump(ddpg_test_multi(), ddpg_multi, indent=2)
    #     json.dump(dnq_test_multi(), dnq_multi, indent=2)
    with open('test_result/time_split_ddpg_multi_optimal', 'w') as ddpg_multi:
        json.dump(ddpg_test_multi(), ddpg_multi, indent=2)

def dnq_test_multi():
    concurrency = 20
    parameter_test = ['1458', 1.0 / 32] + [1e-4] + [0.0001, 2500, 500, 1e-4, random.randint(0, 100), 96]
    with Pool(processes=concurrency) as pool:
        results = pool.map(parameter_camp_test, [parameter_test for _ in range(concurrency)])
        return results

def dnq_test():
    imp_threshold = 50000
    click_threshold = 400
    results = []
    for i in range(3):
        parameter_test = ['1458', 1.0 / 32] + [1e-4] + [0.0001, 2500, 500, 1e-4, random.randint(0, 100), 96]
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
        parameter_test = ['1458', 1.0 / 32] + [1e-4] + [0.0001, 2500, 500, 96]
        result = ddpg_parameter_test.parameter_camp_test(parameter_test)
        camp_result = result['camp_result']
        results.append(result)
        print(json.dumps(result))
        if camp_result[0] >= imp_threshold and camp_result[1] >= click_threshold:
            break
    return results

def ddpg_test_multi():
    concurrency = 20
    parameter_test = ['1458', 1.0 / 32] + [1e-4] + [0.0001, 2500, 500, 96]
    with Pool(processes=concurrency) as pool:
        results = pool.map(ddpg_parameter_test.parameter_camp_test, [parameter_test for i in range(concurrency)])
        return results



if __name__ == "__main__":
    main()
