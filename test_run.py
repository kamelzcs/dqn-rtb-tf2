import random

import ddpg_parameter_test
from parameter_test import parameter_camp_test
import json
import faulthandler
import signal


def main():
    # faulthandler.enable()
    # faulthandler.register(signal.SIGINT.value)
    with open('test_result/time_split_dnq_multi', 'w') as dnq_multi, \
            open('test_result/time_split_ddpg_multi', 'w') as ddpg_multi:
        json.dump(ddpg_test(), ddpg_multi)
        json.dump(dnq_test(), dnq_multi)


def dnq_test():
    imp_threshold = 50000
    click_threshold = 400
    results = []
    for i in range(1):
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
    for i in range(1):
        parameter_test = ['1458', 1.0 / 32] + [1e-4] + [0.0001, 2500, 500, 96]
        result = ddpg_parameter_test.parameter_camp_test(parameter_test)
        camp_result = result['camp_result']
        results.append(result)
        print(json.dumps(result))
        if camp_result[0] >= imp_threshold and camp_result[1] >= click_threshold:
            break
    return results


if __name__ == "__main__":
    main()
