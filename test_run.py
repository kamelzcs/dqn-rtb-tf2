import ddpg_parameter_test
from parameter_test import parameter_camp_test

def main():
    # ddpg_test()
    dnq_test()


def dnq_test():
    alpha = [1e-4]
    for al in alpha:
        parameter_test = ['1458', 1.0 / 32] + [al] + [0.0001, 2500, 500, 1e-4, 1]
        result = parameter_camp_test(parameter_test)
        print(result)

def ddpg_test():
    parameter_test = ['1458', 1.0 / 32, 1e-4, 0.0001, 2500, 500, 1e-4, 1]
    result = ddpg_parameter_test.parameter_camp_test(parameter_test)
    print(result)


if __name__ == "__main__":
    main()