from parameter_test import parameter_camp_test

def main():
    parameter_test = ['1458', 1.0 / 32, 1e-4, 0.0001, 2500, 500, 0.0001, 1]
    result = parameter_camp_test(parameter_test)
    print(result)

if __name__ == "__main__":
    main()