
import numpy as np

from test_result.model.CampResult import CampResult


def rand_bidding_test(train_file_dict, test_file_dict, budget, distribution):
    """
    This function takes a specific iPinYou-campaign and evaluates the
    performance of a bidding algorithm using random bidding.
    """
    click = list(test_file_dict['data']['click'])
    winning_bids = list(test_file_dict['data']['winprice'])

    impressions = 0
    clicks = 0
    cost = 0
    win_rate = 0
    ecpc = 0
    ecpi = 0

    if distribution == 'normal':
        bid_average = np.array(train_file_dict['data']['winprice']).mean()
        bid_var = np.array(train_file_dict['data']['winprice']).var()
        for i in range(test_file_dict['imp']):
            bid = np.random.normal(bid_average, bid_var)
            if bid > winning_bids[i] and bid < budget:
                impressions += 1
                budget -= winning_bids[i]
                clicks += click[i]
                cost += winning_bids[i]
                win_rate += 1 / test_file_dict['imp']
            else:
                continue


    elif distribution == 'uniform':
        bid_max = np.array(train_file_dict['data']['winprice']).max()
        bid_min = np.array(train_file_dict['data']['winprice']).min()
        for i in range(test_file_dict['imp']):
            bid = np.random.uniform(bid_min, bid_max)
            if bid > winning_bids[i] and bid < budget:
                impressions += 1
                budget -= winning_bids[i]
                clicks += click[i]
                cost += winning_bids[i]
                win_rate += 1 / test_file_dict['imp']
            else:
                continue

    if clicks > 0:
        ecpc = cost / clicks
    if impressions > 0:
        ecpi = cost / impressions

    # return impressions, clicks, cost, win_rate, ecpc, ecpi
    return CampResult(imp=impressions, click=clicks, cost=cost, wr=win_rate, ecpc=ecpc, ecpi=ecpi)

