from datetime import timedelta

import numpy as np
import os
import pandas as pd
import pickle as pickle
import matplotlib.pyplot as plt

import utils


class RTB_environment:
    """
    This class will construct and manage the environments which the
    agent will interact with. The distinction between training and
    testing environment is primarily in the episode length.
    """
    def __init__(self, camp_dict, episode_length, step_length):
        """
        We need to initialize all of the data, which we fetch from the
        campaign-specific dictionary. We also specify the number of possible
        actions, the state, the amount of data which has been trained on,
        and so on.
        :param camp_dict: a dictionary containing data on winning bids, ctr
        estimations, clicks, budget, and so on. We copy the data on bids, ctr
        estimations and clicks; then, we delete the rest of the dictionary.
        :param episode_length: specifies the number of steps in an episode
        :param step_length: specifies the number of auctions per step.
        """
        self.camp_dict = camp_dict
        self.data_count = camp_dict['imp']
        self.split_index = len(camp_dict['split']) - 1

        self.result_dict = {'auctions':0, 'impressions':0, 'click':0, 'cost':0, 'win-rate':0, 'eCPC':0, 'eCPI':0}

        self.actions = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]

        self.step_length = step_length
        self.episode_length = episode_length

        self.Lambda = 1
        self.time_step = 0
        self.budget = 1
        self.init_budget = 1
        self.n_regulations = 0
        self.budget_consumption_rate = 0
        self.winning_rate = 0
        self.cost = 0
        self.ctr_value = 0
        self.click = 0
        self.impressions = 0
        self.termination = True

        self.state = utils.one_hot_encode(self.n_regulations - 1, self.episode_length) + [self.budget / self.init_budget, self.n_regulations,
                      self.budget_consumption_rate,
                      self.winning_rate, self.ctr_value]

    def get_camp_data_minute(self):
        start_index = self.data_count - 1
        while start_index >= 0 and\
                self.camp_dict['data'].iloc[start_index].timestamp > self.camp_dict['split'][self.split_index - 1]:
            start_index -= 1

        ctr_estimations = np.array(
            self.camp_dict['data'].iloc[start_index + 1: self.data_count, :]['pctr'])
        winning_bids = np.array(
            self.camp_dict['data'].iloc[start_index + 1: self.data_count, :]['winprice'])
        clicks = list(
            self.camp_dict['data'].iloc[start_index + 1: self.data_count, :]['click'])

        self.data_count = start_index + 1
        self.split_index -= 1
        # print(f"data_count:{self.data_count}, time:{self.camp_dict['data'].iloc[self.data_count - 1].timestamp}, start_index: {start_index}, time{self.camp_dict['data'].iloc[start_index + 1].timestamp}")

        return ctr_estimations, winning_bids, clicks

    def get_camp_data(self):
        """
        This function updates the data variables which are then accessible
        to the step-function. This function also deletes data that has already
        been used and, hence, tries to free up space.
        :return: updated data variables (i.e. bids, ctr estimations and clicks)
        """
        if self.data_count < self.step_length:
            ctr_estimations = np.array(self.camp_dict['data'].iloc[:self.data_count, :]['pctr'])
            winning_bids = np.array(self.camp_dict['data'].iloc[:self.data_count, :]['winprice'])
            clicks = list(self.camp_dict['data'].iloc[:self.data_count, :]['click'])

            self.data_count = 0
            return ctr_estimations, winning_bids, clicks
        else:
            ctr_estimations = np.array(
                self.camp_dict['data'].iloc[self.data_count - self.step_length:self.data_count, :]['pctr'])
            winning_bids = np.array(
                self.camp_dict['data'].iloc[self.data_count - self.step_length:self.data_count, :]['winprice'])
            clicks = list(
                self.camp_dict['data'].iloc[self.data_count - self.step_length:self.data_count, :]['click'])

            self.data_count -= self.step_length
            return ctr_estimations, winning_bids, clicks

    def reset(self, budget, initial_Lambda):
        """
        This function is called whenever a new episode is initiated
        and resets the budget, the Lambda, the time-step, the termination
        bool, and so on.
        :param budget: the amount of money the bidding agent can spend during
        the period
        :param initial_Lambda: the initial scaling of ctr-estimations to form bids
        :return: initial state, zero reward and a false termination bool
        """
        # self.n_regulations = min(self.episode_length, self.data_count / self.step_length)
        self.n_regulations = self.episode_length
        self.budget = budget
        self.init_budget = budget * self.episode_length / self.n_regulations
        self.Lambda = initial_Lambda
        self.time_step = 0

        ctr_estimations, winning_bids, clicks = self.get_camp_data_minute()
        bids = [int(i * (1 / self.Lambda)) for i in ctr_estimations]
        budget = self.budget
        self.budget_consumption_rate = 0
        self.winning_rate = 0
        self.ctr_value = 0
        self.click = 0
        self.impressions = 0

        for i in range(len(ctr_estimations)):
            if bids[i] > winning_bids[i] and budget > bids[i]:
                budget -= winning_bids[i]
                self.impressions += 1
                self.cost += winning_bids[i]
                self.click += clicks[i]
                self.ctr_value += ctr_estimations[i]
                self.winning_rate += 1 / len(ctr_estimations)
            else:
                continue

        self.state = utils.one_hot_encode(self.n_regulations - 1, self.episode_length) + [self.budget / self.init_budget, self.n_regulations,
                      self.budget_consumption_rate,
                      self.winning_rate, self.ctr_value]

        self.budget_consumption_rate = (self.budget - budget) / self.budget
        self.budget = budget
        self.n_regulations -= 1
        self.time_step += 1

        reward = self.ctr_value
        self.termination = False

        return self.state, reward, self.termination

    def step(self, action):
        """
        This function takes an action from the bidding agent (i.e.
        a change in the ctr-estimation scaling, and uses it to compute
        the agent's bids, s.t. it can compare it to the "winning bids".
        If one of the agent's bids exceed a winning bid, it will subtract
        the cost of the impression from the agent's budget, etc, given that
        the budget is not already depleted.
        :param action_index: an index for the list of allowed actions
        :return: a new state, reward and termination bool (if time_step = 96)
        """
        self.Lambda = self.Lambda*(1 + action)
        ctr_estimations, winning_bids, clicks = self.get_camp_data_minute()

        bids = [int(i*(1/self.Lambda)) for i in ctr_estimations]
        budget = self.budget
        self.click = 0
        self.cost = 0
        self.ctr_value = 0
        self.winning_rate = 0
        self.impressions = 0

        for i in range(len(ctr_estimations)):
            if bids[i] > winning_bids[i] and budget > bids[i]:
                budget -= winning_bids[i]
                self.impressions += 1
                self.cost += winning_bids[i]
                self.click += clicks[i]
                self.ctr_value += ctr_estimations[i]
                self.winning_rate += 1 / len(ctr_estimations)
            else:
                continue

        self.result_dict['impressions'] += self.impressions
        self.result_dict['click'] += self.click
        self.result_dict['cost'] += self.cost
        self.result_dict['win-rate'] += self.winning_rate * len(ctr_estimations) / self.camp_dict['imp']

        self.budget_consumption_rate = (self.budget - budget) / self.budget
        self.budget = budget
        self.n_regulations -= 1
        self.time_step += 1

        if self.time_step == self.episode_length or self.data_count == 0:
            self.termination = True
            print(f"time_step: {self.time_step} time: {self.camp_dict['data'].iloc[self.data_count].timestamp}")

        self.state = utils.one_hot_encode(self.n_regulations - 1, self.episode_length) + [self.budget / self.init_budget, self.n_regulations,
                      self.budget_consumption_rate,
                      self.winning_rate, self.ctr_value]

        reward = self.ctr_value

        return self.state, reward, self.termination

    def result(self):
        """
        This function returns some statistics from the episode or test
        :return: number of impressions won, number of
        actual clicks, winning rate, effective cost per click,
        and effective cost per impression.
        """
        if self.result_dict['click'] == 0:
            self.result_dict['eCPC'] = 0
        else:
            self.result_dict['eCPC'] = self.result_dict['cost'] / self.result_dict['click']
        self.result_dict['eCPI'] = self.result_dict['cost'] / self.result_dict['impressions']

        return self.result_dict['impressions'], self.result_dict['click'], self.result_dict['cost'], \
               self.result_dict['win-rate'], self.result_dict['eCPC'], self.result_dict['eCPI']


def get_split(data, episode_length):
    def internal(d):
        min_time = d.timestamp.min()
        max_time = d.timestamp.max()
        delta = timedelta(minutes=24 * 60 // episode_length)
        split = [max_time]
        t = 1
        while max_time - t * delta >= min_time:
            split.insert(0, max_time - t * delta)
            t += 1
        split.insert(0, max_time - t * delta)
        return split
    return [internal(v) for v in data]


def get_data(camp_n):

    """
    This function extracts data for certain specified campaigns
    from a folder in the current working directory.
    :param camp_n: a list of campaign names
    :return: two dictionaries, one for training and one for testing,
    with data on budget, bids, number of auctions, etc. The different
    campaigns are stored in the dictionaries with their respective names.
    """
    if type(camp_n) != str:
        train_file_dict = {}
        test_file_dict = {}
        data_path = os.path.join(os.getcwd(), 'iPinYou_data/enhanced')

        for camp in camp_n:
            test_data = pd.read_csv(f"{data_path}/test.theta_{camp}.txt",
                                     header=None, index_col=False, sep=' ',names=['click', 'winprice', 'pctr', 'timestamp'])
            test_data.timestamp = pd.to_datetime(test_data.timestamp, format='%Y%m%d%H%M%S%f')
            test_data.sort_values('timestamp', inplace=True)
            train_data = pd.read_csv(f"{data_path}/train.theta_{camp}.txt",
                                     header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr', 'timestamp'])
            train_data.timestamp = pd.to_datetime(train_data.timestamp, format='%Y%m%d%H%M%S%f')
            train_data.sort_values('timestamp', inplace=True)
            splits = get_split([train_data, test_data], episode_length)

            camp_info = pickle.load(open(f"{data_path}/info_{camp}.txt", 'rb'))
            test_budget = camp_info['cost_test']
            train_budget = camp_info['cost_train']
            test_imp = camp_info['imp_test']
            train_imp = camp_info['imp_train']

            train = {'imp':train_imp, 'budget':train_budget, 'data':train_data, 'split': splits[0]}
            test = {'imp':test_imp, 'budget':test_budget, 'data':test_data, 'split': splits[1]}

            train_file_dict[camp] = train
            test_file_dict[camp] = test
    else:
        data_path = os.path.join(os.getcwd(), 'iPinYou_data')
        test_data = pd.read_csv(f"{data_path}/test.theta_{camp_n}.txt",
                                header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
        train_data = pd.read_csv(f"{data_path}/train.theta_{camp_n}.txt",
                                 header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
        camp_info = pickle.load(open(f"{data_path}/info_{camp_n}.txt", 'rb'))
        test_budget = camp_info['cost_test']
        train_budget = camp_info['cost_train']
        test_imp = camp_info['imp_test']
        train_imp = camp_info['imp_train']

        train_file_dict = {'imp': train_imp, 'budget': train_budget, 'data': train_data}
        test_file_dict = {'imp': test_imp, 'budget': test_budget, 'data': test_data}

    return train_file_dict, test_file_dict
