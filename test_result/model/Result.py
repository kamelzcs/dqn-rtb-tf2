from typing import Optional

from pydantic import BaseModel

from test_result.model.CampResult import CampResult
from test_result.model.hyper_parameters import Parameters


class Result(BaseModel):
    camp_id: str
    parameters: Parameters
    epsilon: Optional[float]
    total_budget: float
    auctions: float
    optimal_reward: float
    camp_result: CampResult
    budget: list[float]
    lambda_value: list[float]
    unimod: Optional[list[float]]
    actions: Optional[list[float]]
    action_values: Optional[list[float]]
    action_nested_values: Optional[list[list[float]]]
    lin_bid_test: CampResult
    rand_bid_test: CampResult


class Results(BaseModel):
    __root__: list[Result]
