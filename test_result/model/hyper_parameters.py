from typing import Optional

from pydantic import BaseModel


class Parameters(BaseModel):
    camp_id: str
    budget_scaling: float
    initial_Lambda: float
    epsilon_decay_rate: Optional[float]
    budget_init: float
    step_length: int
    learning_rate: Optional[float]
    seed: Optional[int]
    episode_length: int
