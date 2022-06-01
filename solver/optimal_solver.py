import random
import timeit

import pyomo.environ as pyo
from numpy.core.records import ndarray
from pyomo.core import maximize
from pyomo.kernel import pprint
import numpy as np
from scipy import optimize

from enum import Enum


class Solver(Enum):
    CBC = 1
    HIGHS = 2


# %%
def solve(ctrs: ndarray, costs: ndarray, budget: float, solver: Solver = Solver.CBC):
    if solver == Solver.CBC:
        model = pyo.ConcreteModel()
        model.x = pyo.Var(pyo.RangeSet(0, len(ctrs) - 1), domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.y = pyo.Objective(
            expr=sum([ctrs[i] * model.x[i] for i in range(len(ctrs))]),
            sense=maximize
        )
        model.Constraint1 = pyo.Constraint(
            expr=sum([costs[i] * model.x[i] for i in range(len(ctrs))]) <= budget)
        opt = pyo.SolverFactory('cbc')
        opt.solve(model)
        return model.y()
        # return model.y(), [model.x[i].value for i in model.x]
    else:
        bounds = [(0,1.0) for i in range(len(ctrs))]
        min = optimize.linprog(ctrs * -1, A_ub=np.expand_dims(costs, axis=0), b_ub=np.array([budget], ndmin=2), bounds=bounds, method='highs')
        print((min.x > 0.0).sum())
        return -min.fun


# %%
# variables = 500 * 96
# %timeit solve(np.array([random.random() for i in range(variables)]), np.array([random.random() for i in range(variables)]), 100, solver = Solver.CBC)