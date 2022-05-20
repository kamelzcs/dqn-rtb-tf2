import random

import pyomo.environ as pyo
from pyomo.core import maximize
from pyomo.kernel import pprint


# %%
def solve(ctrs: list[float], costs: list[float], budget: float):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(pyo.RangeSet(0, len(ctrs) - 1), domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.y = pyo.Objective(
        expr=sum([ctrs[i] * model.x[i] for i in range(len(ctrs))]),
        sense=maximize
    )
    model.Constraint1 = pyo.Constraint(
        expr=sum([costs[i] * model.x[i] for i in range(len(ctrs))]) <= budget)
    opt = pyo.SolverFactory('glpk')
    opt.solve(model)
    model.x.display()
    model.y.display()
    return model.y(), [model.x[i].value for i in model.x]


# %%
solve([random.random() for i in range(500)], [random.random() for i in range(500)], 100)
