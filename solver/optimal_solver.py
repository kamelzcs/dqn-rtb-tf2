import random

import pyomo.environ as pyo
from pyomo.core import maximize


# %%
def solve(ctrs: list[float], costs: list[float], budget: float):
    model = pyo.ConcreteModel()
    model.x = pyo.Var([i for i in range(len(ctrs))], domain=pyo.NonNegativeReals)
    model.y = pyo.Objective(
        expr=sum([ctrs[i] * model.x[i] for i in range(len(ctrs))]),
        sense=maximize
    )
    model.Constraint1 = pyo.Constraint(
        expr=sum([costs[i] * model.x[i] for i in range(len(ctrs))]) <= budget)
    opt = pyo.SolverFactory('glpk')
    opt.solve(model)
    return model.y()


# %%
# solve([random.random() for i in range(500)], [random.random() for i in range(500)], 100)
