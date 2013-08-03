#! /usr/env/bin python
from dolo import approximate_controls
import numpy as np


def shooter(model, exog_path, T=500, tol=1e-6):
    """
    Implements the shooting algorithm for a dolo fga model.

    Notes
    =====
    The algorithm proceeds as follows (section 11.6.3 of RTM4):

    1. Solve for the terminal steady state (using exog_path[-1])
    2. Guess initial values for the control variable (c_0), use these
       and the transition equations (g) to solve for states in period 1
       (k_1).
    3. Use arbitrage equations (f) to generate next period of controls
       (c_1)
    4. TODO Finish explaining the algorithm here
    """
    s_bar = model.calibration['states'].copy()
    x_bar = model.calibration['controls'].copy()

    # Get equations
    f = model.functions['arbitrage']
    g = model.functions['transition']
    a = model.functions['auxiliary']
    # dist = 10

    # TODO implement the algorithm here.

    pass
