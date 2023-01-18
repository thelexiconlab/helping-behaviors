import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# need some way of defining the observed data
# X possible moves
# 108 x 108 possible move matrix: changes with every turn
# try it out for one particular move of the architect maybe

moves_array = np.array((48, 108, 108))
# literal architect function assigns probability to each possible move
# this is the function that we want to optimize

import pymc3 as pm
import theano.tensor as T
import numpy as np

actions = T.as_tensor_variable(np.array([0, 1, 2]))  # indiv0 selects response0, indiv1 response1, etc.
with pm.Model() as model:
    p = pm.Beta('p', alpha=2, beta=2, shape=[3, 3])  # prob. for each indiv. for each response
    actions = pm.Categorical('actions', p=p, observed=actions)
    trace = pm.sample()


def optimize(y_empirical, model_to_optimize):

    # Initialize random number generator
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")

    import pymc as pm
    basic_model = pm.Model()

    with basic_model:

        # Priors for unknown model parameters: we have beta as our parameter
        beta = pm.Normal("beta", mu=0, sigma=1, shape=1)

        # Expected value of outcome
        if model_to_optimize == "nonRSA":
            # compute a "mu" for the nonRSA model
            mu = nonRSA.speaker_targetboard(boards[boardname], alpha, beta, cluelist_union, representations, 'swow', vocab, target_df)
            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal("Y_obs", mu=mu, observed=y_empirical)

    with basic_model:
        # draw 1000 posterior samples
        idata = pm.sample()

    return az.summary(idata, round_to=2)