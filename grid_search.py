import numpy as np
from scipy.special import softmax
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
np.set_printoptions(suppress=True)
import numpy as np
from ast import literal_eval
from scipy.special import logit, expit
import random
import math
from utils import *
from agents import *
from optimizer import *
from scipy.optimize import fmin

class grid_search:
    def grid_search(currentConfig, ID_df, agent):
        '''
        Implements a grid search over the literal architect's parameters.
        '''
        # Define the parameters to search over
        beta_range = np.arange(0.1, 100, 0.5)

        # basically we iterate over all possible combinations of the parameters
        # and comptue the log likelihood for each combination
        # then we return the combination with the highest log likelihood
        # the grid search happens for each ID separately (ID_df is a dataframe for one ID)
        nll = 0  

        grid_dict = {}

        for b in beta_range:
            print(f"beta={b}")
            if agent == "literal_architect":
                nll = optimize.compute_ll_literal_architect(b, currentConfig, ID_df)
            # elif agent == "standard_helper":
            #     nll = compute_ll_pragmatic(currentConfig, ID_df, b, goalspace)
            # elif agent == "prob_helper":
            #     nll = compute_ll_pragmatic(currentConfig, ID_df, b, goalspace)
            else:
                print("agent not found")
            # store the negative log likelihood for each grid value and nll in a dictionary
            grid_dict[b] = nll
        
        # return the combination with the highest log likelihood
        # find min value in dictionary: min because we are dealing with negative log likelihoods
        min_value = min(grid_dict.values())
        # find key corresponding to max value
        min_key = [k for k, v in grid_dict.items() if v == min_value]
        # return the key
        return min_key[0]
    
    def optimize_IDs(moveID_df):
        '''
        '''
        optimized_agent = pd.DataFrame()    
        IDs = moveID_df.ID.unique().tolist()
        print("IDs=", IDs)
        for ID in IDs:
            print(f"optimizing for ID {ID}")
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]
            config = list(ID_df["config"])[0]

            optimal_beta = grid_search.grid_search(config, ID_df, "literal_architect")
            
            beta_df = pd.DataFrame({'ID': [ID],  'action_noise':optimal_beta})
            
            optimized_agent = pd.concat([optimized_agent, beta_df])

            # write to csv
            optimized_agent.to_csv('opt_results/grid_literalA.csv', index=False)
        
moveID_df = pd.read_csv("e1 results/final_move_df.csv", converters={"moveIDs": literal_eval, "config": literal_eval})
grid_search.optimize_IDs(moveID_df)