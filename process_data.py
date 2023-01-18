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

class data_processing:

    def get_moveIDs(data):
        move_df = pd.read_csv("e1 results/final_move_df.csv")
        #move_df = pd.DataFrame()
        for i in range(0, len(data)):
            d = pd.read_csv(data[i])
            ID =d["helperRandCode"][0]
            # check if ID already exists

            if ID in move_df.ID.unique().tolist():
                print(f"{ID} already in data, i = {i}/{len(data)}")
            else:
                print(f"{ID} not in data yet, i = {i}/{len(data)}")
                if(len(d.goalnumber.unique()) == 11): 
                    config = d["config"][0].split(',')
                for g in range(1, 11):
                #for g in range(1, 2):        
                    goal_data = d[d["goalnumber"]==g].reset_index()
                    if(len(goal_data)>0):
                        ID = str(goal_data["helperRandCode"][0])
                        goal = goal_data["goal"][0]
                        goal_optimal = optimal_moves.select_optimal_moves(goal, config)
                        a_moves = list(goal_data["architectMove"])
                        a_moves = list(filter(('999').__ne__, a_moves))
                        h_moves = list(goal_data["helperMove"])
                        h_moves = list(filter(('999').__ne__, h_moves))
                        h_moves = list(filter(('done').__ne__, h_moves))

                        move_seq = [x for x in itertools.chain.from_iterable(itertools.zip_longest(a_moves,h_moves)) if x]

                        goal_type = goal.split(" ")[0]
                        moveIDs = [general.move_description_to_num(m) if m != "pass" else m for m in move_seq ]

                        moveID_df = pd.DataFrame({'ID': [ID]})
                        moveID_df["config"]= str(config)
                        moveID_df["goal"]= goal
                        moveID_df["goal_optimal"]= goal_optimal
                        moveID_df["goal_type"]= goal_type
                        moveID_df["total_moves"]= len(moveIDs)
                        moveID_df["moveIDs"]= str(moveIDs)

                        move_df = pd.concat([move_df, moveID_df])
                    else:
                        print(f"goal_data is empty ID ={ID} and i = {i}/{len(data)}")
                else:
                    print(f"10 goals not completed. ID ={ID} and i = {i}/{len(data)}")
            
        return move_df

    def individual_move_utilities(moveID_df):

        individual_move_tracker = pd.DataFrame()
        final_individual_move_tracker = pd.read_csv("e1 results/final_tracker.csv")

        for index, row in moveID_df.iterrows():
            moveIDs = list(row["moveIDs"])
            goal = row["goal"]
            ID = row["ID"]
            
            if ID in final_individual_move_tracker.ID.unique().tolist():
                flag = 1
                #print(f"{ID} already in data")
            else:
                #print(f"{ID} not in data yet")

                goal_number = (index % 10)+1
                currentConfig = row["config"]
                print(f"for new ID = {ID}, goal {goal_number}")
                for k in range(0, len(moveIDs)):
                    m = moveIDs[k]
                    if k %2 == 0: # architect move
                        utility = general.compute_move_utility(currentConfig, m, goal)
                    else: # helper move
                        prev_move = moveIDs[k-1] # previous architect move
                        if m == "pass":
                            utility = 0
                            currentConfig = general.update_config(currentConfig, prev_move[0], prev_move[1])
                        else:
                            currentConfig = general.update_config(currentConfig, prev_move[0], prev_move[1])
                        
                        utility = general.compute_move_utility(currentConfig, m, goal)
                        currentConfig = general.update_config(currentConfig, m[0], m[1])
                                
                    # create tracker df and append to df 
                    id_df = pd.DataFrame({'ID': [ID]})
                    id_df["goal"] = goal
                    id_df["goal_number"] = goal_number
                    id_df["goal_type"] = row["goal_type"]
                    id_df["optimal_moves"] = row["goal_optimal"]
                    id_df["move_id"] = k
                    id_df["move"] = str(m)
                    id_df["role"] = "architect" if k%2 == 0 else "helper"
                    id_df["move_utility"] = utility
                    
                    individual_move_tracker = pd.concat([individual_move_tracker, id_df])
            

            final_individual_move_tracker = pd.concat([final_individual_move_tracker, individual_move_tracker])

        return final_individual_move_tracker
    
    def first_architect_move(moveID_df):
        goalspace = general.define_goalspace()
        first_move_rank_df = pd.DataFrame()

        final_df = pd.read_csv("e1 results/first_moves.csv")
        
        for index, row in moveID_df.iterrows():
            moveIDs = row["moveIDs"]
            goal = row["goal"]
            ID = row["ID"]

            if ID in final_df.ID.unique().tolist():
                flag = 1
                #print(f"{ID} already in data")
            else:
                #print(f"{ID} not in data yet")

                goal_number = (index % 10)+1
                config = row["config"]
                print(f"for new ID = {ID}, goal {goal_number}")
                first_move = moveIDs[0]
                    
                # now find all possible moves and what their utility is
                u = general.generate_utility_matrix(config, goalspace)

                move_df = pd.DataFrame()
                
                for m in u[1]:        
                    m_index = u[1].index(m)
                    num_possible_goals = sum(1 for x in u[0][m_index] if x == 1)
                    
                    # get all indices where u[0][m_index] == 1
                    serves_current_goal = [i for i, x in enumerate(u[0][m_index]) if x == 1]
                    # get goals from goalspace at indices in serves_current_goal
                    serves_current_goal_goals = [goalspace[i] for i in serves_current_goal]

                    move_id_df = pd.DataFrame({"ID": [ID], "goal":[goal]})
                    move_id_df["move"] = [m]
                    move_id_df["num_possible_goals"] = [num_possible_goals]
                    move_id_df["goals_served"] = [serves_current_goal_goals]
                    # does this move serve the current goal?
                    m_serves_current_goal = u[0][m_index][goalspace.index(goal)]
                    move_id_df["serves_current_goal"] = [m_serves_current_goal]
                    # pd.concat
                    move_df = pd.concat([move_df, move_id_df])
            
            
            
                # sort move_df
                move_df.sort_values(by=['serves_current_goal', 'num_possible_goals' ], ascending=[False, True], inplace=True)
                # index move_df
                move_df.reset_index(drop=True, inplace=True)
                # find rank of first move
                first_move_rank = move_df.loc[move_df['move'] == first_move].index[0]
                
                # find possible_goals for first move
                first_move_possible_goals = move_df.loc[move_df['move'] == first_move]['num_possible_goals'].values[0]
                first_move_possible_goal_goals = move_df.loc[move_df['move'] == first_move]['goals_served'].values[0]
                first_move_serves_goal = move_df.loc[move_df['move'] == first_move]['serves_current_goal'].values[0]
                
                # total moves that serve the current goal
                total_moves_serves_goal = len(move_df[move_df['serves_current_goal'] == 1])
                # find all rows up to first_move_rank
                before_first_move = move_df.loc[move_df.index < first_move_rank].move.values
                # find all rows with same num_possiible_goals as first_move and  serves_current_goal = 1
                same_num_possible_goals = move_df.loc[(move_df['num_possible_goals'] == first_move_possible_goals) & (move_df['serves_current_goal'] == 1)].move.values
                # take 1000 random samples from the total moves that serve the current goal
                # and find the rank of the each move in move_df
                moves_that_serves_goal = move_df[move_df['serves_current_goal'] == 1].move.values
                # get 1000 random samples from moves_that_serves_goal
                random_samples = np.random.choice(moves_that_serves_goal, 1000, replace=True)
                # find the rank of each random sample in move_df
                random_samples_rank = [move_df.loc[move_df['move'] == m].index[0] for m in random_samples]
                # find mean rank of random samples
                mean_random_samples_rank = np.mean(random_samples_rank)
                # add to first_move_rank_df
                first_id  = pd.DataFrame({"ID": [ID], "goal":[goal], "first_move": [first_move], "first_move_rank":[first_move_rank], "first_move_possible_goals":[first_move_possible_goals], "before_first_move":[before_first_move], "first_move_serves_goal":[first_move_serves_goal], "total_moves_serves_goal":[total_moves_serves_goal], "mean_random_samples_rank":[mean_random_samples_rank], "first_move_possible_goal_goals":[first_move_possible_goal_goals]})
                first_move_rank_df = pd.concat([first_move_rank_df, first_id])

            final_df = pd.concat([final_df, first_move_rank_df])
            
            return final_df


    #goalspace = general.define_goalspace()
    #helper_optimized = pd.read_csv('e1 results/helper_optimized.csv')

    # moveIDs code
    # data = glob.glob('e1 data/*.csv')
    # move_df = data_processing.get_moveIDs(data)
    # print(f"total N=", len(move_df.ID.unique())) # 79 completed
    # move_df.to_csv("e1 results/final_move_df.csv", index=False)

    # individual move tracker code
    # moveID_df = pd.read_csv("e1 results/final_move_df.csv", converters={"moveIDs": literal_eval, "config": literal_eval})
    # individual_move_tracker = data_processing.individual_move_utilities(moveID_df)

    # individual_move_tracker.to_csv('e1 results/final_tracker.csv', index=False)

    # first move analysis code

    # moveID_df = pd.read_csv("e1 results/final_move_df.csv", converters={"moveIDs": literal_eval, "config": literal_eval})
    # first_moves = data_processing.first_architect_move(moveID_df)

    # first_moves.to_csv('e1 results/first_moves.csv', index=False)




    