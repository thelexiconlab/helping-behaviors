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
from scipy.optimize import fmin


class optimize:
    def compute_ll_helper(beta, currentConfig, ID_df):
        '''
        ID_df contains the 10 rows for 10 goals completed by ID
        '''
        initial_config = currentConfig.copy()

        goalspace = general.define_goalspace()
        nll = 0    

        for index, row in ID_df.iterrows():
            goal = row["goal"]
            #print(f"for goal {goal}")
            moveIDs = row["moveIDs"]
            goal_np_initial = general.get_initial_goal_probs(goalspace)
            goal_np = goal_np_initial.copy()
            currentConfig = initial_config.copy() # reset config for every goal

            for i in range(0, len(moveIDs)):
                move = moveIDs[i]
                if i % 2 != 0:
                    # helper move
                    prev_move = moveIDs[i-1] # previous architect move
                
                    softmax_moves, move_labels, goal_np = helper.probabilistic_helper(currentConfig, prev_move, goal_np, goalspace, goal_noise = beta[0], pass_noise = beta[1], action_noise = beta[2])
                    prob = softmax_moves[move_labels.index(("none","none"))] if move == "pass" else softmax_moves[move_labels.index(move)]
                    # now we add the probability to the nll
                    nll += -np.log(prob)
                    # update based on architect move    
                    currentConfig = general.update_config(currentConfig, prev_move[0], prev_move[1])
                    # update based on helper move
                    if move == "pass":
                        currentConfig = general.update_config(currentConfig, "none", "none")
                    else:
                        currentConfig = general.update_config(currentConfig, move[0], move[1])
        #print(f"beta = {beta}, nll=", nll)
        return nll
    
    def compute_ll_helper_standard(beta, currentConfig, ID_df):
        '''
        ID_df contains the 10 rows for 10 goals completed by ID
        '''
        initial_config = currentConfig.copy()

        goalspace = general.define_goalspace()
        nll = 0    

        for index, row in ID_df.iterrows():
            goal = row["goal"]
            #print(f"for goal {goal}")
            moveIDs = row["moveIDs"]
            goal_np_initial = general.get_initial_goal_probs(goalspace)
            goal_np = goal_np_initial.copy()
            currentConfig = initial_config.copy() # reset config for every goal

            for i in range(0, len(moveIDs)):
                move = moveIDs[i]
                if i % 2 != 0:
                    # helper move
                    prev_move = moveIDs[i-1] # previous architect move
                
                    softmax_moves, move_labels, goal_np = helper.standard_helper(currentConfig, prev_move, goal_np, goalspace, goal_noise = beta[0], action_noise = beta[1])
                    prob = softmax_moves[move_labels.index(("none","none"))] if move == "pass" else softmax_moves[move_labels.index(move)]
                    # now we add the probability to the nll
                    nll += -np.log(prob)
                    # update based on architect move    
                    currentConfig = general.update_config(currentConfig, prev_move[0], prev_move[1])
                    # update based on helper move
                    if move == "pass":
                        currentConfig = general.update_config(currentConfig, "none", "none")
                    else:
                        currentConfig = general.update_config(currentConfig, move[0], move[1])
        #print(f"beta = {beta}, nll=", nll)
        return nll
        
    def optimize_helper(moveID_df):
        '''
        moveID_df contains 10 rows per participant for the 10  goals they completed
        fmin optimization needs to happen with all goals combined
        '''
        helper_optimized = pd.read_csv("e1 results/probhelper_optimized_standard.csv")
        #helper_optimized = pd.DataFrame()
        IDs = moveID_df.ID.unique().tolist()
        print("IDs=", IDs)
        for ID in IDs[1:]:
            print(f"optimizing helper for ID {ID}")
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]
            config = list(ID_df["config"])[0]
            
            beta_initial = [np.random.rand(), np.random.rand()]
        
            beta_opt = fmin(optimize.compute_ll_helper_standard, beta_initial, args=(config, ID_df), ftol = 0.001, full_output=True, disp=False)
            beta_df = pd.DataFrame({'ID': [ID], 'goal_noise': beta_opt[0][0], 'action_noise':beta_opt[0][1]})
            
            helper_optimized = pd.concat([helper_optimized, beta_df])

            # write to csv
            helper_optimized.to_csv('e1 results/probhelper_optimized_standard.csv', index=False)

    def compute_fitted_LL_helper(moveID_df):
        '''
        moveID_df contains 10 rows per participant for the 10  goals they completed
        '''
        NLL_optimized = pd.read_csv("opt_results/helperNLLs.csv")
        #NLL_optimized = pd.DataFrame()
        IDs = moveID_df.ID.unique().tolist()
        print("IDs=", IDs)

        std_beta = pd.read_csv("opt_results/helper_std_opt.csv")
        prob_beta = pd.read_csv("opt_results/helper_prob_opt.csv")

        for ID in IDs[17:]:
            print(f"optimizing helper for ID {ID}")
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]

            std_beta_id = std_beta.loc[std_beta['ID'] == ID].values.flatten().tolist()[1:]
            print("std_beta_id=",std_beta_id)

            prob_beta_id = prob_beta.loc[prob_beta['ID'] == ID].values.flatten().tolist()[1:]
            print("prob_beta_id=",prob_beta_id)

            config = list(ID_df["config"])[0]

            std_NLL = optimize.compute_ll_helper_standard(std_beta_id, config, ID_df)
            prob_NLL = optimize.compute_ll_helper(prob_beta_id, config, ID_df)
            
            NLL_df = pd.DataFrame({'ID': [ID], 'std_NLL': std_NLL, 'prob_NLL':prob_NLL})                    
            
            NLL_optimized = pd.concat([NLL_optimized, NLL_df])

            # write to csv
            NLL_optimized.to_csv('opt_results/helperNLLs.csv', index=False)
    
    def compute_fitted_second_move(moveID_df):
        '''
        computes likelihood of first move made by helper (second move in game)
        '''

        secondmoves_optimized = pd.read_csv("opt_results/second_optimized_probs.csv")
        already_optimized_IDs = secondmoves_optimized.ID.unique().tolist()
        
        betas = pd.read_csv("opt_results/helper_secondmove_optimized.csv")
        
        beta_IDs = betas.ID.unique().tolist()

        for ID in beta_IDs:
            if ID not in already_optimized_IDs:
                print(f"optimizing helper for ID {ID}")
                ID_df = moveID_df.loc[moveID_df['ID'] == ID]

                std_beta_id = betas.loc[betas['ID'] == ID].values.flatten().tolist()[1:3]
                #print("std_beta_id=",std_beta_id)
                prob_beta_id = betas.loc[betas['ID'] == ID].values.flatten().tolist()[3:]
                #print("prob_beta_id=",prob_beta_id)

                config = list(ID_df["config"])[0]
                
                ## second move only

                helper_df = moveID_df.loc[moveID_df['ID'] == ID]
                for index, row in helper_df.iterrows():
                    goalspace = general.define_goalspace()
                    goal_np = general.get_initial_goal_probs(goalspace)
                    goal = row["goal"]
                    goal_type = goal.split(" ")[0]
                    prev_move = list(row["moveIDs"])[0]
                    first_helper_move = list(row["moveIDs"])[1]
                    

                    utility = 0 if first_helper_move == "pass" else general.compute_move_utility(config, first_helper_move, goal)

                    std_softmax_moves, std_move_labels, std_goal_np = helper.standard_helper(config, prev_move, goal_np, goalspace, goal_noise = std_beta_id[0], action_noise = std_beta_id[1])
                    std_prob = std_softmax_moves[std_move_labels.index(("none","none"))] if first_helper_move == "pass" else std_softmax_moves[std_move_labels.index(first_helper_move)]
                    

                    prob_softmax_moves, prob_move_labels, prob_goal_np = helper.probabilistic_helper(config, prev_move, goal_np, goalspace, goal_noise = prob_beta_id[0], pass_noise = prob_beta_id[1], action_noise = prob_beta_id[2])
                    prob_prob = prob_softmax_moves[prob_move_labels.index(("none","none"))] if first_helper_move == "pass" else prob_softmax_moves[prob_move_labels.index(first_helper_move)]
                    
                    len_std = len(std_softmax_moves)
                    len_prob = len(prob_softmax_moves)
            
                    std_better_than = sum([std_prob > i for i in std_softmax_moves])
                    prob_better_than = sum([prob_prob > i for i in prob_softmax_moves])

                    prob_df = pd.DataFrame({'ID': [ID], 'goal': [goal], 'goal_type': [goal_type],'first_helper_move': [first_helper_move], 'utility':[utility], 'std_prob':[std_prob], 'prob_prob':[prob_prob], 'std_better_than':[std_better_than], 'prob_better_than': [prob_better_than],'len_std':[len_std], 'len_prob':[len_prob]})                    

                    
                    secondmoves_optimized = pd.concat([secondmoves_optimized, prob_df])

                # write to csv
                secondmoves_optimized.to_csv('opt_results/second_optimized_probs.csv', index=False)

    
    def compute_fitted_first_move(first_move_df, moveID_df):
        '''
        takes the optimal parameters from first moves and then computes probabilities for each of the first moves
        '''

        #firstmoves_optimized = pd.DataFrame()
        firstmoves_optimized = pd.read_csv("opt_results/first_optimized_probs.csv")
        already_optimized_IDs = firstmoves_optimized.ID.unique().tolist()
        
        betas = pd.read_csv("opt_results/first_arch_optimized.csv")
        beta_IDs = betas.ID.unique().tolist()

        for ID in beta_IDs:
            if ID not in already_optimized_IDs:
                print(f"optimizing architect for ID {ID}")
                ID_df = moveID_df.loc[moveID_df['ID'] == ID]

                std_beta_id = betas.loc[betas['ID'] == ID].values.flatten().tolist()[1:2]
                prob_beta_id = betas.loc[betas['ID'] == ID].values.flatten().tolist()[2:]
                list_of_betas = std_beta_id + prob_beta_id

                config = list(ID_df["config"])[0]
                goalspace = general.define_goalspace()
                goal_np = general.get_initial_goal_probs(goalspace)
                
                ## first move only

                first_ID_df = first_move_df.loc[first_move_df['ID'] == ID]
                for index, row in first_ID_df.iterrows():
                    goal = row["goal"]
                    goal_type = goal.split(" ")[0]
                    first_move = row["first_move"]
                    utility = general.compute_move_utility(config, first_move, goal)
                    
                    literal_m, literal_c = architect.literal_architect_trial(config, goal, literalA_beta= list_of_betas[0], goalspace = goalspace)
                    prag_m, prag_c = architect.pragmatic_architect_trial(config,goal, goal_np, goal_noise = list_of_betas[1], action_noise= list_of_betas[2])

                    literal_prob = literal_m[literal_c.index((first_move))]
                    prag_prob = prag_m[prag_c.index((first_move))]
                    
                    ## problem is that prob is overall higher in literal model so it causes confusion
                    ## it ranks all useful moves as equal, i.e., high probability
                    # maybe if we looked at how many moves is this move better than or has higher prob of?
                    
                    # import scipy.stats as ss

                    # literal_ranks = ss.rankdata(literal_m)
                    # prag_ranks = ss.rankdata(prag_m)

                    len_literal = len(literal_m)
                    len_prag = len(prag_m)
                
                    # move_rank_literal = literal_ranks[prag_c.index((first_move))]
                    # move_rank_pragmatic = prag_ranks[prag_c.index((first_move))]

                    literal_better_than = sum([literal_prob > i for i in literal_m])
                    prag_better_than = sum([prag_prob > i for i in prag_m])

                    prob_df = pd.DataFrame({'ID': [ID], 'goal': [goal], 'goal_type': [goal_type],'first_move': [first_move], 'utility':[utility], 'literal_prob':[literal_prob], 'prag_prob':[prag_prob], 'literal_better_than':[literal_better_than], 'prag_better_than': [prag_better_than],'len_literal':[len_literal], 'len_prag':[len_prag]})                    

                    
                    firstmoves_optimized = pd.concat([firstmoves_optimized, prob_df])

                # write to csv
                firstmoves_optimized.to_csv('opt_results/first_optimized_probs.csv', index=False)


    def compute_fitted_LL_architect(moveID_df):
        '''
        moveID_df contains 10 rows per participant for the 10  goals they completed
        '''
        # NLL_optimized = pd.read_csv("opt_results/architectNLLs.csv")
        # already_optimized_IDs = NLL_optimized.ID.unique().tolist()
        NLL_optimized = pd.DataFrame()
        
        betas = pd.read_csv("opt_results/useful_first_arch_optimized.csv")
        beta_IDs = betas.ID.unique().tolist()

        for ID in beta_IDs:
            #if ID not in already_optimized_IDs:
            print(f"optimizing architect for ID {ID}")
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]

            std_beta_id = betas.loc[betas['ID'] == ID].values.flatten().tolist()[1:2]
            #print("literal_beta_id=",std_beta_id)

            prob_beta_id = betas.loc[betas['ID'] == ID].values.flatten().tolist()[2:]
            #print("pragmatic_beta_id=",prob_beta_id)

            config = list(ID_df["config"])[0]

            list_of_betas = std_beta_id + prob_beta_id
            print("list_of_betas=",list_of_betas)

            ## first move only

            literal_NLL, prag_NLL = optimize.compute_firstmove_LL(list_of_betas, config, ID_df)

            ## all moves

            # std_NLL = optimize.compute_ll_literal_architect(std_beta_id, config, ID_df)
            # prob_NLL = optimize.compute_ll_pragmatic_architect(prob_beta_id, config, ID_df)
            
            NLL_df = pd.DataFrame({'ID': [ID], 'literal_NLL': literal_NLL, 'prag_NLL':prag_NLL})                    
            
            NLL_optimized = pd.concat([NLL_optimized, NLL_df])

            # write to csv
            NLL_optimized.to_csv('opt_results/useful_firstOPT_architectNLLs.csv', index=False)

    def compute_firstmove_LL(beta, currentConfig, ID_df):
        '''
        takes the optimal parameters from the full game but then looks at LL for only first move
        '''

        initial_config = currentConfig.copy()

        goalspace = general.define_goalspace()
        nll_literal = 0   
        nll_pragmatic = 0 
        

        for index, row in ID_df.iterrows():
            goal = row["goal"]
            #print(f"for goal {goal}")
            moveIDs = row["moveIDs"]
        
            goal_np_initial = general.get_initial_goal_probs(goalspace)
            goal_np = goal_np_initial.copy()
            currentConfig = initial_config.copy() # reset config for every goal
            
            first_move = moveIDs[0]

            utility = general.compute_move_utility(currentConfig, first_move,goal)

            if(utility == 1):
                ## get literal probs

                move_probs, c  = architect.literal_architect_trial(currentConfig, goal, literalA_beta= beta[0], goalspace = goalspace)
                prob = move_probs[c.index(first_move)]
                # now we add the probability to the nll
                nll_literal += -np.log(prob)
                
                ## get pragmatic probs

                move_probs, c  = architect.pragmatic_architect_trial(currentConfig, goal, goal_np, goal_noise = beta[1], action_noise = beta[2])
                prob = move_probs[c.index(first_move)]
                # now we add the probability to the nll
                nll_pragmatic += -np.log(prob)
       
        return nll_literal, nll_pragmatic
    
    def compute_ll_pragmatic_architect(beta, currentConfig, ID_df):
        '''
        this function finds the optimal action noise for the pragmatic architect, using already fitted parameters for the helper
        '''
        initial_config = currentConfig.copy()

        goalspace = general.define_goalspace()
        nll = 0    

        for index, row in ID_df.iterrows():
            goal = row["goal"]
            #print(f"for goal {goal}")
            moveIDs = row["moveIDs"]
        
            goal_np_initial = general.get_initial_goal_probs(goalspace)
            goal_np = goal_np_initial.copy()
            currentConfig = initial_config.copy() # reset config for every goal

            #print("moveIDs=", moveIDs)

            for i in range(0, len(moveIDs)):
                move = moveIDs[i]
                #print("move=", move)
                if i % 2 == 0:
                    # architect move
                    
                    move_probs, c  = architect.pragmatic_architect_trial(currentConfig, goal, goal_np, goal_noise = beta[0], action_noise = beta[1])
                    prob = move_probs[c.index(move)]
                    # now we add the probability to the nll
                    nll += -np.log(prob)
                else:
                    # helper move
                    prev_move = moveIDs[i-1] # previous architect move
                    
                    # NEED TO UPDATE goal_np HERE!!! otherwise optimization will not happen correctly
                    ## not sure if goal_noise should be the beta[0] or not but I guess we can optimize it accordingly

                    goal_np = helper.probabilistic_goal_inference(currentConfig, prev_move, goal_np, goalspace, goal_noise = beta[0])
                
                    # update based on architect move
                    currentConfig = general.update_config(currentConfig, prev_move[0], prev_move[1])
                    # update based on helper move 
                    if move == "pass":
                        currentConfig = general.update_config(currentConfig, "none", "none")
                    else:
                        currentConfig = general.update_config(currentConfig, move[0], move[1]) 
                
                    

        #print(f"beta = {beta}, nll=", nll)
        return nll

    def compute_ll_literal_architect(beta, currentConfig, ID_df):
        '''
        this function finds the optimal action noise for the literal architect
        '''
        initial_config = currentConfig.copy()

        goalspace = general.define_goalspace()
        nll = 0    

        for index, row in ID_df.iterrows():
            goal = row["goal"]
            #print(f"for goal {goal}")
            moveIDs = row["moveIDs"]
        
            goal_np_initial = general.get_initial_goal_probs(goalspace)
            goal_np = goal_np_initial.copy()
            currentConfig = initial_config.copy() # reset config for every goal

            #print("moveIDs=", moveIDs)

            for i in range(0, len(moveIDs)):
                move = moveIDs[i]
                #print("move=", move)
                if i % 2 == 0:
                    # architect move
                    move_probs, c  = architect.literal_architect_trial(currentConfig, goal, literalA_beta= beta, goalspace = goalspace)
                    prob = move_probs[c.index(move)]
                    # now we add the probability to the nll
                    nll += -np.log(prob)
                else:
                    # helper move
                    prev_move = moveIDs[i-1] # previous architect move
                    # update based on architect move
                    currentConfig = general.update_config(currentConfig, prev_move[0], prev_move[1])
                    # update based on helper move 
                    if move == "pass":
                        currentConfig = general.update_config(currentConfig, "none", "none")
                    else:
                        currentConfig = general.update_config(currentConfig, move[0], move[1]) 
                
                    

        #print(f"beta = {beta}, nll=", nll)
        return nll
    
    def compute_firstmove_literal_LL(beta, currentConfig, ID_df):
        '''
        takes the optimal parameters from the full game but then looks at LL for only first move
        '''

        initial_config = currentConfig.copy()

        goalspace = general.define_goalspace()
        nll_literal = 0   
        
        for index, row in ID_df.iterrows():
            goal = row["goal"]
            #print(f"for goal {goal}")
            moveIDs = row["moveIDs"]
        
            goal_np_initial = general.get_initial_goal_probs(goalspace)
            goal_np = goal_np_initial.copy()
            currentConfig = initial_config.copy() # reset config for every goal
            
            first_move = moveIDs[0]

            utility = general.compute_move_utility(currentConfig, first_move,goal)

            if(utility == 1):
                ## get literal probs
                move_probs, c  = architect.literal_architect_trial(currentConfig, goal, literalA_beta= beta[0], goalspace = goalspace)
                prob = move_probs[c.index(first_move)]
                # now we add the probability to the nll
                nll_literal += -np.log(prob)
            
       
        return nll_literal

    def compute_firstmove_pragmatic_LL(beta, currentConfig, ID_df):
        '''
        takes the optimal parameters from the full game but then looks at LL for only first move
        '''

        initial_config = currentConfig.copy()

        goalspace = general.define_goalspace()
        nll_pragmatic = 0   
        
        for index, row in ID_df.iterrows():
            goal = row["goal"]
            #print(f"for goal {goal}")
            moveIDs = row["moveIDs"]
        
            goal_np_initial = general.get_initial_goal_probs(goalspace)
            goal_np = goal_np_initial.copy()
            currentConfig = initial_config.copy() # reset config for every goal
            
            first_move = moveIDs[0]
            ## could get utility of this move and only do NLL if it is a useful move?? seems weird to subset like this tho

            #utility = general.compute_move_utility(currentConfig, first_move,goal)

            #if(utility == 1):
            ## get pragmatic probs
            move_probs, c  = architect.pragmatic_architect_trial(currentConfig, goal, goal_np, goal_noise = beta[0], action_noise = beta[1])
            prob = move_probs[c.index(first_move)]
            # now we add the probability to the nll
            nll_pragmatic += -np.log(prob)
            
       
        return nll_pragmatic
    
    def compute_secondmove_std_LL(beta, currentConfig, ID_df):
        '''
        computes optimal parameters for second move based on standard helper
        '''

        initial_config = currentConfig.copy()

        goalspace = general.define_goalspace()
        nll_standard = 0   
        
        for index, row in ID_df.iterrows():
            goal = row["goal"]
            #print(f"for goal {goal}")
            moveIDs = row["moveIDs"]
        
            goal_np_initial = general.get_initial_goal_probs(goalspace)
            goal_np = goal_np_initial.copy()
            currentConfig = initial_config.copy() # reset config for every goal
            
            prev_move = moveIDs[0]
            first_helper_move = moveIDs[1]
            
            std_softmax_moves, std_move_labels, std_goal_np = helper.standard_helper(currentConfig, prev_move, goal_np, goalspace, goal_noise = beta[0], action_noise = beta[1])
            std_prob = std_softmax_moves[std_move_labels.index(("none","none"))] if first_helper_move == "pass" else std_softmax_moves[std_move_labels.index(first_helper_move)]
            
            nll_standard += -np.log(std_prob)
            
       
        return nll_standard
    
    def compute_secondmove_prob_LL(beta, currentConfig, ID_df):
        '''
        computes optimal parameters for second move based on probabilistic helper
        '''

        initial_config = currentConfig.copy()

        goalspace = general.define_goalspace()
        nll_prob = 0   
        
        for index, row in ID_df.iterrows():
            goal = row["goal"]
            #print(f"for goal {goal}")
            moveIDs = row["moveIDs"]
        
            goal_np_initial = general.get_initial_goal_probs(goalspace)
            goal_np = goal_np_initial.copy()
            currentConfig = initial_config.copy() # reset config for every goal
            
            prev_move = moveIDs[0]
            first_helper_move = moveIDs[1]

            prob_softmax_moves, prob_move_labels, prob_goal_np = helper.probabilistic_helper(currentConfig, prev_move, goal_np, goalspace, goal_noise = beta[0], pass_noise = beta[1], action_noise = beta[2])
            prob_prob = prob_softmax_moves[prob_move_labels.index(("none","none"))] if first_helper_move == "pass" else prob_softmax_moves[prob_move_labels.index(first_helper_move)]
            
            nll_prob += -np.log(prob_prob)
            
       
        return nll_prob
    
    def optimize_firstmove_pragmatic(moveID_df):
        '''
        '''
        #architect_optimized = pd.read_csv("opt_results/arch_first_prag.csv")    
        architect_optimized = pd.DataFrame()    
        IDs = moveID_df.ID.unique().tolist()
        print("IDs=", IDs)
        for ID in IDs[26:]:
            print(f"optimizing architect for ID {ID}")
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]
            config = list(ID_df["config"])[0]
            
            beta_initial = [np.random.rand(), np.random.rand()]
        
            beta_opt = fmin(optimize.compute_firstmove_pragmatic_LL, beta_initial, args=(config, ID_df), ftol = 0.001, full_output=True, disp=False)
            beta_df = pd.DataFrame({'ID': [ID], 'goal_noise': beta_opt[0][0], 'action_noise':beta_opt[0][1]})
            
            architect_optimized = pd.concat([architect_optimized, beta_df])

            # write to csv
            architect_optimized.to_csv('opt_results/arch_first_prag.csv', index=False)
    
    def optimize_secondmove(moveID_df):
        '''
        '''
        helper_optimized = pd.read_csv("opt_results/helper_first_std.csv")    
        #helper_optimized = pd.DataFrame()    
        IDs = moveID_df.ID.unique().tolist()
        print("IDs=", IDs)
        for ID in IDs[11:]:
            print(f"optimizing helper for ID {ID}")
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]
            config = list(ID_df["config"])[0]
            
            beta_initial = [np.random.rand(), np.random.rand()]
        
            beta_opt = fmin(optimize.compute_secondmove_std_LL, beta_initial, args=(config, ID_df), ftol = 0.001, full_output=True, disp=False)
            beta_df = pd.DataFrame({'ID': [ID], 'helper_goal_noise': beta_opt[0][0], 'helper_action_noise':beta_opt[0][1]})
            
            helper_optimized = pd.concat([helper_optimized, beta_df])

            # write to csv
            helper_optimized.to_csv('opt_results/helper_first_std.csv', index=False)
    
    def optimize_secondmove_prob(moveID_df):
        '''
        '''
        #architect_optimized = pd.read_csv("opt_results/arch_first_prag.csv")    
        helper_optimized = pd.DataFrame()    
        IDs = moveID_df.ID.unique().tolist()
        print("IDs=", IDs)
        for ID in IDs[5:]:
            print(f"optimizing helper for ID {ID}")
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]
            config = list(ID_df["config"])[0]
            
            beta_initial = [np.random.rand(), np.random.rand(),np.random.rand()]
        
            beta_opt = fmin(optimize.compute_secondmove_prob_LL, beta_initial, args=(config, ID_df), ftol = 0.001, full_output=True, disp=False)
            beta_df = pd.DataFrame({'ID': [ID], 'helper_goal_noise': beta_opt[0][0], 'helper_pass_noise':beta_opt[0][1], 'helper_action_noise':beta_opt[0][2]})
            
            helper_optimized = pd.concat([helper_optimized, beta_df])

            # write to csv
            helper_optimized.to_csv('opt_results/helper_first_prob.csv', index=False)
    
    def optimize_firstmove_literal(moveID_df):
        '''
        '''
        architect_optimized = pd.DataFrame()    
        IDs = moveID_df.ID.unique().tolist()
        print("IDs=", IDs)
        for ID in IDs:
            print(f"optimizing architect for ID {ID}")
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]
            config = list(ID_df["config"])[0]
            
            beta_initial = [np.random.rand()]
        
            beta_opt = fmin(optimize.compute_firstmove_literal_LL, beta_initial, args=(config, ID_df), ftol = 0.001, full_output=True, disp=False)
            beta_df = pd.DataFrame({'ID': [ID],  'action_noise':beta_opt[0][0]})
            
            architect_optimized = pd.concat([architect_optimized, beta_df])

            # write to csv
            architect_optimized.to_csv('opt_results/arch_first_lit_useful.csv', index=False)
        

    def optimize_pragmatic_architect(moveID_df):
        '''
        moveID_df contains 10 rows per participant for the 10  goals they completed
        fmin optimization needs to happen with all goals combined
        '''
        #architect_optimized = pd.read_csv("e1 results/probarchitect_optimized.csv")
        architect_optimized = pd.DataFrame()
        IDs = moveID_df.ID.unique().tolist()
        print("IDs=", IDs)
        for ID in IDs:
            print(f"optimizing architect for ID {ID}")
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]
            config = list(ID_df["config"])[0]
            
            beta_initial = [np.random.rand(), np.random.rand()]
        
            beta_opt = fmin(optimize.compute_ll_pragmatic_architect, beta_initial, args=(config, ID_df), ftol = 0.001, full_output=True, disp=False)
            beta_df = pd.DataFrame({'ID': [ID], 'goal_noise': beta_opt[0][0], 'action_noise':beta_opt[0][1]})
            
            architect_optimized = pd.concat([architect_optimized, beta_df])

            # write to csv
            architect_optimized.to_csv('opt_results/arch_prag_goal_opt.csv', index=False)
    
    def optimize_literal_architect(moveID_df):
        '''
        moveID_df contains 10 rows per participant for the 10  goals they completed
        fmin optimization needs to happen with all goals combined
        '''
        architect_optimized = pd.read_csv("opt_results/arch_lit_opt.csv")
        opt_IDs = architect_optimized.ID.unique().tolist()
        #architect_optimized = pd.DataFrame()
        IDs = moveID_df.ID.unique().tolist()
        #print("IDs=", IDs)
        for ID in IDs:
            if(ID in opt_IDs):
                print(f"{ID} already optimized")
            else:
                print(f"optimizing architect for ID {ID}")
                ID_df = moveID_df.loc[moveID_df['ID'] == ID]
                config = list(ID_df["config"])[0]
                
                beta_initial = [np.random.rand()]
            
                beta_opt = fmin(optimize.compute_ll_literal_architect, beta_initial, args=(config, ID_df), ftol = 0.001, full_output=True, disp=False)
                beta_df = pd.DataFrame({'ID': [ID], 'action_noise':beta_opt[0][0]})
                
                architect_optimized = pd.concat([architect_optimized, beta_df])

                # write to csv
                architect_optimized.to_csv('opt_results/arch_lit_opt.csv', index=False)

    

    def simulate_optimized(beta, currentConfig, goal, ID):

        goalspace = general.define_goalspace()
        goal_np_initial = general.get_initial_goal_probs(goalspace)

        goal_np = goal_np_initial.copy()
        move_df = pd.DataFrame()

        for i in range(20):
            ## simulate an OPTIMIZED literal architect + probabilistic OPTIMIZED helper
            move_probs, c = architect.literal_architect_trial(currentConfig,goal, literalA_beta=5.82668988)
            a_move = general.get_random_move(c, move_probs)
            #print("a_move=",a_move)
            a_utility = general.compute_move_utility(currentConfig,a_move,goal)
            # update config 
            newConfig = general.update_config(currentConfig, a_move[0], a_move[1])
            
            success = general.check_goal(goal, newConfig)
            if success == 1:
                #print(f"goal achieved in {i+1} moves")
                break
        
            # helper move
            
            softmax_moves, move_labels, goal_np = helper.probabilistic_helper(currentConfig, a_move, goal_np, goalspace, goal_noise = beta[0], pass_noise = beta[1], action_noise = beta[2])

            h_move = general.get_random_move(move_labels, softmax_moves)
            h_utility = general.compute_move_utility(newConfig,h_move,goal) #utility depends on architect move already happened

            # update config for helper
            currentConfig = general.update_config(newConfig, h_move[0], h_move[1]) 

            success = general.check_goal(goal, currentConfig)
    
            if success == 1:
                #print(f"goal achieved in {i+1} moves")
                break

            move_id_df = pd.DataFrame({'ID': [ID]}) 
            move_id_df['goal'] = goal
            move_id_df['a_move'] = str(a_move)
            move_id_df['a_utility'] = a_utility
            move_id_df['h_move'] = str(h_move)
            move_id_df['h_utility'] = h_utility
            move_id_df["goal_noise"]=beta[0]
            move_id_df["pass_noise"]=beta[1]
            move_id_df["action_noise"]=beta[2]
            move_id_df["move_number"]=i
        
            move_df = pd.concat([move_df, move_id_df])
        
        return move_df
    
    def optimized_simulation(moveID_df):
        '''
        uses optimal parameters for the agents and simulates the first moves
        '''
        simulation_df = pd.DataFrame()
        IDs = moveID_df.ID.unique().tolist()

        architect_betas = pd.read_csv("opt_results/first_arch_optimized.csv")
        helper_betas = pd.read_csv("opt_results/helper_secondmove_optimized.csv")

        for ID in IDs:
            print("for ID=", ID)
            ID_df = moveID_df.loc[moveID_df['ID'] == ID]
            ID_arch_literal = architect_betas.loc[architect_betas['ID'] == ID].values.flatten().tolist()[1]
            ID_arch_prag = architect_betas.loc[architect_betas['ID'] == ID].values.flatten().tolist()[2:]

            ID_helper_baseline = helper_betas.loc[helper_betas['ID'] == ID].values.flatten().tolist()[1:3]
            ID_helper_careful = helper_betas.loc[helper_betas['ID'] == ID].values.flatten().tolist()[3:]
            # each person did multiple goals

            for g in range(len(ID_df)):
                goal = list(ID_df["goal"])[g]
                print("for goal = ", goal)
                ID = list(ID_df["ID"])[g]
                currentConfig = list(ID_df["config"])[g]
                move_IDs = list(ID_df["moveIDs"])[g]
                first_a_move = move_IDs[0]
                first_h_move = move_IDs[1]
                print("first_a_move=",first_a_move)
                print("first_h_move=",first_h_move)
                
                goalspace = general.define_goalspace()
                goal_np = general.get_initial_goal_probs(goalspace)

            
                # calculate optimized moves
                ## for architect

                literal_m, literal_c = architect.literal_architect_trial(currentConfig, goal, literalA_beta= ID_arch_literal, goalspace = goalspace)
                prag_m, prag_c = architect.pragmatic_architect_trial(currentConfig,goal, goal_np, goal_noise = ID_arch_prag[0], action_noise= ID_arch_prag[1])

                literal_move = general.get_random_move(literal_c, literal_m)
                literal_utility = general.compute_move_utility(currentConfig, literal_move, goal)

                prag_move = general.get_random_move(prag_c, prag_m)
                prag_utility = general.compute_move_utility(currentConfig, prag_move, goal)

                ## for helper

                std_softmax_moves, std_move_labels, std_goal_np = helper.standard_helper(currentConfig, first_a_move, goal_np, goalspace, goal_noise = ID_helper_baseline[0], action_noise = ID_helper_baseline[1])
                baseline_move = general.get_random_move(std_move_labels, std_softmax_moves)
                baseline_utility = 0 if baseline_move == "('none','none')" else general.compute_move_utility(currentConfig, baseline_move, goal)
                
                prob_softmax_moves, prob_move_labels, prob_goal_np = helper.probabilistic_helper(currentConfig, first_a_move, goal_np, goalspace, goal_noise = ID_helper_careful[0], pass_noise = ID_helper_careful[1], action_noise = ID_helper_careful[2])
                careful_move = general.get_random_move(prob_move_labels, prob_softmax_moves)
                careful_utility = 0 if careful_move == "('none','none')" else general.compute_move_utility(currentConfig, careful_move, goal)
                

                prob_df = pd.DataFrame({'ID': [ID], 'goal': [goal], 'literal_move': [literal_move], 'literal_utility':[literal_utility], 'prag_move': [prag_move], 'prag_utility': [prag_utility], 'baseline_move': [baseline_move], 'baseline_utility': [baseline_utility], 'careful_move':[careful_move], 'careful_utility': careful_utility})                    
            
                simulation_df = pd.concat([simulation_df, prob_df])
                simulation_df.to_csv('opt_results/final_simulations.csv', index=False)
        
        


#moveID_df = pd.read_csv("e1 results/final_move_df.csv", converters={"moveIDs": literal_eval, "config": literal_eval})
#x = optimize.optimize_helper(moveID_df)

# moveID_df = pd.read_csv("e1 results/final_move_df.csv", converters={"moveIDs": literal_eval, "config": literal_eval})
# x = optimize.optimize_secondmove(moveID_df)

    
