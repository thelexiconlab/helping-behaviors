def level_1_helper_trial(configArray, a_move, goal_probs_history, goalspace, level1Helper_beta, literalAbeta, pragAbeta):
      # helper function for level 1 has two steps
      # 1. calculate goal probabilities based on previous move
      # 2. calculate expected utility for each possible next move
      # returns the expected utility for each move
      # and the move label

      # first get goal matrix for every possible move
      if len(a_move) > 0:
          goal_np_overall, u, s, c = helper.create_goal_matrix(configArray, goal_probs_history, model="literal", literalAbeta =literalAbeta, pragAbeta =  pragAbeta, goalspace = goalspace)
          goal_np = goal_np_overall[c.index(a_move)]
      else:
          goal_np = goal_probs_history
      # next change the configArray based on the move made by architect
      
      configArray = general.update_config(configArray, a_move[0], a_move[1])
      
      # gnerate the utility matrix, moves for this new config
      u_i = general.generate_utility_matrix(configArray, goalspace)
      
      # # shape of u_i is len(config_change) x 2 (for utility matrix of size future_moves x goal and future_move_labels)
      # for each u_i, we calculated expected utility 
      
      u_matrix = u_i[0]

      move_labels = u_i[1]
      
      # add a pass move here with utility 0

      pass_utility = [0]*len(goalspace)

      u_matrix = np.row_stack((u_matrix,pass_utility))
      move_labels = move_labels + [("none","none")]

      softmax_moves, move_labels = general.get_expected_utilities_softmax(u_matrix, goal_np, move_labels, level1Helper_beta)
      
      return softmax_moves, move_labels, goal_np

  def level_1_helper_trial_new_utils(configArray, a_move, goal_probs_history, goalspace, level1Helper_beta, literalAbeta, pragAbeta):
      # helper function for level 1 has two steps
      # 1. calculate goal probabilities based on previous move
      # 2. calculate expected utility for each possible next move
      # returns the expected utility for each move
      # and the move label

      # first get goal matrix for every possible move
      if len(a_move) > 0:
          goal_np_overall, u, s, c = helper.create_goal_matrix(configArray, goal_probs_history, model="literal", literalAbeta =literalAbeta, pragAbeta =  pragAbeta, goalspace = goalspace)
          goal_np = goal_np_overall[c.index(a_move)]
      else:
          goal_np = goal_probs_history
      # next change the configArray based on the move made by architect
      
      configArray = general.update_config(configArray, a_move[0], a_move[1])
      
      # gnerate the utility matrix, moves for this new config
      u_i = general.generate_utility_matrix(configArray, goalspace)
      
      


      # # shape of u_i is len(config_change) x 2 (for utility matrix of size future_moves x goal and future_move_labels)
      # for each u_i, we calculated expected utility 
      
      u_matrix = u_i[0]

      # set all values that are -1 in u_1 to -5
      # for the helper, the utility of a harmful move is much higher than the utility of a beneficial move
      u_matrix[u_matrix == -1] = -10
      move_labels = u_i[1]
      
      # add a pass move here with utility 0

      pass_utility = [0]*len(goalspace)

      u_matrix = np.row_stack((u_matrix,pass_utility))
      move_labels = move_labels + [("none","none")]

      softmax_moves, move_labels = general.get_expected_utilities_softmax(u_matrix, goal_np, move_labels, level1Helper_beta)
      
      return softmax_moves, move_labels, goal_np

  def level_1_helper_trial_unk_goals(configArray, a_move, goal_probs_history, goalspace, level1Helper_beta, literalAbeta, pragAbeta, unk_parameter):
      '''
      This helper keeps a fixed probability for unknown goals, and renormalizes the probabilities for known goals to sum to 1 with that unk parameter
      '''
      # helper function for level 1 has two steps
      # 1. calculate goal probabilities based on previous move
      # 2. calculate expected utility for each possible next move
      # returns the expected utility for each move
      # and the move label

      # first get goal matrix for every possible move
      if len(a_move) > 0:
          goal_np_overall, u, s, c = helper.create_goal_matrix(configArray, goal_probs_history, model="literal", literalAbeta =literalAbeta, pragAbeta =  pragAbeta, goalspace = goalspace)
          goal_np = goal_np_overall[c.index(a_move)]
      else:
          goal_np = goal_probs_history

      ## renormalize based on the unk_probability
      known_p = 1-unk_parameter
      goal_known_p = goal_np*known_p
      goal_probs_final = np.array(goal_known_p.tolist() + [unk_parameter])

      #goal_probs_final = np.array([goal_np]*(len(goalspace)) + [unk_parameter])
      # next change the configArray based on the move made by architect
      
      configArray = general.update_config(configArray, a_move[0], a_move[1])
      
      # gnerate the utility matrix, moves for this new config
      u_i = general.generate_utility_matrix(configArray, goalspace)

      ## add a row to each move with utility = -1 for an unknown goal
      utility_for_unk_goal = np.array([-1]*len(u_i[0]))
      
      u_i_with_unk_goal = np.column_stack((u_i[0],utility_for_unk_goal.T))
      
      # # shape of u_i is len(config_change) x 2 (for utility matrix of size future_moves x goal and future_move_labels)
      # for each u_i, we calculated expected utility 
      
      u_matrix = u_i_with_unk_goal

      # for each u_i, we calculated expected utility 
      
      # print(u_matrix)
      move_labels = u_i[1]
      # add a pass move here with utility 0 
      pass_utility = [0]*(len(goalspace)) + [0]

      u_matrix = np.row_stack((u_matrix,pass_utility))
      move_labels = move_labels + [("none","none")]

      softmax_moves, move_labels = general.get_expected_utilities_softmax(u_matrix, goal_probs_final, move_labels, level1Helper_beta)
      
      return softmax_moves, move_labels, goal_probs_final[:-1]

  def level_1_helper_trial_memoryless(configArray, a_move, goalspace, level1Helper_beta, literalAbeta, pragAbeta):
      '''
      This helper keeps evaluates goals only based on the most recent move
      '''
      
      # first get goal matrix for every possible move
      goal_np_overall, u, s, c = helper.memoryless_goal_matrix(configArray, model="literal", literalAbeta=literalAbeta, pragAbeta=pragAbeta, goalspace=goalspace)
      goal_np = goal_np_overall[c.index(a_move)]

      # next change the configArray based on the move made by architect
      
      configArray = general.update_config(configArray, a_move[0], a_move[1])
      
      # gnerate the utility matrix, moves for this new config
      u_i = general.generate_utility_matrix(configArray, goalspace)
      
      # # shape of u_i is len(config_change) x 2 (for utility matrix of size future_moves x goal and future_move_labels)
      # for each u_i, we calculated expected utility 
      
      u_matrix = u_i[0]

      move_labels = u_i[1]
      
      # add a pass move here with utility 0

      pass_utility = [0]*len(goalspace)

      u_matrix = np.row_stack((u_matrix,pass_utility))
      move_labels = move_labels + [("none","none")]

      softmax_moves, move_labels = general.get_expected_utilities_softmax(u_matrix, goal_np, move_labels, level1Helper_beta)
      
      return softmax_moves, move_labels, goal_np

  def level_1_helper_trial_learning(configArray, a_move, goalspace, goal_probs_history, level1Helper_beta, literalAbeta, pragAbeta, threshold):
      # this helper computes change in goal probabilities based on the move made by architect
      # and uses a threshold to determine if the change is significant
      if len(a_move) > 0:
          goal_np_overall, u, s, c = helper.create_goal_matrix(configArray, goal_probs_history, model="literal", literalAbeta =literalAbeta, pragAbeta =  pragAbeta, goalspace = goalspace)
          goal_np = goal_np_overall[c.index(a_move)]
      else:
          goal_np = goal_probs_history

      # find the top two goals and compute the difference 
      goal_np_list = goal_np.tolist()
      goal_np_list.sort()
      top_two_goals = goal_np_list[-2:]
      goal_change = top_two_goals[0] - top_two_goals[1]
      print("top_two_goals=", top_two_goals)
      print("goal_change=",goal_change)
      
      # # next change the configArray based on the move made by architect
      configArray = general.update_config(configArray, a_move[0], a_move[1])

      # gnerate the utility matrix, moves for this new config
      u_i = general.generate_utility_matrix(configArray, goalspace)    
      
      u_matrix = u_i[0]

      move_labels = u_i[1]

      # add a pass move here with utility 0 for known goals and utility = +3 for unknown goals
      pass_utility = [0]*(len(goalspace))

      u_matrix = np.row_stack((u_matrix,pass_utility))
      move_labels = move_labels + [("none","none")]

      softmax_moves, move_labels = general.get_expected_utilities_softmax(u_matrix, goal_np, move_labels, level1Helper_beta)

      # if the difference between the top 2 most likely goals is over a threshold
      # then continue, else pass
    
      if goal_change <= threshold:
          # then we want a pass move (none, none)
          # get index of pass move
          pass_index = len(softmax_moves) - 1
          # set probabibilty of pass move to 1 and everything else to 0
          softmax_moves[pass_index] = 1
          for i in range(len(softmax_moves)):
              if i != pass_index:
                  softmax_moves[i] = 0
      return softmax_moves, move_labels, goal_np

  def compute_ll_helper(beta, participant_data, model, goalspace):
      nll = 0    
      
      goal_np_initial = general.get_initial_goal_probs(goalspace)
      if(len(participant_data.goalnumber.unique()) == 11): 
          for g in range(1, 11):
          #for g in range(1, 2):
              goal_data = participant_data[participant_data["goalnumber"]==g].reset_index()
              
              if(len(goal_data)>0):
                  goal = goal_data["goal"][0]
                  a_moves = list(goal_data["architectMove"])
                  a_moves = list(filter(('999').__ne__, a_moves))
                  h_moves = list(goal_data["helperMove"])
                  h_moves = list(filter(('999').__ne__, h_moves))
                  h_moves = list(filter(('done').__ne__, h_moves))
                  move_seq = [x for x in itertools.chain.from_iterable(itertools.zip_longest(a_moves,h_moves)) if x]

                  # now we pass the move sequence into the ID func
                  moveIDs = [general.move_description_to_num(m) if m != "pass" else m for m in move_seq ]
                  config = participant_data["config"][0].split(',')

                  std_goal_np = goal_np_initial.copy()
                  mem_goal_np = goal_np_initial.copy()
                  unk_goal_np = goal_np_initial.copy()
                  mod_goal_np = goal_np_initial.copy()
                  learn_goal_np = goal_np_initial.copy()
                  currentConfig = config.copy()
                  for i in range(0, len(moveIDs)):
                      move = moveIDs[i]
                      if i % 2 != 0:
                          # helper move
                          prev_move = moveIDs[i-1] # previous architect move
                          # figure out which kind of helper we are optimizing and compute probs only for that helper

                          if model == "standard":
                              std_softmax_moves, std_move_labels, std_goal_np = helper.level_1_helper_trial(currentConfig,prev_move,std_goal_np, goalspace,level1Helper_beta=beta[0], literalAbeta=beta[1], pragAbeta=1)
                              prob = std_softmax_moves[std_move_labels.index(("none","none"))] if move == "pass" else std_softmax_moves[std_move_labels.index(move)]
                          elif model == "memoryless":
                              mem_softmax_moves, mem_move_labels, mem_goal_np = helper.level_1_helper_trial_memoryless(currentConfig,prev_move, goalspace,level1Helper_beta=beta[0], literalAbeta=beta[1], pragAbeta=1)
                              prob = mem_softmax_moves[mem_move_labels.index(("none","none"))] if move == "pass" else mem_softmax_moves[mem_move_labels.index(move)]
                          elif model == "unknown":
                              unk_softmax_moves, unk_move_labels, unk_goal_np = helper.level_1_helper_trial_unk_goals(currentConfig,prev_move,unk_goal_np, goalspace,level1Helper_beta=beta[0], literalAbeta=beta[1], pragAbeta=1, unk_parameter=beta[2])
                              prob = unk_softmax_moves[unk_move_labels.index(("none","none"))] if move == "pass" else unk_softmax_moves[unk_move_labels.index(move)]
                          elif model == "modified":
                              mod_softmax_moves, mod_move_labels, mod_goal_np = helper.level_1_helper_trial_new_utils(currentConfig, prev_move, mod_goal_np, goalspace, level1Helper_beta=beta[0], literalAbeta=beta[1], pragAbeta=1)
                              prob = mod_softmax_moves[mod_move_labels.index(("none","none"))] if move == "pass" else mod_softmax_moves[mod_move_labels.index(move)]
                          else: # model == "learning"
                              learn_softmax_moves, learn_move_labels, learn_goal_np = helper.level_1_helper_trial_learning(currentConfig, prev_move, goalspace, learn_goal_np, level1Helper_beta = beta[0], literalAbeta = beta[1], pragAbeta = 1, threshold = beta[2])
                              prob = learn_softmax_moves[learn_move_labels.index(("none","none"))] if move == "pass" else learn_softmax_moves[learn_move_labels.index(move)]
                          
                          # now we add the probability to the nll
                          nll += -np.log(prob)
                          # update based on architect move    
                          currentConfig = general.update_config(currentConfig, prev_move[0], prev_move[1])
                          # update based on helper move
                          if move == "pass":
                              currentConfig = general.update_config(currentConfig, "none", "none")
                          else:
                              currentConfig = general.update_config(currentConfig, move[0], move[1])
                          
                          
                          # print("move prob is ", prob)
                          # print("nll is ", nll)
      #print(f"for ID {id}, nll is ", nll)
      return nll
    
  def optimize_helper(data):
    for i in range(1, len(data)):
      from scipy.optimize import fmin
      helper_optimized = pd.DataFrame()
      #for i in range(0, 1):
      d = pd.read_csv(data[i])
      ID = d["helperRandCode"][0]
      beta_initial_two_params = [np.random.rand(), np.random.rand()]
      beta_initial_three_paramas = [np.random.rand(), np.random.rand(), np.random.rand()]
      print(f"optimizing helper for ID {ID}")
      beta_opt_standard = fmin(helper.compute_ll_helper, beta_initial_two_params, args=(d, "standard"), ftol = 0.001, full_output=True, disp=False)
      beta_df = pd.DataFrame({'ID': [ID], 'helperBeta': beta_opt_standard[0][0], 'assumedArchitectBeta': beta_opt_standard[0][1], 'unk/thresh':'NA', 'model' : "standard"})
      print("standard model done")

      beta_opt_memoryless = fmin(helper.compute_ll_helper, beta_initial_two_params, args=(d, "memoryless"), ftol = 0.001, full_output=True, disp=False)
      beta_df = pd.concat([beta_df, pd.DataFrame({'ID': [ID], 'helperBeta': beta_opt_memoryless[0][0], 'assumedArchitectBeta': beta_opt_memoryless[0][1], 'unk/thresh':'NA', 'model' : "memoryless"})])
      print("memoryless model done")
      
      beta_opt_unknown = fmin(helper.compute_ll_helper, beta_initial_three_paramas, args=(d, "unknown"), ftol = 0.001, full_output=True, disp=False)
      beta_df = pd.concat([beta_df, pd.DataFrame({'ID': [ID], 'helperBeta': beta_opt_unknown[0][0], 'assumedArchitectBeta': beta_opt_unknown[0][1], 'unk/thresh':beta_opt_unknown[0][2], 'model' : "unknown"})])
      print("unknown model done")

      beta_opt_modified = fmin(helper.compute_ll_helper, beta_initial_two_params, args=(d, "modified"), ftol = 0.001, full_output=True, disp=False)
      beta_df = pd.concat([beta_df, pd.DataFrame({'ID': [ID], 'helperBeta': beta_opt_modified[0][0], 'assumedArchitectBeta': beta_opt_modified[0][1], 'unk/thresh':'NA', 'model' : "modified"})])
      print("modified model done")

      beta_opt_learning = fmin(helper.compute_ll_helper, beta_initial_three_paramas, args=(d, "learning"), ftol = 0.001, full_output=True, disp=False)
      beta_df = pd.concat([beta_df, pd.DataFrame({'ID': [ID], 'helperBeta': beta_opt_learning[0][0], 'assumedArchitectBeta': beta_opt_learning[0][1], 'unk/thresh':beta_opt_learning[0][2], 'model' : "learning"})])
      print("learning model done")
      
      helper_optimized = pd.concat([helper_optimized, beta_df])

      # write to csv
      helper_optimized.to_csv('e1 results/helper_optimized.csv', index=False)
def memoryless_goal_matrix(configArray, model, literalAbeta, pragAbeta, goalspace):
      # for each possible move that could be taken, it calculates the goal probability
      # by looking at literal architect probabilities by MOVE and normalizing
      # then it multiplies with the previous goal probability

      # get the architect probabilities starting for current configuration
      if model == "literal":
          u, s, c = architect.literal_architect_probs(configArray, literalAbeta, goalspace)
      else:    
          print("pragmatic")
          #u, s, c = pragmatic_architect_probs(configArray, goal_probs_history, pragAbeta, literalAbeta)
      move_axis_sum = s.sum(axis=1) 
      s_norm = s/move_axis_sum[:,np.newaxis]
      return s_norm, u, s,c  
    
  def noisy_goal(distribution, evidence_units):
    '''
    takes a distribution and makes it noisy based on eta
    '''

    amount_of_noise = 1/(evidence_units+1) # inversely proportional to evidence units
    N = len(distribution)
    if amount_of_noise == 0: # perfectly noisy: all goals equally likely ()
      new_d = np.array(softmax([1]*N))
    else:
      new_d = distribution + np.random.normal(0, amount_of_noise*.00005)
    return new_d
