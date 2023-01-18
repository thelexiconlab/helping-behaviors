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

class architect:

  def literal_architect_probs(configArray, literalA_beta, goalspace):
      # gets utility matrix and computes softmax probabilities under all goals
      utility_matrix, combs = general.generate_utility_matrix(configArray, goalspace)
      soft_optimal_moves = general.softmax_t(utility_matrix, literalA_beta)
      return utility_matrix, soft_optimal_moves, combs
  
  def literal_architect_trial(configArray, goal, literalA_beta= 100, goalspace = general.define_goalspace()):
    u, s, c = architect.literal_architect_probs(configArray, literalA_beta, goalspace)
    # select probs for specific goal only
    goal_index = goalspace.index(goal)
    move_probs = s[:,goal_index]
    return move_probs, c
  
  def pragmatic_architect_probs(configArray, goal_probs_history, goal_noise, goalspace, action_noise):
    # computes goal matrix for each possible move
    gp, u, s, c = helper.create_goal_matrix(configArray, goal_probs_history, noise_value = goal_noise ,goalspace = goalspace)
    # once we get the goal probabilities for each possible move, we want to extract probabilities of true goal here and softmax over all moves
    new_s = general.softmax_t(gp, action_noise)
    return u, new_s, c
  
  def pragmatic_architect_trial(configArray, goal, goal_probs_history, goal_noise, action_noise):
    goalspace = general.define_goalspace()
    u, new_s, c = architect.pragmatic_architect_probs(configArray, goal_probs_history, goal_noise, goalspace, action_noise)
    # select probs for specific goal only
    goal_index = goalspace.index(goal)
    move_probs = new_s[:,goal_index]
    return move_probs, c



class helper:

  def create_goal_matrix(configArray, goal_probs_history, noise_value, goalspace):
      # for each possible move that could be taken, it calculates the goal probability
      # by looking at literal architect probabilities by MOVE and normalizing
      # then it multiplies with the previous goal probability

      # get the architect probabilities starting for current configuration
      u, s, c = architect.literal_architect_probs(configArray, noise_value, goalspace)

      move_axis_sum = s.sum(axis=1) 
      s_norm = s/move_axis_sum[:,np.newaxis]
      # multiply with previous goal probability
      goal_np = np.multiply(s_norm, goal_probs_history)
      # normalize by summing over all moves
      goal_sum = goal_np.sum(axis=1)
      goal_np = goal_np/goal_sum[:,np.newaxis]
      return goal_np, u, s,c   
  
  def get_expected_utilities_softmax(utility_moves, goal_np, move_labels, level1Helper_beta = 20):
    # utility_moves is a np array of shape goals x possible move_from x move_to utilities 
    # we multiply the goal probs with the rawcost E(ai) = p(goal)*U(ai)
    expectation_matrix = np.matmul(utility_moves, goal_np)
    # sum for each given action
    #expected_utility = expectation_matrix.sum(axis=0)
    softmax_moves = general.softmax_t(expectation_matrix, level1Helper_beta)
    
    return softmax_moves, move_labels

  
  def probabilistic_goal_inference(configArray, a_move, goal_probs_history, goalspace, goal_noise):
    '''
    assumption would be that goal inference improves with time, i.e., more moves
    '''

    if len(a_move) > 0:
          goal_np_overall, u, s, c = helper.create_goal_matrix(configArray, goal_probs_history, noise_value =goal_noise,  goalspace = goalspace)
          goal_np = goal_np_overall[c.index(a_move)]
    else:
        goal_np = goal_probs_history
    
    return goal_np
  
  def logit2prob(logit):
    odds = math.exp(logit)
    prob = odds / (1 + odds)
    return(prob)


  def pass_or_act(goal_probs, goalspace, pass_noise):
    '''
    based on goal probabilities, decides probability of passing
    '''
    # find the top two goals and compute the difference 
    goal_np_list = goal_probs.tolist()
    goal_np_list.sort()
    top_two_goals = goal_np_list[-2:]
    top_two_indices = sorted(range(len(goal_probs)), key=lambda i: goal_probs[i])[-2:]
    top_two_verbal = [goalspace[i] for i in top_two_indices]
    #print("top two goals =", top_two_verbal)
    # probability of passing = expit (goal_change)

    goal_change = top_two_goals[1] - top_two_goals[0]
    # print("top_two_goals probs=", top_two_goals)
    # print("goal_change=",goal_change)

    pass_parameter = 0.0001 if 1-(pass_noise*goal_change) < 0 else 1-(pass_noise*goal_change)
    prob_pass= helper.logit2prob(logit(pass_parameter))

    return prob_pass, goal_change


  def probabilistic_action_inference(configArray, a_move, goal_np, goalspace, action_noise,prob_pass):
      # helper function for level 1 has two steps
      # 1. calculate goal probabilities based on previous move
      # 2. calculate expected utility for each possible next move
      # returns the expected utility for each move
      # and the move label
  
      # # next change the configArray based on the move made by architect
      configArray = general.update_config(configArray, a_move[0], a_move[1])

      # gnerate the utility matrix, moves for this new config
      u_i = general.generate_utility_matrix(configArray, goalspace)    
      u_matrix = u_i[0]
      move_labels = u_i[1]
      softmax_moves, move_labels = helper.get_expected_utilities_softmax(u_matrix, goal_np, move_labels, action_noise)

      prob_action = 1-prob_pass
      # renormalize all the non-pass moves

      softmax_moves_renormalized = softmax_moves*prob_action
      softmax_moves_final = np.array(softmax_moves_renormalized.tolist() + [prob_pass])
      #print("best action AFTER passing considered:")
      #best_move = np.where(softmax_moves_final == softmax_moves_final.max())[0][0]
      move_labels = move_labels + [("none","none")]
      #print(move_labels[best_move])
      
      return softmax_moves_final, move_labels
  
  def probabilistic_helper(configArray, a_move, goal_np, goalspace, goal_noise, pass_noise, action_noise):
    goal_hypothesis = helper.probabilistic_goal_inference(configArray, a_move, goal_np, goalspace, goal_noise)

    pass_prob, goal_change = helper.pass_or_act(goal_hypothesis, goalspace, pass_noise)
        
    # next they do action inference
    softmax_moves, move_labels = helper.probabilistic_action_inference(configArray, a_move, goal_hypothesis, goalspace, action_noise, pass_prob )  

    return softmax_moves, move_labels, goal_hypothesis
  
  def standard_helper(configArray, a_move, goal_np, goalspace, goal_noise, action_noise):
    goal_hypothesis = helper.probabilistic_goal_inference(configArray, a_move, goal_np, goalspace, goal_noise)

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

    softmax_moves, move_labels = helper.get_expected_utilities_softmax(u_matrix, goal_hypothesis, move_labels, action_noise)
    
    return softmax_moves, move_labels, goal_hypothesis
