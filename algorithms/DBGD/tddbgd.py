# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.linearmodel import LinearModel
from algorithms.basiconlineranker import BasicOnlineRanker
from multileaving.TeamDraftMultileave import TeamDraftMultileave

# Dueling Bandit Gradient Descent
class TD_DBGD(BasicOnlineRanker):

  def __init__(self, learning_rate, learning_rate_decay,
               *args, **kargs):
    super(TD_DBGD, self).__init__(*args, **kargs)
    self.learning_rate = learning_rate
    self.LearningModel= LinearModel(n_features = self.n_features,
                             learning_rate = learning_rate,
                             n_candidates = 1,
                             learning_rate_decay = learning_rate_decay)
    self.LoggingModel=self.LearningModel.copy()
    self.multileaving = TeamDraftMultileave(
                             n_results=self.n_results)
    self.parent_parameters=TD_DBGD.default_parameters()


  @staticmethod
  def default_parameters():
    parent_parameters = BasicOnlineRanker.default_parameters()
    parent_parameters.update({
      'learning_rate': 0.1,
      'learning_rate_decay': 0.99999,
      })
    return parent_parameters
  def updateLoggingWithLearning(self,*args, **kwargs):
    '''Update logging model with learning model.
    '''
    self.LoggingModel=self.LearningModel.copy()
    self.LearningModel.resetLearning_rate() ##reset the decay rate to start another round ofleanring.
  
  def get_test_rankings(self, query_id, query_feat, inverted=False,**kwargs):
    scores = self.LearningModel.score(query_feat)
    ranking=rnk.rank_single_query(
                        scores,
                        n_results=self.n_results)  ##higher score, ranked higher.
    return ranking

  def _create_train_ranking(self, query_id, query_feat, inverted=False,data_split=None,**kwargs):
    assert inverted == False
    self.LoggingModel.sample_candidates()
    scores = self.LoggingModel.candidate_score(query_feat)
    rankings = rnk.rank_single_query(scores, inverted=False, n_results=self.n_results)
    multileaved_list = self.multileaving.make_multileaving(rankings)
    return multileaved_list

  def update_to_interaction(self, clicks):
    winners = self.multileaving.winning_rankers(clicks)
    gradient=self.LoggingModel.getGradients(winners)
    if gradient is not None:
      self.LearningModel.updateGradient(gradient)
  
  
