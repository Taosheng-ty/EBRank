# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import utils.rankings as rnk

class BasicOnlineRanker(object):

  def __init__(self, n_results, n_features,LogPath=None,*args, **kargs):
    self.n_features = n_features
    self.n_results = n_results
    self.LogPath=LogPath
    self.n_interactions = 0
    self.model_updates = 0
    self._messages = {}
    self._default_messages = {}

    self._train_features = None
    self._train_query_ranges = None
    self.LearningModel=None
    self.LoggingModel=None
  @staticmethod
  def default_parameters():
    '''Return all parameter values for this ranker.
       used for logging purposes.'''
    return {}
  def updateLoggingWithLearning(self,*args, **kwargs):
    '''Update logging model with learning model.
    '''
    self.LoggingModel=self.LearningModel.copy()
  def add_message(self, name, default_value=0):
    self._default_messages[name] = default_value

  def remove_message(self, name):
    del self._default_messages[name]

  def set_message(self, name, value):
    self._messages[name] = value

  def get_messages(self):
    messages = self._default_messages.copy()
    messages.update(self._messages)
    return messages

  def reset_messages(self):
    self._messages.clear()

  def setup(self, train_features, train_query_ranges):
    self._train_features = train_features
    self._train_query_ranges = train_query_ranges

  def clean(self):
    del self._train_features
    del self._train_query_ranges

  # def get_test_rankings(self, features,
  #                       query_ranges, inverted=True):
  #   return rnk.rank_multiple_queries(
  #                     np.zeros(features.shape[0]),
  #                     query_ranges,
  #                     inverted=inverted,
  #                     n_results=self.n_results)

  def get_query_features(self, query_id, features,
                         query_ranges):
    start_i = query_ranges[query_id]
    end_i = query_ranges[query_id + 1]
    return features[start_i:end_i, :]

  def get_query_label(self, query_id, label_vector,
                      query_ranges):
    start_i = query_ranges[query_id]
    end_i = query_ranges[query_id + 1]
    return label_vector[start_i:end_i]

  def get_query_size(self, query_id, query_ranges):
    return query_ranges[query_id+1] - query_ranges[query_id]

  def get_train_query_ranking(self, query_id, inverted=False):
    self._last_query_id = query_id
    query_feat = self.get_query_features(query_id,
                                     self._train_features,
                                     self._train_query_ranges)
    self._last_ranking = self._create_train_ranking(
                                        query_id,
                                        query_feat,
                                        inverted)[:self.n_results]
    return self._last_ranking
  
  def get_query_dataSplit_LoggingRanking(self, query_id, data_split,inverted=False):
    query_feat=data_split.query_values_from_vector(query_id,data_split.feature_matrix)
    self._last_ranking = self._create_train_ranking(
                                      query_id,
                                      query_feat,
                                      data_split=data_split)[:self.n_results]
    return self._last_ranking   
  def get_query_dataSplit_TestRanking(self, query_id, data_split,inverted=False,Cold=True):
    query_feat=data_split.query_values_from_vector(query_id,data_split.feature_matrix)
    if Cold and data_split.ExpandFeature:  
      query_featCold=np.zeros_like(query_feat)
      query_featCold[:,:-1]=query_feat[:,:-1]
      query_feat=query_featCold
    ranking = self.get_test_rankings(
                                    query_id=query_id,
                                    query_feat=query_feat,
                                    data_split=data_split,Cold=Cold)[:self.n_results]
    return ranking

  def _create_train_ranking(self, query_id, query_feat, inverted=False):
    """_summary_  This funciton will usually be overwrited.

    Args:
        query_id (_type_): _description_
        query_feat (_type_): _description_
        inverted (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_docs = self.get_query_size(query_id,
                                 self._train_query_ranges)
    return rnk.rank_single_query(np.zeros(n_docs),
                    inverted=inverted,
                    n_results=self.n_results)[:self.n_results]

  def process_clicks(self, clicks):
    self.update_to_interaction(clicks)
    self.n_interactions += 1
  
  def update_to_interaction(self, clicks):
    pass
  
  def OutputWeights(self):
    if self.LearningModel is not None:
      Weight=self.LearningModel.getWeight()
      return Weight
    else:
      return None