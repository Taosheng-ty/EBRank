import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.TorchModel import TorchLeakyReluNeuralModel
from algorithms.basiconlineranker import BasicOnlineRanker
from collections import deque
from progressbar import progressbar
import torch
from torch import nn
class BM25(BasicOnlineRanker):

    def __init__(self, learning_rate, learning_rate_decay,data,
                *args, **kargs):
        super(BM25, self).__init__(*args, **kargs)
        self.BM25Dim=data.BM25Dim


    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({
        'learning_rate': 0.05,
        'learning_rate_decay': 1.0,
        })
        return parent_parameters


    def get_test_rankings(self, query_id, query_feat, inverted=False,**kwargs):
        scores = query_feat[:,self.BM25Dim]
        ranking=rnk.single_ranking(
                            scores,
                            n_results=self.n_results) ##higher score ranked higher.
        return ranking
    def _create_train_ranking(self, query_id, query_feat, inverted=False,data_split=None,**kwargs):
        assert inverted == False
        scores = query_feat[:,self.BM25Dim]
        ranking=rnk.single_ranking(
                            scores,
                            n_results=self.n_results)##higher score ranked higher.
        
        
        return ranking
    

    def updateLoggingWithLearning(self,data):
        '''Update logging model with learning model.
        '''
        
        pass
    def update_to_interaction(self, clicks):
        pass

    def _update_to_clicks(self, clicks):
        pass
    def train_model(self, dataset,  trainmult=1, valmult=1, num_epochs=100, epochs_top=0):
        pass