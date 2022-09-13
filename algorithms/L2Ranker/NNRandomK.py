import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.TorchModel import TorchLeakyReluNeuralModel
from algorithms.L2Ranker.NNTopK import NNTopK
from algorithms.basiconlineranker import BasicOnlineRanker
from collections import deque
from progressbar import progressbar
import torch
from torch import nn
class NNRandomK(NNTopK):
    """_summary_ randomly rank items and serve to users.

    Args:
        NNTopK (_type_): _description_
    """
    def __init__(self,  *args, **kargs):
        super(NNRandomK, self).__init__(*args, **kargs)


    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({
        'learning_rate': 0.01,
        'learning_rate_decay': 1.0,
        })
        return parent_parameters


    def _create_train_ranking(self, query_id, query_feat, inverted=False,**kwargs):
        assert inverted == False
        n=query_feat.shape[0]
        scores = np.random.uniform(0,1,size=n)
        ranking=rnk.single_ranking(
                            scores,
                            n_results=self.n_results)
        
        
        return ranking