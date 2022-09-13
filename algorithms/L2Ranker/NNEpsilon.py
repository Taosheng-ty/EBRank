import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.TorchModel import TorchLeakyReluNeuralModel
from algorithms.basiconlineranker import BasicOnlineRanker
from algorithms.L2Ranker.NNTopK import NNTopK
from collections import deque
from progressbar import progressbar
import torch
from torch import nn
class NNEpsilon(NNTopK):
    """_summary_ similar to NNTop but use a epsilon strategy to explore items.
    """
    def __init__(self, exploreParam=0.1,
                *args, **kargs):
        super(NNEpsilon, self).__init__(*args, **kargs)
        self.exploreParam=exploreParam
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
        query_featTensor=torch.from_numpy(query_feat)
        scores = self.LoggingModel.score(query_featTensor).detach().numpy()
        n=query_feat.shape[0]
        FinalScores = np.random.uniform(0,1,size=n)*self.exploreParam+scores
        ranking=rnk.single_ranking(
                            FinalScores,
                            n_results=self.n_results)
        return ranking