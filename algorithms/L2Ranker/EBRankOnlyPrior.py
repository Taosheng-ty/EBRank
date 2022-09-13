import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.TorchModel import TorchLeakyReluNeuralModel,TorchLinearModel
from algorithms.basiconlineranker import BasicOnlineRanker
from algorithms.L2Ranker.EBRank import EBRank
from algorithms.L2Ranker.EBRankV1 import EBRankV1
from collections import deque
from progressbar import progressbar
import torch
from torch import nn
class EBRankOnlyPrior(EBRank):

    def __init__(self,
                *args, **kargs):
        """_summary_ a class only use prior model to rank items which inherit EBRank

        Args:
            EBRank (_type_): _description_
        """
        super(EBRankOnlyPrior, self).__init__(*args, **kargs)
        # self.outputDim=1  ## Neural Networks' output dimension.
        # self.LearningModel = TorchLinearModel(n_features = self.n_features,
        #                         learning_rate = self.learning_rate,
        #                         learning_rate_decay = self.learning_rate_decay,
        #                         outputDim=self.outputDim)
        # self.LoggingModel = self.LearningModel.copy()
        # self.DefaultBeta=10 
    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({
        'learning_rate': 0.05,
        'learning_rate_decay': 1.0,
        })
        return parent_parameters
    def _create_train_ranking(self, query_id, query_feat,inverted=False,data_split=None,  **kwargs):
        assert inverted == False
        RelScore=self.GetRelScore(query_id, data_split,loggingModel=True,Cold=False)
        # ExploreScore=self.GetExploreScore(query_id, data_split)
        FinalScore=RelScore
        ranking=rnk.single_ranking(
                            FinalScore,
                            n_results=self.n_results)
        return ranking
    def GetRelScore(self,query_id, data_split,loggingModel=True,Cold=False,returnPriorParam=True):
        """_summary_

        Args:
            query_id (_type_): _description_
            data_split (_type_): _description_

        Returns:
            _type_: _description_
        """
        query_feat=data_split.query_values_from_vector(query_id,data_split.feature_matrix)
        query_featTensor=torch.from_numpy(query_feat)
        model= self.LoggingModel if loggingModel else self.LearningModel
            
        Output = model.score(query_featTensor).detach().numpy()
        alpha,beta=Output,self.DefaultBeta
        beta=np.clip(beta,1e-4,np.inf)
        Impressions=data_split.query_values_from_vector(query_id,data_split.docFreq)
        CumC=data_split.query_values_from_vector(query_id,data_split.DebiasedClickSum)
        scores=alpha/(beta+alpha)
        return scores