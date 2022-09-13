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
class UCBRank(NNTopK):

    def __init__(self, exploreParam=0.1,
                *args, **kargs):
        """_summary_ this is the UCBRank's implementation.

        Args:
            exploreParam (float, optional): _description_. Defaults to 0.1.
        """
        super(UCBRank, self).__init__(*args, **kargs)
        self.exploreParam=exploreParam
    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({
        'learning_rate': 0.01,
        'learning_rate_decay': 1.0,
        })
        return parent_parameters

    def MergeBehavAndNonBehav(self,Behav,NonBehav, thre=0):
        """_summary_ return the final relevance score. 

        Args:
            Behav (_type_): _description_  relevance score from users behaviour
            NonBehav (_type_): _description_ relevance score from non-user behaviour features. 
        """
        FinalRelScore=np.copy(Behav)
        FinalRelScore[Behav==0]=NonBehav[Behav==0]
        return FinalRelScore
    def GetRelScore(self,query_id, data_split,loggingModel=True,Cold=False):
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
            
        scores = model.score(query_featTensor).detach().numpy()        
        SumClicksOverSumExposure=data_split.query_values_from_vector(query_id,data_split.SumClicksOverSumExposure)
        if Cold:
            return scores
        return self.MergeBehavAndNonBehav(SumClicksOverSumExposure,scores)   
    def GetExploreScore(self,query_id, data_split):
        """_summary_

        Args:
            query_id (_type_): _description_
            data_split (_type_): _description_

        Returns:
            _type_: _description_
        """
        docFreq=data_split.query_values_from_vector(query_id,data_split.docFreq)
        ExploreScore=1/np.clip(np.sqrt(docFreq),1e-3,np.inf)
        return ExploreScore           
    def get_test_rankings(self, query_id, query_feat, data_split,inverted=True, **kwargs):
        Cold=kwargs["Cold"]
        RelScore=self.GetRelScore(query_id, data_split,loggingModel=False,Cold=Cold)
        ranking=rnk.single_ranking(
                            RelScore,
                            n_results=self.n_results) ##higher score ranked higher.
        return ranking
    def _create_train_ranking(self, query_id, query_feat,inverted=False,data_split=None,  **kwargs):
        assert inverted == False
        RelScore=self.GetRelScore(query_id, data_split,loggingModel=True,Cold=False)
        ExploreScore=self.GetExploreScore(query_id, data_split)
        FinalScore=RelScore+ExploreScore*self.exploreParam
        ranking=rnk.single_ranking(
                            FinalScore,
                            n_results=self.n_results)
        return ranking
    

