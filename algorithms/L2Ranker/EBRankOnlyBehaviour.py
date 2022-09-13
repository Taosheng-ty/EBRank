import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.TorchModel import TorchLeakyReluNeuralModel,TorchLinearModel
from algorithms.basiconlineranker import BasicOnlineRanker
from algorithms.L2Ranker.UCBRank import UCBRank
from algorithms.L2Ranker.EBRankV1 import EBRankV1
from collections import deque
from progressbar import progressbar
import torch
from torch import nn
class EBRankOnlyBehaviour(UCBRank):
    """_summary_ a class only use behaviour to rank items which inherit UCBRank

    Args:
        UCBRank (_type_): _description_
    """
    def __init__(self,
                *args, **kargs):
        super(EBRankOnlyBehaviour, self).__init__(*args, **kargs)
    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({
        'learning_rate': 0.01,
        'learning_rate_decay': 1.0,
        })
        return parent_parameters

    def GetRelScore(self,query_id, data_split,loggingModel=True,Cold=False):
        """_summary_

        Args:
            query_id (_type_): _description_
            data_split (_type_): _description_

        Returns:
            _type_: _description_
        """
        Impressions=data_split.query_values_from_vector(query_id,data_split.docFreq)
        CumC=data_split.query_values_from_vector(query_id,data_split.DebiasedClickSum)
        Impressions=np.clip(Impressions,0.01,1e10)
        if Cold:
            return np.zeros_like(Impressions)
        scores=(CumC)/(Impressions)
        return scores 
    def GetExploreScore(self,query_id, data_split):
        """_summary_

        Args:
            query_id (_type_): _description_
            data_split (_type_): _description_

        Returns:
            _type_: _description_
        """
        docFreq=data_split.query_values_from_vector(query_id,data_split.docFreq)
        docFreq=np.clip(docFreq,0.1,np.inf)
        return 0           
    
    def train_model(self, dataset,  trainmult=1, valmult=1, num_epochs=100, epochs_top=0):
        pass