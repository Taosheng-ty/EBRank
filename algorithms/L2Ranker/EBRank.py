import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.TorchModel import TorchLeakyReluNeuralModel,TorchLinearModel
from algorithms.basiconlineranker import BasicOnlineRanker
from algorithms.L2Ranker.UCBRank import UCBRank
from collections import deque
from progressbar import progressbar
import torch
from torch import nn
class EBRank(UCBRank):

    def __init__(self,
                *args, **kargs):
        super(EBRank, self).__init__(*args, **kargs)
        self.outputDim=1  ## Neural Networks' output dimension.
        self.LearningModel = TorchLinearModel(n_features = self.n_features,
                                learning_rate = self.learning_rate,
                                learning_rate_decay = self.learning_rate_decay,
                                outputDim=self.outputDim)
        self.LoggingModel = self.LearningModel.copy()
        self.DefaultBeta=5
    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({
        'learning_rate': 0.05,
        'learning_rate_decay': 1.0,
        })
        return parent_parameters

    def GetRelScore(self,query_id, data_split,loggingModel=True,Cold=False,returnPriorParam=False):
        """_summary_ get the relevance scores.

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
        if Cold:
            return alpha/(beta+alpha)
        scores=(CumC+alpha)/(Impressions+beta+alpha)
        if returnPriorParam:
            return scores,alpha,beta
        return scores 
    def GetExploreScore(self,query_id, data_split,alpha,beta):
        """_summary_ get the exploration scores.

        Args:
            query_id (_type_): _description_
            data_split (_type_): _description_

        Returns:
            _type_: _description_
        """
        q_Exposure=data_split.query_values_from_vector(query_id,data_split.exposure)
        return 1/((q_Exposure+alpha+beta)**2)           
    def _create_train_ranking(self, query_id, query_feat,inverted=False,data_split=None,  **kwargs):
        assert inverted == False
        RelScore,alpha,beta=self.GetRelScore(query_id, data_split,loggingModel=True,Cold=False,returnPriorParam=True)
        ExploreScore=self.GetExploreScore(query_id, data_split,alpha,beta)
        FinalScore=RelScore+ExploreScore*self.exploreParam
        ranking=rnk.single_ranking(
                            FinalScore,
                            n_results=self.n_results)
        
        
        return ranking
    def lossFcn(self,n_trials,CumC,alpha,beta):
        """_summary_ function used to help construct loss.

        Args:
            n_trials (_type_): _description_
            CumC (_type_): _description_
            alpha (_type_): _description_
            beta (_type_): _description_

        Returns:
            _type_: _description_
        """
        t1=self.LogBetaFunction(alpha,beta)
        t2=self.LogBetaFunction(alpha+CumC,beta+n_trials-CumC)
        loss=t1-t2
        return  loss.mean()
    def LogBetaFunction(self,x,y):
        """_summary_  get log(beta(x,y)) by using torch.special.gammaln

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        gammaln=torch.special.gammaln
        return gammaln(x)+gammaln(y)-gammaln(x+y)
    def loss(self,model,data,train=True, **kargs):
        """_summary_ loss function which will be used in training.

        Args:
            model (_type_): _description_
            data (_type_): _description_
            train (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        feature_matrix, docFreq,DebiasedClickSum = data["feature_matrix"],data["docFreq"],data["DebiasedClickSum"]
        alpha = model(feature_matrix)[:,0] # notinception
        beta=torch.tensor(self.DefaultBeta,requires_grad=False)
        ind=alpha>0 *(docFreq>DebiasedClickSum)
        if ind.sum()<1 and train:
            loss=0
        loss = self.lossFcn(docFreq[ind],DebiasedClickSum[ind],alpha[ind],beta)
        return loss    