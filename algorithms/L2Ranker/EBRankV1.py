import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.TorchModel import TorchLeakyReluNeuralModel,TorchLinearModel
from algorithms.basiconlineranker import BasicOnlineRanker
from algorithms.L2Ranker.EBRank import EBRank
from algorithms.L2Ranker.UCBRank import UCBRank
from collections import deque
from progressbar import progressbar
import torch
from torch import nn
class EBRankV1(UCBRank):

    def __init__(self,
                *args, **kargs):
        """_summary_ EBRank version 1 which use clicks/exposure to train.
        """
        super(EBRankV1, self).__init__(*args, **kargs)
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
        q_ClickSum=data_split.query_values_from_vector(query_id,data_split.ClickSum)
        q_Exposure=data_split.query_values_from_vector(query_id,data_split.exposure)
        if Cold:
            return alpha/(beta+alpha)
        scores=(q_ClickSum+alpha)/(q_Exposure+beta+alpha)
        if returnPriorParam:
            return scores,alpha,beta
        return scores 
    def GetExploreScore(self,query_id, data_split,alpha,beta):
        """_summary_

        Args:
            query_id (_type_): _description_
            data_split (_type_): _description_

        Returns:
            _type_: _description_
        """
        q_ClickSum=data_split.query_values_from_vector(query_id,data_split.ClickSum)
        q_Exposure=data_split.query_values_from_vector(query_id,data_split.exposure)
        return (q_ClickSum+alpha)/((q_Exposure+alpha+beta)**3)            
    def get_test_rankings(self, query_id, query_feat, data_split,inverted=True, **kwargs):
        Cold=kwargs["Cold"]
        RelScore=self.GetRelScore(query_id, data_split,loggingModel=False,Cold=Cold)
        ranking=rnk.single_ranking(
                            RelScore,
                            n_results=self.n_results) ##higher score ranked higher.
        return ranking
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
        feature_matrix, exposure,ClickSum = data["feature_matrix"],data["exposure"],data["ClickSum"]
        alpha = model(feature_matrix)[:,0]
        beta=torch.tensor(self.DefaultBeta,requires_grad=False)
        ind=alpha>0 
        if ind.sum()<1 and train:
            loss=0
        loss = self.lossFcn(exposure[ind],ClickSum[ind],alpha[ind],beta)
        return loss
    # def train_model(self, dataset,  trainmult=1, valmult=1, num_epochs=50, epochs_top=0):
    #     dataset.train.setFilteredFreqThreshod(20)
    #     dataset.validation.setFilteredFreqThreshod(20)
    #     train_loader=dataset.train.getDataLoader()
    #     val_loader=dataset.validation.getDataLoader()
    #     optimizer=self.LearningModel.GetOptimizer()
    #     model=self.LearningModel.NNmodel
    #     for epoch in progressbar(range(num_epochs)):                        
    #         for phase in ['train', 'val']:
    #             running_loss = 0.0
    #             running_acc = 0
    #             total = 0
    #             # Iterate over data.
    #             if phase=="train":
    #                 model.train(True)  # Set model to training mode
    #                 for i in range(trainmult):
    #                     for data in train_loader:
    #                         # get the inputs
    #                         # feature_matrix, exposure,ClickSum,docFreq = data["feature_matrix"],data["exposure"],data["ClickSum"],data["docFreq"]
    #                         # inputs, labels = inputs.to(torch.device("cuda")), labels.to(torch.device("cuda"))
    #                         # zero the parameter gradients
    #                         optimizer.zero_grad()
    #                         # forward
    #                         # alpha = model(feature_matrix)[:,0] # notinception
    #                         # beta=torch.tensor(self.DefaultBeta,requires_grad=False)
    #                         # ind=alpha>0
    #                         # if ind.sum()<1:
    #                         #     continue
    #                         # loss = self.lossFcn(exposure[ind],ClickSum[ind],alpha[ind],beta)
    #                         loss=self.loss(model,data)
    #                         # backward + optimize only if in training phase
    #                         loss.backward()
    #                         optimizer.step()
    #                         # statistics                      
    #                         total += docFreq.size(0)
    #                         running_loss += loss.item()*docFreq.size(0)

    #             else:
    #                 model.train(False)  # Set model to evaluate mode
    #                 with torch.no_grad():
    #                     for i in range(valmult):
    #                         for data in val_loader:
    #                             # get the inputs
    #                             # feature_matrix, exposure,ClickSum,docFreq = data["feature_matrix"],data["exposure"],data["ClickSum"],data["docFreq"]
    #                             # inputs, labels = inputs.to(torch.device("cuda")), labels.to(torch.device("cuda"))
    #                             # zero the parameter gradients
    #                             optimizer.zero_grad()
    #                             # forward
    #                             # alpha = model(feature_matrix)[:,0] # notinception
    #                             # beta=torch.tensor(self.DefaultBeta,requires_grad=False)
    #                             # loss = self.lossFcn(exposure,ClickSum,alpha,beta)
    #                             loss=self.loss(model,data)
    #                             docFreq=data["docFreq"]
    #                             # statistics
    #                             total += docFreq.size(0)
    #                             running_loss += loss.item()*docFreq.size(0)

    #                             val_loss=(running_loss/total)
    #                         # print(val_loss)

    #     return model