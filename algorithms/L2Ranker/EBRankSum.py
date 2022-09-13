import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.TorchModel import TorchLeakyReluNeuralModel,TorchLinearModel,TorchSigmoidModel
from algorithms.basiconlineranker import BasicOnlineRanker
from algorithms.L2Ranker.UCBRank import UCBRank
from algorithms.L2Ranker.EBRank import EBRank
from collections import deque
from progressbar import progressbar
import torch
from torch import nn
class EBRankSum(EBRank):
    """_summary_ deprecated ones.

    Args:
        EBRank (_type_): _description_
    """
    def __init__(self,
                *args, **kargs):
        super(EBRankSum, self).__init__(*args, **kargs)
        self.outputDim=1  ## Neural Networks' output dimension.
        self.DefaultBetaAlphaSum=10
        self.LearningModel = TorchSigmoidModel(n_features = self.n_features,
                                learning_rate = self.learning_rate,
                                learning_rate_decay = self.learning_rate_decay,
                                outputDim=self.outputDim,
                                Scaleweight=self.DefaultBetaAlphaSum)
        self.LoggingModel = self.LearningModel.copy()
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
        query_feat=data_split.query_values_from_vector(query_id,data_split.feature_matrix)
        query_featTensor=torch.from_numpy(query_feat)
        model= self.LoggingModel if loggingModel else self.LearningModel
            
        Output = model.score(query_featTensor).detach().numpy()
        alpha,beta=Output,self.DefaultBetaAlphaSum-Output
        Impressions=data_split.query_values_from_vector(query_id,data_split.docFreq)
        CumC=data_split.query_values_from_vector(query_id,data_split.DebiasedClickSum)
        if Cold:
            return alpha/(beta+alpha)
        scores=(CumC+alpha)/(Impressions+beta+alpha)
        return scores          
    
    def train_model(self, dataset,  trainmult=1, valmult=1, num_epochs=50, epochs_top=0):
        dataset.train.setFilteredFreqThreshod(20)
        dataset.validation.setFilteredFreqThreshod(20)
        train_loader=dataset.train.getDataLoader()
        val_loader=dataset.validation.getDataLoader()
        optimizer=self.LearningModel.GetOptimizer()
        model=self.LearningModel.NNmodel
        for epoch in progressbar(range(num_epochs)):                        
            for phase in ['train', 'val']:
                running_loss = 0.0
                running_acc = 0
                total = 0
                # Iterate over data.
                if phase=="train":
                    model.train(True)  # Set model to training mode
                    for i in range(trainmult):
                        for data in train_loader:
                            # get the inputs
                            feature_matrix, docFreq,DebiasedClickSum = data["feature_matrix"],data["docFreq"],data["DebiasedClickSum"]
                            # inputs, labels = inputs.to(torch.device("cuda")), labels.to(torch.device("cuda"))
                            # zero the parameter gradients
                            optimizer.zero_grad()
                            # forward
                            alpha = model(feature_matrix)[:,0] # notinception
                            BetaAlphaSum=torch.tensor(self.DefaultBetaAlphaSum,requires_grad=False)
                            loss = self.lossFcn(docFreq,DebiasedClickSum,alpha,BetaAlphaSum-alpha)
                            # backward + optimize only if in training phase
                            loss.backward()
                            optimizer.step()
                            # statistics                      
                            total += docFreq.size(0)
                            running_loss += loss.item()*docFreq.size(0)

                else:
                    model.train(False)  # Set model to evaluate mode
                    with torch.no_grad():
                        for i in range(valmult):
                            for data in val_loader:
                                # get the inputs
                                feature_matrix, docFreq,DebiasedClickSum = data["feature_matrix"],data["docFreq"],data["DebiasedClickSum"]
                                # inputs, labels = inputs.to(torch.device("cuda")), labels.to(torch.device("cuda"))
                                # zero the parameter gradients
                                optimizer.zero_grad()
                                # forward
                                alpha = model(feature_matrix)[:,0] # notinception
                                BetaAlphaSum=torch.tensor(self.DefaultBetaAlphaSum,requires_grad=False)
                                loss = self.lossFcn(docFreq,DebiasedClickSum,alpha,BetaAlphaSum-alpha)
                                # statistics
                                total += docFreq.size(0)
                                running_loss += loss.item()*docFreq.size(0)

                                val_loss=(running_loss/total)
                            # print(val_loss)

        return model