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
class NNTopK(BasicOnlineRanker):

    def __init__(self, learning_rate, learning_rate_decay,
                *args, **kargs):
        super(NNTopK, self).__init__(*args, **kargs)
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.LearningModel = TorchLeakyReluNeuralModel(n_features = self.n_features,
                                learning_rate = learning_rate,
                                learning_rate_decay = learning_rate_decay)
        self.LoggingModel = self.LearningModel.copy()


    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({
        'learning_rate': 0.05,
        'learning_rate_decay': 1.0,
        })
        return parent_parameters


    def get_test_rankings(self, query_id, query_feat, inverted=False,**kwargs):
        query_featTensor=torch.from_numpy(query_feat)
        scores = self.LearningModel.score(query_featTensor).detach().numpy()
        ranking=rnk.single_ranking(
                            scores,
                            n_results=self.n_results) ##higher score ranked higher.
        return ranking
    def _create_train_ranking(self, query_id, query_feat, inverted=False,data_split=None,**kwargs):
        assert inverted == False
        query_featTensor=torch.from_numpy(query_feat)
        scores = self.LoggingModel.score(query_featTensor).detach().numpy()
        Impressions=data_split.query_values_from_vector(query_id,data_split.docFreq)
        CumC=data_split.query_values_from_vector(query_id,data_split.DebiasedClickSum)
        ranking=rnk.single_ranking(
                            scores,
                            n_results=self.n_results)
        
        
        return ranking
    

    def updateLoggingWithLearning(self,data):
        '''Update logging model with learning model.
        '''
        
        # self.LearningModel
        self.train_model(dataset=data)
        self.LoggingModel=self.LearningModel.copy()
    def update_to_interaction(self, clicks):
        pass

    def _update_to_clicks(self, clicks):
        pass
    
    def loss(self,model,data, **kargs):
        criterion = nn.MSELoss()
        inputs, labels = data["feature_matrix"],data["MeanDebiasedClicks"]
        outputs = model(inputs) # notinception
        loss = criterion(outputs[:,0], labels)
        return loss    
    def train_model(self, dataset,  trainmult=1, valmult=1, num_epochs=1000, epochs_top=0,patience=60):
        dataset.train.setFilteredFreqThreshod(20)
        dataset.validation.setFilteredFreqThreshod(20)
        train_loader=dataset.train.getDataLoader(shuffle=True)
        val_loader=dataset.validation.getDataLoader(shuffle=False)
        
        optimizer=self.LearningModel.GetOptimizer()
        model=self.LearningModel.NNmodel
        Best_val_running_loss=np.inf
        bestCKpoint=None
        ValCounter=0
        for epoch in progressbar(range(num_epochs)):                        
            # for phase in ['val','train']:
            phase="val"
            val_running_loss=0
            total = 0
            model.train(False)  # Set model to evaluate mode
            with torch.no_grad():
                for i in range(valmult):
                    for data in val_loader:
                        # get the inputs
                        # inputs, labels = data["feature_matrix"],data["MeanDebiasedClicks"]
                        # inputs, labels = inputs.to(torch.device("cuda")), labels.to(torch.device("cuda"))
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        # outputs = model(inputs)[:,0]

                        loss = self.loss(model, data,train=True)
                        # statistics
                        Batchsize=data["docFreq"].size(0)
                        total += Batchsize
                        val_running_loss += loss.item()*Batchsize
            if val_running_loss<Best_val_running_loss:
                Best_val_running_loss=val_running_loss
                bestCKpoint=os.path.join(self.LogPath,"epoch_"+str(epoch))
                torch.save(model.state_dict(), bestCKpoint)
                ValCounter=0        
            
            
            phase="train"
            running_loss = 0.0
            total = 0
            # Iterate over data.
            if phase=="train":
                ValCounter+=1
                if ValCounter>patience:          
                    break
                model.train(True)  # Set model to training mode
                for i in range(trainmult):
                    for data in train_loader:
                        # get the inputs
                        # inputs, labels = data["feature_matrix"],data["MeanDebiasedClicks"]
                        # inputs, labels = inputs.to(torch.device("cuda")), labels.to(torch.device("cuda"))
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        # outputs = model(inputs) # notinception
                        loss = self.loss(model, data)
                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()
                        # statistics     
                        Batchsize=data["docFreq"].size(0)                
                        total += Batchsize
                        running_loss += loss.item()*Batchsize
                        

        model.load_state_dict(torch.load(bestCKpoint))
        model.eval()   
        return model