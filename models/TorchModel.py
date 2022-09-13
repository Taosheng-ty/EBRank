import torch
from torch import nn, optim
def init_weights(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.uniform_(m.weight,a=0,b=0.1)
        m.bias.data.fill_(0.01)
def Kaiminginit_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
class TorchLeakyReluNeuralModel(object):
  """_summary_ linear model class built on torch.

  Args:
      object (_type_): _description_
  """
  def __init__(self, learning_rate,
               learning_rate_decay,
               hidden_layers=[32,32], n_features=100,outputDim=1, **kargs):


    self.learning_rate = learning_rate
    self.hidden_layers = hidden_layers
    self.biases = []
    self.n_features=n_features
    self.outputDim=outputDim
    self.NNmodel=self.modelConstruct([n_features]+hidden_layers,outputDim=outputDim)
    self.learning_rate_decay = learning_rate_decay
    self.optimizer = torch.optim.SGD(self.NNmodel.parameters(), lr=learning_rate) # built-in L2
  def GetOptimizer(self,optimizer="SGD"):
    if optimizer=="SGD":
      optimizer = torch.optim.SGD(self.NNmodel.parameters(), lr=self.learning_rate) # built-in L2
    if optimizer=="Adam":
      optimizer = torch.optim.Adam(self.NNmodel.parameters(), lr=self.learning_rate) # built-in L2
    return optimizer
  def score(self, features):
    return self.NNmodel(features)[:, 0]
  def getWeight(self):
    """_summary_ return weights of the model
    """
    return self.NNmodel[0].weight[0].detach().tolist()
  def copy(self):
    copy = TorchLeakyReluNeuralModel(learning_rate = self.learning_rate,
                                   learning_rate_decay=self.learning_rate_decay,
                                   hidden_layers = self.hidden_layers,\
                                  n_features=self.n_features,outputDim=self.outputDim)
    copy.NNmodel.load_state_dict(self.NNmodel.state_dict())
    return copy
  def modelConstruct(self,hidden_layers,outputDim=1):
    layers=[]
    # for ind,_ in enumerate(hidden_layers[:-1]):
    #   layers.append(nn.Linear(hidden_layers[ind], hidden_layers[ind+1]))
    #   layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_layers[0], outputDim))
    layers.append(nn.LeakyReLU())
    layers.append(nn.Flatten())
    NNmodel = nn.Sequential(
    *layers)
    NNmodel.apply(Kaiminginit_weights)
    return NNmodel
  
class TorchLinearModel(TorchLeakyReluNeuralModel):
  def __init__(self, initiScale=5, *args, **kargs):
    self.initiScale=initiScale
    super(TorchLinearModel, self).__init__(*args, **kargs)
  def modelConstruct(self,hidden_layers,outputDim=1):
    layers=[]
    # for ind,_ in enumerate(hidden_layers[:-1]):
    #   layers.append(nn.Linear(hidden_layers[ind], hidden_layers[ind+1]))
    #   layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_layers[0], outputDim))
    # layers.append(nn.s())
    layers.append(nn.Flatten())
    NNmodel = nn.Sequential(
    *layers)
    NNmodel.apply(self.init_weights)
    return NNmodel  
  def init_weights(self,m):
    """_summary_ initialize the weights

    Args:
        m (_type_): _description_
    """

    if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight)
        nWeight=m.weight.shape[1]
        b=4*self.initiScale/nWeight 
        torch.nn.init.uniform_(m.weight,a=0,b=b)
        m.bias.data.fill_(0.01)
  def copy(self):
    copy = TorchLinearModel(learning_rate = self.learning_rate,
                                   learning_rate_decay=self.learning_rate_decay,
                                   hidden_layers = self.hidden_layers,\
                                  n_features=self.n_features,outputDim=self.outputDim)
    copy.NNmodel.load_state_dict(self.NNmodel.state_dict())
    return copy
class ScaledSigmoid(nn.Module):
    def __init__(self,Scaleweight):
        super().__init__()
        self.Scaleweight=Scaleweight

    def forward(self, x):
        y = nn.Sigmoid()(x)*self.Scaleweight
        return y
class TorchSigmoidModel(TorchLeakyReluNeuralModel):
  def __init__(self, Scaleweight,*args, **kargs):
    self.Scaleweight=Scaleweight
    super(TorchSigmoidModel, self).__init__(*args, **kargs)
    self.Scaleweight=Scaleweight
  def modelConstruct(self,hidden_layers,outputDim=1):
    """_summary_ construct a NN model according to hidden layers.

    Args:
        hidden_layers (_type_): _description_
        outputDim (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    layers=[]
    layers.append(nn.Linear(hidden_layers[0], outputDim))
    layers.append(ScaledSigmoid(self.Scaleweight))
    layers.append(nn.Flatten())
    NNmodel = nn.Sequential(
    *layers)
    NNmodel.apply(init_weights)
    return NNmodel  
  def copy(self):
    """_summary_ copy the model

    Returns:
        _type_: _description_
    """
    copy = TorchSigmoidModel(learning_rate = self.learning_rate,
                                   learning_rate_decay=self.learning_rate_decay,
                                   hidden_layers = self.hidden_layers,\
                                  n_features=self.n_features,outputDim=self.outputDim,\
                                    Scaleweight=self.Scaleweight)
    copy.NNmodel.load_state_dict(self.NNmodel.state_dict())
    return copy