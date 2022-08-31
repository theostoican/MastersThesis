import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

## A customizable student network, initialized using Glorot initialization.
class StudentNetwork(nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.

    D_in: input dimension
    H: dimension of hidden layer
    D_out: output dimension of the first layer
    """
    super(StudentNetwork, self).__init__()
    self.linear1 = nn.Linear(D_in, H, bias=False).double()
    self.linear2 = nn.Linear(H, D_out, bias=False).double()
    nn.init.xavier_uniform_(self.linear1.weight)
    nn.init.xavier_uniform_(self.linear2.weight)
  def forward(self, x):
    h_sigmoid = torch.sigmoid(self.linear1(x))
    y_pred = self.linear2(h_sigmoid)
    return y_pred

## A preinitialized teacher network
class TeacherNetwork(nn.Module):
  def __init__(self):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    D_in, H, D_out = 2, 4, 1
    super(TeacherNetwork, self).__init__()
    self.linear1 = nn.Linear(D_in, H, bias=False).double()
    self.linear2 = nn.Linear(H, D_out, bias=False).double()
    self.linear1.weight = torch.nn.Parameter(torch.transpose(
      torch.DoubleTensor([[0.6, -0.5, -0.2, 0.1],
                         [0.5, 0.5, -0.6, -0.6]]),
                         0, 1))
    self.linear2.weight = torch.nn.Parameter(torch.transpose(
      torch.DoubleTensor([[1], [-1], [1], [-1]]),
                         0, 1))
  def forward(self, x):
    h_sigmoid = torch.sigmoid(self.linear1(x))
    y_pred = self.linear2(h_sigmoid)
    return y_pred



## A network customly initialized with input weights.
class DummyNetwork(nn.Module):
  def __init__(self, D_in, H, D_out, w_in, w_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.

    D_in: input dimension
    H: dimension of hidden layer
    D_out: output dimension of the first layer
    """
    super(DummyNetwork, self).__init__()
    self.linear1 = nn.Linear(D_in, H, bias=False).double()
    self.linear2 = nn.Linear(H, D_out, bias=False).double()
    self.linear1.weight = torch.nn.Parameter(w_in)
    self.linear2.weight = torch.nn.Parameter(w_out)
  def forward(self, x):
    h_sigmoid = torch.sigmoid(self.linear1(x))
    y_pred = self.linear2(h_sigmoid)
    return y_pred
