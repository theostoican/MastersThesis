#!/usr/bin/env python
# coding: utf-8

# In[1]:


from copy import deepcopy

import pandas as pd
import torch.nn as nn
import torch

import nlopt
from numpy import *
import numpy as np 

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=32)


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[3]:


data = pd.read_csv('mnist/train_10pca.csv', float_precision='round_trip')
data.head()


# In[4]:


dataset_inputs = []
dataset_labels = []

for idx, row in data.iterrows():
    pca_components = []
    for idx_pca_component in range(1, 11):
        pca_components.append(row[str(idx_pca_component) + '_principal'])
    dataset_inputs.append(pca_components)
    dataset_labels.append(row['label'])


# In[ ]:


torch_dataset_inputs = torch.DoubleTensor(dataset_inputs[:10000]).to(device)
torch_dataset_labels = torch.DoubleTensor([dataset_labels[:10000]]).T.to(device)


# In[ ]:


# jnp_dataset_inputs = jnp.array(dataset_inputs, dtype=jnp.float64)
# jnp_dataset_labels = jnp.array(dataset_labels, dtype=jnp.float64)


# In[ ]:


N, D_in, H_student, D_out = 1, 10, 10, 1


# In[ ]:


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
    self.linear1 = nn.Linear(D_in, H, bias=True).double()
    self.linear2 = nn.Linear(H, H, bias=True).double()
    self.linear3 = nn.Linear(H, H, bias=True).double()
    self.linear4 = nn.Linear(H, D_out, bias=True).double()

    nn.init.xavier_uniform_(self.linear1.weight)
    nn.init.xavier_uniform_(self.linear2.weight)
    nn.init.xavier_uniform_(self.linear3.weight)
    nn.init.xavier_uniform_(self.linear4.weight)
    
    nn.init.constant_(self.linear1.bias, 0)
    nn.init.constant_(self.linear2.bias, 0)
    nn.init.constant_(self.linear3.bias, 0)
    nn.init.constant_(self.linear4.bias, 0)

  def forward(self, x):
    h1 = torch.sigmoid(self.linear1(x))
    h2 = torch.sigmoid(self.linear2(h1))
    h3 = torch.sigmoid(self.linear3(h2))
    y_pred = self.linear4(h3)
    return y_pred


# In[ ]:


## Gradient norm evaluation
def eval_grad_norm(loss_grad):
  cnt = 0
  for g in loss_grad:
      if cnt == 0:
        g_vector = g.contiguous().view(-1)
      else:
        g_vector = torch.cat([g_vector, g.contiguous().view(-1)])
      cnt = 1
  grad_norm = torch.norm(g_vector)
 
  return grad_norm.cpu().detach().numpy()

## Main training entry point.
def train(model, x, y_labels, N = 10 ** 4, Ninner = (10 ** 3), Nstart = 10,
          maxtime = 7, nlopt_threshold = 1e-7,
          collect_history = True):
  lr = 1e-4
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#   checkpoint = torch.load("model_1e-6.pt")
#   model.load_state_dict(checkpoint['model_state_dict'])
  # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)


  loss_fn = nn.MSELoss()
  loss_vals = []
  trace = []
  if collect_history:
    trace.append((deepcopy(model.module.linear1.weight.cpu().data.detach().numpy()),
                  deepcopy(model.module.linear2.weight.cpu().data.detach().numpy()),
                  deepcopy(model.module.linear3.weight.cpu().data.detach().numpy()),
                  deepcopy(model.module.linear4.weight.cpu().data.detach().numpy())))
  for i in range(1, N + 1):
    loss_tmp = []
    for j in range(1, Ninner + 1):
      y = model(x)
      loss = loss_fn(y, y_labels)
      loss_grad = torch.autograd.grad(loss, model.parameters(),
                                      retain_graph=True)
      grad_norm = eval_grad_norm(loss_grad)
      if grad_norm <= 5e-6:
        print('found it')
        EPOCH = 0
        PATH = "model.pt"
        LOSS = 0.4

        torch.save({
                'epoch': EPOCH,
                'model_state_dict': student_model.state_dict(),
                'loss': LOSS,
                }, PATH)
        return loss_vals, trace
      loss_tmp.append(loss.item())
      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()
      if i == 1 and (j % Nstart == 0) and j < Ninner:
        loss_vals.append(np.mean(loss_tmp[j - Nstart  : j]))
        if collect_history:
          trace.append((deepcopy(model.module.linear1.weight.cpu().data.detach().numpy()),
                      deepcopy(model.module.linear2.weight.cpu().data.detach().numpy()),
                      deepcopy(model.module.linear3.weight.cpu().data.detach().numpy()),
                      deepcopy(model.module.linear4.weight.cpu().data.detach().numpy())))
    loss_vals.append(np.mean(loss_tmp))
    if collect_history:
      trace.append((deepcopy(model.module.linear1.weight.cpu().data.detach().numpy()),
                  deepcopy(model.module.linear2.weight.cpu().data.detach().numpy()),
                  deepcopy(model.module.linear3.weight.cpu().data.detach().numpy()),
                  deepcopy(model.module.linear4.weight.cpu().data.detach().numpy())))
    grad_norm = eval_grad_norm(loss_grad)
    print("Iteration: %d, loss: %s, gradient norm: %s" % (Ninner * i,
                                                          np.mean(loss_tmp),
                                                          grad_norm))

  EPOCH = i
  PATH = "model.pt"
  LOSS = 0.4

  torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
  return loss_vals, trace


# In[ ]:


student_model = StudentNetwork(D_in, H_student, D_out)
student_model = student_model.to(device)
if device == 'cuda':
    student_model = torch.nn.DataParallel(student_model)


# In[ ]:


torch.manual_seed(100)
loss_vals, trace = train(student_model,
                          torch_dataset_inputs,
                         torch_dataset_labels)


# In[ ]:


EPOCH = 0
PATH = "model.pt"
LOSS = 0.4

torch.save({
        'epoch': EPOCH,
        'model_state_dict': student_model.state_dict(),
        'loss': LOSS,
        }, PATH)


# In[ ]:


# weights = np.append(
#     np.append(
#         np.append(trace[-1][0].reshape(H_student * D_in),
#                   trace[-1][1].reshape(H_student * D_in)),
#         trace[-1][2].reshape(H_student * D_in)), 
#     trace[-1][3][0])
# print(len(weights))


# In[ ]:


# w_layer1 = weights[0 : 100].reshape(H_student, D_in)
# w_layer2 = weights[100 : 200].reshape(H_student, H_student)
# w_layer3 = weights[200 : 300].reshape(H_student, H_student)
# w_out = weights[300 : ].reshape(D_out, H_student)

# print(jax_loss(weights), jnp.linalg.norm(jax_grad(jax_loss)(weights)))


# In[ ]:


# final_weights = second_order_opt(weights, 5000)


# In[ ]:


# print(jax_loss(final_weights), jnp.linalg.norm(jax_grad(jax_loss)(final_weights)))


# In[ ]:




