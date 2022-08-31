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

# get_ipython().system('pip install --upgrade "jax[cpu]"')

import jax
import jax.numpy as jnp
from jax import nn as jax_nn
from jax.config import config; config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=32) 
from jax import jacfwd, jacrev
from jax import grad as jax_grad


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[3]:


N, D_in, H_student, D_out = 1, 10, 10, 1


# In[4]:


data = pd.read_csv('mnist/train_10pca.csv', float_precision='round_trip')
data.head()


# In[5]:


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


# In[6]:


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


jnp_dataset_inputs = jnp.array(dataset_inputs[:10000], dtype=jnp.float64)
jnp_dataset_labels = jnp.array(dataset_labels[:10000], dtype=jnp.float64)


# In[ ]:


student_model = StudentNetwork(D_in, H_student, D_out)
student_model = student_model.to(device)
if device == 'cuda':
    student_model = torch.nn.DataParallel(student_model)

checkpoint = torch.load("model_1e-6.pt")
student_model.load_state_dict(checkpoint['model_state_dict'])


# In[ ]:


y = student_model(torch_dataset_inputs)
loss = nn.MSELoss()(y, torch_dataset_labels)


# In[ ]:


print(loss.item())


# In[ ]:


loss_grad = torch.autograd.grad(loss, student_model.parameters(),
                                      retain_graph=True)


# In[ ]:


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

print(eval_grad_norm(loss_grad))


# In[ ]:


trace = []
trace.append((deepcopy(student_model.module.linear1.weight.cpu().data.detach().numpy()),
              deepcopy(student_model.module.linear1.bias.cpu().data.detach().numpy()),
              deepcopy(student_model.module.linear2.weight.cpu().data.detach().numpy()),
              deepcopy(student_model.module.linear2.bias.cpu().data.detach().numpy()),
              deepcopy(student_model.module.linear3.weight.cpu().data.detach().numpy()),
              deepcopy(student_model.module.linear3.bias.cpu().data.detach().numpy()),
              deepcopy(student_model.module.linear4.weight.cpu().data.detach().numpy()),
              deepcopy(student_model.module.linear4.bias.cpu().data.detach().numpy())))

weights = np.append(
    np.append(
        np.append(trace[-1][0].reshape(H_student * D_in),
                  trace[-1][1].reshape(H_student)),
        np.append(
            np.append(trace[-1][2].reshape(H_student * D_in), 
                  trace[-1][3].reshape(H_student)),
            np.append(trace[-1][4].reshape(H_student * D_in), 
                  trace[-1][5].reshape(H_student)))),
    np.append(trace[-1][6][0],
              trace[-1][7][0]))
print(len(weights))


# In[ ]:


def predict(w_layer1, b_layer1,
            w_layer2, b_layer2,
            w_layer3, b_layer3,
            w_out, b_out):
  h1 = jax_nn.sigmoid(jnp_dataset_inputs @ jnp.transpose(w_layer1) + b_layer1)
  h2 = jax_nn.sigmoid(h1 @ jnp.transpose(w_layer2) + b_layer2)
  h3 = jax_nn.sigmoid(h2 @ jnp.transpose(w_layer3) + b_layer3)

  return jnp.transpose((h3 @ w_out + b_out).T)

def jax_loss(w):
  w_layer1 = w[0 : 100].reshape(H_student, D_in)
  b_layer1 = w[100 : 110].reshape(H_student)
  w_layer2 = w[110 : 210].reshape(H_student, H_student)
  b_layer2 = w[210 : 220].reshape(H_student)
  w_layer3 = w[220 : 320].reshape(H_student, H_student)
  b_layer3 = w[320 : 330].reshape(H_student)
  w_out = w[330 : 340].reshape(H_student, D_out)
  b_out = w[340]

  preds = jnp.transpose(predict(w_layer1, b_layer1,
                                w_layer2, b_layer2,
                                w_layer3, b_layer3,
                                w_out, b_out))
  return jnp.mean(jnp.square(preds - jnp_dataset_labels))

def hessian(f):
  return jacfwd(jacrev(f))

print(jax_loss(weights), jnp.linalg.norm(jax_grad(jax_loss)(weights)))
H = hessian(jax_loss)(weights)
H = (H + H.T) / 2.0
jnp.linalg.eigh(H)[0][:12]


# In[ ]:


def loss_obj(weights, grad):
  loss_val = jax_loss(weights)
  if grad.size > 0:
    grad[:] = np.array(jax_grad(jax_loss)(weights), dtype=np.float64)
  return np.float64(loss_val)

def second_order_opt(weights, maxtime):
  opt = nlopt.opt(nlopt.LD_SLSQP, len(weights))
  opt.set_lower_bounds([w - 1000 for w in weights])
  opt.set_upper_bounds([w + 1000 for w in weights])
  opt.set_min_objective(loss_obj)
  opt.set_maxtime(maxtime)
  # opt.set_xtol_rel(1e-32)
  opt.set_initial_step(1e-32)
  final_weights = opt.optimize(weights)
  return final_weights


# In[ ]:


# final_weights = second_order_opt(weights, 10)
# print(jax_loss(final_weights), jnp.linalg.norm(jax_grad(jax_loss)(final_weights)))


# In[ ]:


first_derivative_sigmoid = lambda x : jax_nn.sigmoid(x) *                                           (1 - jax_nn.sigmoid(x))
second_derivative_sigmoid = lambda x : jax_nn.sigmoid(x) *                                            ((1 - jax_nn.sigmoid(x)) ** 2)-                                           (jax_nn.sigmoid(x) ** 2) *                                           (1 - jax_nn.sigmoid(x))

w_layer1 = weights[0 : 100].reshape(H_student, D_in)
b_layer1 = weights[100 : 110].reshape(H_student)
w_layer2 = weights[110 : 210].reshape(H_student, H_student)
b_layer2 = weights[210 : 220].reshape(H_student)
w_layer3 = weights[220 : 320].reshape(H_student, H_student)
b_layer3 = weights[320 : 330].reshape(H_student)
w_out = weights[330 : 340].reshape(H_student, D_out)
b_out = weights[340]

preds = predict(w_layer1, b_layer1,
                                w_layer2, b_layer2,
                                w_layer3, b_layer3,
                                w_out, b_out).reshape(10000)

e = preds - jnp_dataset_labels

print(e.shape)

Y = jnp.zeros((10, 10))

for idx, x in enumerate(jnp_dataset_inputs):
    Y += w_out[0][0] * second_derivative_sigmoid(jnp.dot(x, w_layer3[0]) + b_layer3[0]) * e[idx] *          jnp.array(x).reshape(len(x), 1) @ jnp.array(x).reshape(1, len(x))

Y /= len(jnp_dataset_inputs)


# In[ ]:


evals, _ = jnp.linalg.eigh(Y)
print(evals)


# In[ ]:


new_w_layer3 = np.append(weights[220 : 230], weights[220 : 320]).reshape(H_student + 1, H_student)
new_b_layer3 = np.append([weights[320]], weights[320 : 330]).reshape(H_student + 1)
new_w_out = np.append([weights[330] / 2.0, weights[330] / 2.0], weights[331 : 340]).reshape(H_student + 1, D_out)


# In[ ]:


# new_weights = []
# new_weights = np.append(new_weights, weights[0 : 100])
# new_weights = np.append(new_weights, weights[100 : 110])

# new_weights = np.append(new_weights, weights[110 : 210])
# new_weights = np.append(new_weights, weights[210 : 220])

# new_weights = np.append(new_weights, new_w_layer3)
# new_weights = np.append(new_weights, new_b_layer3)

# new_weights = np.append(new_weights, new_w_out)
# new_weights = np.append(new_weights, [weights[340]])


# In[ ]:


class DummyNetwork(nn.Module):
  def __init__(self, D_in, H, D_out,
               w_layer1, b_layer1,
               w_layer2, b_layer2,
               w_layer3, b_layer3,
               w_out, b_out):
    super(DummyNetwork, self).__init__()
    self.linear1 = nn.Linear(D_in, H, bias=True).double()
    self.linear2 = nn.Linear(H, H, bias=True).double()
    self.linear3 = nn.Linear(H, H + 1, bias=True).double()
    self.linear4 = nn.Linear(H + 1, D_out, bias=True).double()
    
    self.linear1.weight = torch.nn.Parameter(w_layer1)
    self.linear2.weight = torch.nn.Parameter(w_layer2)
    self.linear3.weight = torch.nn.Parameter(w_layer3)
    self.linear4.weight = torch.nn.Parameter(w_out)
    
    self.linear1.bias = torch.nn.Parameter(b_layer1)
    self.linear2.bias = torch.nn.Parameter(b_layer2)
    self.linear3.bias = torch.nn.Parameter(b_layer3)
    self.linear4.bias = torch.nn.Parameter(b_out)

  def forward(self, x):
    h1 = torch.sigmoid(self.linear1(x))
    h2 = torch.sigmoid(self.linear2(h1))
    h3 = torch.sigmoid(self.linear3(h2))
    y_pred = self.linear4(h3)
    return y_pred


# In[ ]:


dummy_network = DummyNetwork(D_in, H_student, D_out, torch.DoubleTensor(w_layer1), torch.DoubleTensor(b_layer1),
                            torch.DoubleTensor(w_layer2), torch.DoubleTensor(b_layer2),
                            torch.DoubleTensor(new_w_layer3), torch.DoubleTensor(new_b_layer3),
                            torch.DoubleTensor(new_w_out.T), torch.DoubleTensor([b_out]))
dummy_network = dummy_network.to(device)
if device == 'cuda':
    dummy_network = torch.nn.DataParallel(dummy_network)


# In[ ]:


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
#     if i % 3 == 0:
#       optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     elif i % 3 == 1:
#       optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
#     else:
#       optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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

#   EPOCH = i
#   PATH = "model.pt"
#   LOSS = 0.4

#   torch.save({
#             'epoch': EPOCH,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
#             }, PATH)
  return loss_vals, trace


# In[ ]:


loss_vals, trace = train(dummy_network,
                          torch_dataset_inputs,
                          torch_dataset_labels)


# In[ ]:


EPOCH = 0
PATH = "local_min_moddle.pt"
LOSS = 0.4
torch.save({
            'epoch': EPOCH,
            'model_state_dict': dummy_network.state_dict(),
            'loss': LOSS,
            }, PATH)


# In[ ]:




