from copy import deepcopy
import csv
import multiprocess
import argparse

import torch.nn as nn
import torch

import nlopt
from numpy import *
import numpy as np

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=32)

# Various modelling parameters:
# - N is batch size; 
# - D_in is input dimension;
# - H is the dimension of the hidden layer
# - D_out is output dimension.
N, D_in, H_teacher, H_student, D_out = 1, 2, 4, 4, 1

# Dataset creation
def construct_dataset():
  data = []
  for y in np.arange(-5, 5.1, .25):
    for x in np.arange(-5, 5.1, .25):
      data.append([x, y])
  return data

data = torch.DoubleTensor(construct_dataset()) 

# Models

## A preinitialized teacher network
class TeacherNetwork(nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.

    D_in: input dimension
    H: dimension of hidden layer
    D_out: output dimension of the first layer
    """
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

# Generation of the dataset labels based on the teacher model.
teacher_model = TeacherNetwork(D_in, H_teacher, D_out)
y_labels = teacher_model(data).detach()

# Training utils

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
 
  return grad_norm.detach().numpy()

## Loss objective of the second order optimization
def loss_obj(weights, grad):
  w_in = weights[0 : H_student * 2]
  w_out = weights[H_student * 2 :]

  w_in_torch_format = []
  for i in range(H_student):
    w_in_torch_format.append([w_in[2 * i], w_in[2 * i + 1]])
  w_in_torch_format = torch.DoubleTensor(w_in_torch_format)

  w_out_torch_format = torch.DoubleTensor([w_out])

  dummy_model = DummyNetwork(D_in, H_student, D_out, w_in_torch_format,
                             w_out_torch_format)

  loss_val = nn.MSELoss()(dummy_model(data), y_labels)
  
  if grad.size > 0:
    loss_grad = torch.autograd.grad(loss_val, dummy_model.parameters(),
                                    create_graph=True)
    gradients = loss_grad[0].reshape(H_student * 2).detach().numpy()
    gradients = np.append(gradients, loss_grad[1][0].detach().numpy())
    grad[:] = gradients

  return loss_val.item()

## Main method used for computing second order opt.
def second_order_opt(weights, maxtime):
  opt = nlopt.opt(nlopt.LD_SLSQP, len(weights))
  opt.set_lower_bounds([w - 10 for w in weights])
  opt.set_upper_bounds([w + 10 for w in weights])
  opt.set_min_objective(loss_obj)
  opt.set_maxtime(maxtime)
  opt.set_xtol_rel(1e-10)
  final_weights = opt.optimize(weights)

  return final_weights

## Main training entry point.
def train(model, x, y_labels, N = 10, Ninner = 10 ** 4, Nstart = 10,
          maxtime = 7, nlopt_threshold = 1e-7,
          collect_history = True):
  optimizer = torch.optim.Adam(model.parameters())
  for param_group in optimizer.param_groups:
        lr = param_group['lr']

  loss_fn = nn.MSELoss()
  loss_vals = []
  trace = []
  if collect_history:
    trace.append((deepcopy(model.linear1.weight.data.detach().numpy()),
                  deepcopy(model.linear2.weight.data.detach().numpy())))
  for i in range(1, N + 1):
    loss_tmp = []
    for j in range(1, Ninner + 1):
      y = model(x)
      loss = loss_fn(y, y_labels)
      loss_grad = torch.autograd.grad(loss, model.parameters(),
                                      retain_graph=True)
      loss_tmp.append(loss.item())
      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()
      if i == 1 and (j % Nstart == 0) and j < Ninner:
        loss_vals.append(np.mean(loss_tmp[j - Nstart  : j]))
        if collect_history:
          trace.append((deepcopy(model.linear1.weight.data.detach().numpy()),
                        deepcopy(model.linear2.weight.data.detach().numpy())))
    loss_vals.append(np.mean(loss_tmp))
    if collect_history:
      trace.append((deepcopy(model.linear1.weight.data.detach().numpy()),
                    deepcopy(model.linear2.weight.data.detach().numpy())))
    grad_norm = eval_grad_norm(loss_grad)
    print("Iteration: %d, loss: %s, gradient norm: %s" % (Ninner * i,
                                                          np.mean(loss_tmp),
                                                          grad_norm))
    
    # Adjust the learning rate.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1 + i)
    
    # Stopping criterion
    if np.mean(loss_tmp) < nlopt_threshold or i == N:
        weights = np.append(trace[-1][0].reshape(H_student * 2),
                            trace[-1][1][0].reshape(H_student))
        final_weights = second_order_opt(weights, maxtime)
        return loss_vals, trace, final_weights

## The main method used for running an experiment, starting from a certain seed.
## Run in parallel for multiple seeds.
def run_experiment(num_experiment):
  print('Number of experiment:', num_experiment)
  torch.manual_seed(num_experiment)

  student_model = StudentNetwork(D_in, H_student, D_out)
  _, trace, final_weights = train(student_model, data, y_labels)

  w_in = torch.DoubleTensor(final_weights[0 : 2 * H_student].reshape(H_student,
                                                                     2))
  w_out = torch.DoubleTensor([final_weights[2 * H_student :]])
  dummy_model = DummyNetwork(D_in, H_student, D_out, w_in, w_out)
  last_loss_val = nn.MSELoss()(dummy_model(data), y_labels).item()
  loss_grad = torch.autograd.grad(nn.MSELoss()(dummy_model(data), y_labels),
                                               dummy_model.parameters(),
                                               create_graph=True)
  grad_norm = eval_grad_norm(loss_grad)

  row = [last_loss_val, grad_norm, H_student]

  for i in range(0, H_student):
    neuron_w_x = []
    neuron_w_y = []
    neuron_a = []
    for (inp_weights, out_weights) in trace:
      neuron_w_x.append(inp_weights[i][0])
      neuron_w_y.append(inp_weights[i][1])
      neuron_a.append(out_weights[0][i])
    neuron_w_x.append(w_in[i][0].item())
    neuron_w_y.append(w_in[i][1].item())
    neuron_a.append(w_out[0][i].item())
    row.append(neuron_w_x)
    row.append(neuron_w_y)
    row.append(neuron_a)

  teacher_neurons_x = [0.6, -0.5, -0.2, 0.1]
  teacher_neurons_y = [0.5, 0.5, -0.6, -0.6]
  row.append(teacher_neurons_x)
  row.append(teacher_neurons_y)

  return row

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--num_experiments', dest='num_experiments', required=True)
  args = parser.parse_args()
  num_experiments = int(args.num_experiments)

  file_experiment_header = ['loss', 'gradient norm', 'student size']

  for i in range(0, H_student):
    file_experiment_header.append('neuron_' + str(i) + '_traj_x')
    file_experiment_header.append('neuron_' + str(i) + '_traj_y')
    file_experiment_header.append('neuron_' + str(i) + '_a')

  file_experiment_header.append('teacher_neurons_x')
  file_experiment_header.append('teacher_neurons_y')


  file_experiment_data = open('experiments_data.csv', 'w')
  writer = csv.writer(file_experiment_data)
  writer.writerow(file_experiment_header)

  with multiprocess.Pool(4) as pool:
      result = pool.map(run_experiment, range(num_experiments))
      for row in result:
          writer.writerow(row)

  file_experiment_data.close()
