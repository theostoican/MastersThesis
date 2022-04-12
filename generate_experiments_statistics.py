
from collections import deque
from copy import deepcopy
import csv
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import jax.numpy as jnp
from jax import nn as jax_nn
from jax import grad as jax_grad
from jax import jacfwd, jacrev
from jax.config import config

config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=32) 

import nlopt

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=32)

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
    self.linear1 = nn.Linear(D_in, H, bias=False)
    self.linear2 = nn.Linear(H, D_out, bias=False)
    self.linear1.weight = torch.nn.Parameter(torch.transpose(
      torch.DoubleTensor([[0.6, -0.5, -0.2, 0.1], 
                          [0.5, 0.5, -0.6, -0.6]]),
                          0,1))
    self.linear2.weight = torch.nn.Parameter(
      torch.transpose(torch.DoubleTensor([[1], [-1], [1], [-1]]), 0, 1))
  def forward(self, x):
    h_sigmoid = torch.sigmoid(self.linear1(x))
    y_pred = self.linear2(h_sigmoid)
    return y_pred

## A dummy network that receives as input the prest incoming and
## outgoing weights and sets them directly.
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
    self.linear1 = nn.Linear(D_in, H, bias=False)
    self.linear2 = nn.Linear(H, D_out, bias=False)
    self.linear1.weight = torch.nn.Parameter(w_in)
    self.linear2.weight = torch.nn.Parameter(w_out)
  def forward(self, x):
    h_sigmoid = torch.sigmoid(self.linear1(x))
    y_pred = self.linear2(h_sigmoid)
    return y_pred

# Various modelling parameters:
# - N is batch size; 
# - D_in is input dimension;
# - H is the dimension of the hidden layer
# - D_out is output dimension.
D_in, H_teacher, H_student, D_out = 2, 4, 5, 1

# Dataset
def construct_dataset():
  data = []
  for y in np.arange(-5, 5.1, .25):
    for x in np.arange(-5, 5.1, .25):
      data.append([x, y])
  return data

dataset = torch.DoubleTensor(construct_dataset()) 

# Labels
teacher_model = TeacherNetwork(D_in, H_teacher, D_out)
y_labels = teacher_model(dataset).detach()

# Data containing local minima, for which the statistics is done.
data = pd.read_csv('experiments_data_student_5_local_min_only.csv',
                  float_precision='round_trip')

# Helper for extracting the weights from one data point.

def extract_weights(data_point):
  incoming_weights_x = []
  incoming_weights_y = []
  outgoing_weights = []

  for i in range(0, int(data_point['student size'])):
    neuron_traj_x = np.fromstring(
      data_point['neuron_' + str(i) + '_traj_x'][1:-1], dtype=float, sep=',')
    neuron_traj_y = np.fromstring(
      data_point['neuron_' + str(i) + '_traj_y'][1:-1], dtype=float, sep=',')
    incoming_weights_x.append(neuron_traj_x[-1])
    incoming_weights_y.append(neuron_traj_y[-1])
    neuron_traj_out = np.fromstring(
      data_point['neuron_' + str(i) + '_a'][1:-1],
      dtype=float, sep=',')
    outgoing_weights.append(neuron_traj_out[-1])

  return incoming_weights_x, incoming_weights_y, outgoing_weights

# JAX helpers for finding an escape route

jnp_inputs = jnp.array(construct_dataset(), dtype=jnp.float64)
jnp_labels = jnp.array(teacher_model(dataset).detach(), dtype=jnp.float64)

def predict(w_in, w_out):
  return w_out @ jnp.transpose(
    jax_nn.sigmoid(jnp_inputs @ jnp.transpose(w_in)))

def jax_loss(w):
  network_size = int(len(w) / 3)
  w_in = w[0 : 2 * network_size].reshape(network_size, D_in)
  w_out = w[2 * network_size : ].reshape(1, network_size)
  preds = jnp.transpose(predict(w_in, w_out))
  return jnp.mean(jnp.mean(jnp.square(preds - jnp_labels)))

def hessian(f):
  return jacfwd(jacrev(f))

# Define loss objective for second order opt.
def loss_obj(weights, grad):
  loss_val = jax_loss(weights)
  if grad.size > 0:
    grad[:] = np.array(jax_grad(jax_loss)(weights), dtype=np.float64)
  return np.float64(loss_val)


# End-to-end training helper method

## First order optimization and second order optimization, together.
def train(model, x, y_labels, N = 10, Ninner = 10 ** 3, Nstart = 10,
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
    cnt = 0
    for g in loss_grad:
      g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat(
                    [g_vector, g.contiguous().view(-1)])
      cnt = 1
    print("Iteration: %d, loss: %s, gradient norm: %s" % \
        (Ninner * i, np.mean(loss_tmp), torch.norm(g_vector)))
    
    # Adjust the learning rate.
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr / (1 + i)
    
    # stopping criterion
    if i == N:
      num_neurons = model.linear1.weight.shape[0]
      weights = np.append(trace[-1][0].reshape(num_neurons * 2),
                          trace[-1][1][0].reshape(num_neurons))
      opt = nlopt.opt(nlopt.LD_SLSQP, len(weights))
      opt.set_lower_bounds([w - 10 for w in weights])
      opt.set_upper_bounds([w + 10 for w in weights])
      opt.set_min_objective(loss_obj)
      opt.set_maxtime(5)
      final_weights = opt.optimize(weights)
      return loss_vals, trace, final_weights

## Second order optimization only.
def train_second_order(model):
    num_neurons = model.linear1.weight.shape[0]
    weights = np.append(model.linear1.weight.data.detach().numpy().reshape(num_neurons * 2),
                        model.linear2.weight.data.detach().numpy().reshape(num_neurons))
    lower_bound = weights - 10
    upper_bound = weights + 10
    opt = nlopt.opt(nlopt.LD_SLSQP, len(weights))
    opt.set_lower_bounds(lower_bound)
    opt.set_upper_bounds(upper_bound)
    opt.set_min_objective(loss_obj)
    opt.set_maxtime(5)
    opt.set_xtol_rel(1e-10)
    opt.set_initial_step(1e-32)
    final_weights = opt.optimize(weights)
    return final_weights

def run_perturbation_algo(start_weights,
                          perturb_losses,
                          eps = 1e-1):
  q = deque()
  q.append((start_weights, 0))

  while len(q) > 0:
    current_weights, current_dist = q.popleft()
    perturb_losses.append(jax_loss(current_weights))

    H = hessian(jax_loss)(current_weights)
    H = (H + H.T) / 2
    evals, evectors = jnp.linalg.eigh(H)

    for idx, eval in enumerate(evals):
      if eval >= 1e-10:
        continue
      
      current_evector = evectors[:, idx]

      new_weights = current_weights + eps * current_evector
      new_dist = round(np.linalg.norm(new_weights - start_weights) / eps)
        
      if jnp.linalg.norm(jax_grad(jax_loss)(new_weights)) <= 1e-9 and \
         new_dist > current_dist:
        q.append((new_weights, new_dist))

      new_weights = current_weights - eps * current_evector
      new_dist = round(np.linalg.norm(new_weights - start_weights) / eps)

      if jnp.linalg.norm(jax_grad(jax_loss)(new_weights)) <= 1e-9 and \
         new_dist > current_dist:
        q.append((new_weights, new_dist))

def run_experiment(num_seed):
  sample_point = data.loc[num_seed]

  incoming_weights_x, incoming_weights_y, outgoing_weights = \
      extract_weights(sample_point)

  w_in = jnp.array([[incoming_weights_x[0], incoming_weights_y[0]],
                   [incoming_weights_x[1], incoming_weights_y[1]],
                   [incoming_weights_x[2], incoming_weights_y[2]],
                   [incoming_weights_x[3], incoming_weights_y[3]],
                   [incoming_weights_x[4], incoming_weights_y[4]]],
                   dtype=jnp.float64)
  w_out = jnp.array(outgoing_weights, dtype=jnp.float64)
    
  H = hessian(jax_loss)(jnp.append(w_in.reshape(D_in * H_student),
                      w_out.reshape(H_student)))
  H = (H + H.T) / 2

  evals, evectors = jnp.linalg.eigh(H)
  evals.sort()
    
  smallest_eval = evals[0]
  smallest_evector = evectors[:, jnp.argmin(evals)]

  # Perturbation (based on the smallest evector)
  old_loss = jax_loss(jnp.append(w_in.reshape(D_in * H_student),
                                 w_out.reshape(H_student)))

  perturb_lower_bound = -8.0
  perturb_upper_bound = 8.0
  perturb_step = 0.01

  perturb_losses = []
  perturb_evals = []
  perturb_grads = []

  weights = jnp.append(w_in.reshape(D_in * H_student),
                       w_out.reshape(H_student))
  for eps in np.arange(perturb_lower_bound, perturb_upper_bound,
                       perturb_step):
    new_weights = weights + eps * smallest_evector
    perturb_grads.append(jnp.linalg.norm(jax_grad(jax_loss)(new_weights)))
  
    H = hessian(jax_loss)(new_weights)
    H = (H + H.T) / 2
    evals, _ = jnp.linalg.eigh(H)

    perturb_evals.append(min(evals))
    perturb_losses.append(jax_loss(new_weights))
  
  min_perturbed_loss_1d = min(perturb_losses)
    
  # Perturbation (based on smallest evectors)
  print('before perturb')
  perturb_losses = []
  eps = 1e-1
  while len(perturb_losses) < 50 and eps >= 1e-12:
    perturb_losses = []
    run_perturbation_algo(weights, perturb_losses, eps)
    eps /= 1.2
    if eps < 1e-12:
      break
  print('after perturb')
  min_perturbed_loss_2d = min(perturb_losses)
  num_points_perturbed_loss_2d = len(perturb_losses)

  # Extract the average of the 2 neurons close to each other.

  ## Find the two closest neurons
  min_dist = np.inf
  idx_neuron1 = None
  idx_neuron2 = None

  for i in range(H_student):
    current_neuron = np.array([incoming_weights_x[i], incoming_weights_y[i]])
    for j in range(i + 1, H_student):
      potential_closest_neuron = np.array([incoming_weights_x[j],
                                           incoming_weights_y[j]])
      if min_dist > np.linalg.norm(current_neuron - potential_closest_neuron):
        min_dist = np.linalg.norm(current_neuron - potential_closest_neuron)
        idx_neuron1 = i
        idx_neuron2 = j

  pair_smallest_distance = min_dist

  min_dist = np.inf
  second_pair_idx_neuron1 = None
  second_pair_idx_neuron2 = None

  for i in range(H_student):
    if i == idx_neuron1 or i == idx_neuron2:
      continue
    current_neuron = np.array([incoming_weights_x[i], incoming_weights_y[i]])
    for j in range(i + 1, H_student):
      if j == idx_neuron1 or j == idx_neuron2:
        continue
      potential_closest_neuron = np.array([incoming_weights_x[j],
                                           incoming_weights_y[j]])
      if min_dist > np.linalg.norm(current_neuron - potential_closest_neuron):
        min_dist = np.linalg.norm(current_neuron - potential_closest_neuron)
        second_pair_idx_neuron1 = i
        second_pair_idx_neuron2 = j
    
  second_pair_smallest_distance = min_dist

  min_dist = np.inf

  for i in range(H_student):
    if i == idx_neuron1 or i == idx_neuron2:
      continue
    current_neuron = np.array([incoming_weights_x[i], incoming_weights_y[i]])
    neuron_1 = np.array([incoming_weights_x[idx_neuron1],
                         incoming_weights_y[idx_neuron1]])
    if min_dist > np.linalg.norm(current_neuron - neuron_1):
      min_dist = np.linalg.norm(current_neuron - neuron_1)

  for i in range(H_student):
    if i == second_pair_idx_neuron1 or i == second_pair_idx_neuron2:
      continue
    current_neuron = np.array([incoming_weights_x[i], incoming_weights_y[i]])
    neuron_1 = np.array([incoming_weights_x[second_pair_idx_neuron1],
                         incoming_weights_y[second_pair_idx_neuron1]])
    if min_dist > np.linalg.norm(current_neuron - neuron_1):
      min_dist = np.linalg.norm(current_neuron - neuron_1)
    
  triplet_smallest_distance = min_dist

  # Extract the average neuron from the previous 2 ones.
  new_incoming_weights_x = [(incoming_weights_x[idx_neuron1] + \
                             incoming_weights_x[idx_neuron2]) / 2.0]
  new_incoming_weights_y = [(incoming_weights_y[idx_neuron1] + \
                             incoming_weights_y[idx_neuron2]) / 2.0]
  new_outgoing_weights = [(outgoing_weights[idx_neuron1] + \
                           outgoing_weights[idx_neuron2])]

  for i in range(H_student):
    if i == idx_neuron1 or i == idx_neuron2:
      continue
    new_incoming_weights_x.append(incoming_weights_x[i])
    new_incoming_weights_y.append(incoming_weights_y[i])
    new_outgoing_weights.append(outgoing_weights[i])

  # Train the model from this point

  ## Create new NN of size 4 after reduction of 1 neuron
  w_in = torch.DoubleTensor([[new_incoming_weights_x[0],
                              new_incoming_weights_y[0]],
                           [new_incoming_weights_x[1],
                            new_incoming_weights_y[1]],
                           [new_incoming_weights_x[2],
                            new_incoming_weights_y[2]],
                           [new_incoming_weights_x[3],
                            new_incoming_weights_y[3]]])
  w_out = torch.DoubleTensor([new_outgoing_weights])
  dummy_model = DummyNetwork(D_in, H_teacher, D_out, w_in, w_out)

  final_weights = train_second_order(dummy_model)

  w_in_torch_format = torch.DoubleTensor(
                      final_weights[0 : H_teacher * 2].reshape(H_teacher, 2))
  w_out_torch_format = torch.DoubleTensor(
                      [final_weights[H_teacher * 2 :].reshape(H_teacher)])

  dummy_model = DummyNetwork(D_in, H_teacher, D_out, w_in_torch_format,
                             w_out_torch_format)
  loss_val = nn.MSELoss()(dummy_model(dataset), y_labels)

  w_in_reduced_nn = jnp.array(
      final_weights[0 : H_teacher * 2].reshape(H_teacher, 2),
      dtype=jnp.float64)
  w_out_reduced_nn = jnp.array(
      final_weights[H_teacher * 2 :].reshape(H_teacher),
      dtype=jnp.float64)

  H = hessian(jax_loss)(jnp.append(w_in_reduced_nn.reshape(D_in * H_teacher),
                                   w_out_reduced_nn))
  H = (H + H.T) / 2
  
  print('after training')

  # Eigenvalues in JAX
  evals, evectors = jnp.linalg.eigh(H)
  evals.sort()
    
  smallest_eval_reduced_nn = evals[0]
  

  local_min = np.array([incoming_weights_x[0], incoming_weights_y[0],
             incoming_weights_x[1], incoming_weights_y[1],
             incoming_weights_x[2], incoming_weights_y[2],
             incoming_weights_x[3], incoming_weights_y[3],
             incoming_weights_x[4], incoming_weights_y[4],
             outgoing_weights[0], outgoing_weights[1], outgoing_weights[2],
             outgoing_weights[3], outgoing_weights[4]])


  # Find the SI saddle line and the optimal \mu

  saddle_smallest_u = np.array([])
  teacher_idx = 1

  for i in range(H_student):
    if i == idx_neuron1 or i == idx_neuron2:
      saddle_smallest_u = np.append(saddle_smallest_u,
                                    w_in_torch_format[0][0].item())
      saddle_smallest_u = np.append(saddle_smallest_u,
                                    w_in_torch_format[0][1].item())
      continue
    saddle_smallest_u = np.append(saddle_smallest_u,
                                  w_in_torch_format[teacher_idx][0].item())
    saddle_smallest_u = np.append(saddle_smallest_u,
                                  w_in_torch_format[teacher_idx][1].item())
    teacher_idx += 1

  saddle_largest_u = deepcopy(saddle_smallest_u)

  teacher_idx = 1

  for i in range(H_student):
    if i == idx_neuron1:
      saddle_smallest_u = np.append(saddle_smallest_u,
                                    -w_out_torch_format[0][0].item())
      saddle_largest_u = np.append(saddle_largest_u,
                                   2 * w_out_torch_format[0][0].item())
      continue
    if i == idx_neuron2:
      saddle_smallest_u = np.append(saddle_smallest_u,
                                    2 * w_out_torch_format[0][0].item())
      saddle_largest_u = np.append(saddle_largest_u,
                                   -w_out_torch_format[0][0].item())
      continue
    saddle_smallest_u = np.append(saddle_smallest_u,
                                w_out_torch_format[0][teacher_idx].item())
    saddle_largest_u = np.append(saddle_largest_u,
                               w_out_torch_format[0][teacher_idx].item())
    teacher_idx += 1

  saddle_line_dir = (saddle_largest_u - saddle_smallest_u) / \
                    np.linalg.norm(saddle_largest_u - saddle_smallest_u)
  closest_saddle = saddle_smallest_u + saddle_line_dir * \
                    np.dot(saddle_line_dir, (local_min - saddle_smallest_u))
  optimal_mu = closest_saddle[2 * H_student + idx_neuron1] / w_out_torch_format[0][0].item()
  l2_saddle_min = np.linalg.norm(local_min - closest_saddle)

  # Compute loss across the 1D line between local min and closest saddle


  return [num_seed, old_loss, loss_val.item(), smallest_eval,
          smallest_eval_reduced_nn, l2_saddle_min, optimal_mu,
         min_perturbed_loss_1d, min_perturbed_loss_2d,
         num_points_perturbed_loss_2d, pair_smallest_distance,
         second_pair_smallest_distance, triplet_smallest_distance]


file_experiment_header = ['seed', 'loss', 'loss_reduced_nn', 'smallest_eval',
                          'smallest_eval_reduced_nn', 'l2_saddle_min',
                          'optimal_mu', 'min_perturbed_loss_1d',
                          'min_perturbed_loss_2d', 'num_points_perturbed_loss_2d',
                          'pair_smallest_dist', 'second_pair_smallest_dist',
                          'triplet_smallest_dist']

file_experiment_data = open('si_saddles_statistics.csv', 'w')
writer = csv.writer(file_experiment_data)
writer.writerow(file_experiment_header)

for i in range(len(data)):
  print(i)
  result = run_experiment(i)
  writer.writerow(result)

file_experiment_data.close()
