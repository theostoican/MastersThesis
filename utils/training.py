import numpy as np

import torch
import torch.nn as nn

import nlopt

from jax.config import config; config.update("jax_enable_x64", True)
from jax import grad as jgrad
import jax.numpy as jnp

from utils.jax_helpers import jax_loss, jax_grad, predict
from utils.utils import find_closest_neurons
from utils.datasets import construct_toy_dataset


def train_first_order(model, x, y_labels, N = 10, Ninner = 10 ** 3, Nstart = 10,
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
        g_vector = g.contiguous().view(-1) \
            if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    print("Iteration: %d, loss: %s, gradient norm: %s" % \
          (Ninner * i, np.mean(loss_tmp), torch.norm(g_vector)))
    
    # Adjust the learning rate.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1 + i)
    
    num_neurons = model.linear1.weight.shape[0]
    weights = np.append(trace[-1][0].reshape(num_neurons * 2),
                        trace[-1][1][0].reshape(num_neurons))

    return loss_vals, trace, weights

def second_order_loss_obj(weights, grad):
  loss_val = jax_loss(weights)
  if grad.size > 0:
    grad[:] = np.array(jax_grad(weights), dtype=np.float64)
  return np.float64(loss_val)

def train_second_order(model):
    num_neurons = model.linear1.weight.shape[0]
    weights = np.append(model.linear1.weight.data.detach().numpy().reshape(num_neurons * 2),
                        model.linear2.weight.data.detach().numpy().reshape(num_neurons))
    lower_bound = weights - 10
    upper_bound = weights + 10
    opt = nlopt.opt(nlopt.LD_SLSQP, len(weights))
    opt.set_lower_bounds(lower_bound)
    opt.set_upper_bounds(upper_bound)
    opt.set_min_objective(second_order_loss_obj)
    opt.set_maxtime(100)
    opt.set_xtol_rel(1e-20)
    opt.set_initial_step(1e-32)
    final_weights = opt.optimize(weights)
    return final_weights

def perturbed_train_second_order(model):
    num_neurons = model.linear1.weight.shape[0]
    weights = np.append(model.linear1.weight.data.detach().numpy().reshape(num_neurons * 2),
                        model.linear2.weight.data.detach().numpy().reshape(num_neurons))

    mu, sigma = 0, 0.5 # mean and standard deviation of the sample Gaussian
    num_perturbations = 1
    
    perturbed_losses = []
    perturbed_grads = []
    perturbed_final_weights = []
    for i in range(num_perturbations):
        perturbation = np.random.normal(mu, sigma, len(weights))
        perturbed_weights = weights + perturbation
        lower_bound = perturbed_weights - 10
        upper_bound = perturbed_weights + 10
        opt = nlopt.opt(nlopt.LD_SLSQP, len(perturbed_weights))
        opt.set_lower_bounds(lower_bound)
        opt.set_upper_bounds(upper_bound)
        opt.set_min_objective(second_order_loss_obj)
        opt.set_maxtime(1)
        opt.set_xtol_rel(1e-10)
        opt.set_initial_step(1e-32)
        final_weights = opt.optimize(perturbed_weights)
        perturbed_losses.append(jax_loss(final_weights))
        perturbed_grads.append(jnp.linalg.norm(jax_grad(final_weights)))
        perturbed_final_weights.append(final_weights)
    return perturbed_losses, perturbed_grads, perturbed_final_weights

def regularized_loss(weights, idx_neuron1, idx_neuron2):
    network_size = int(len(weights) / 3)
    w_in = weights[0 : 2 * network_size].reshape(network_size, 2)
    w_out = weights[2 * network_size : ].reshape(network_size)
    # neuron_1 = jnp.append(w_in[idx_neuron1, :], w_out[idx_neuron1])
    # neuron_2 = jnp.append(w_in[idx_neuron2, :], w_out[idx_neuron2])
    # neuron_1 = w_in[idx_neuron1, :]
    # neuron_2 = w_in[idx_neuron2, :]

    jnp_inputs, jnp_labels = construct_toy_dataset()
    preds = np.transpose(predict(jnp.array(w_in), jnp.array(w_out)))
    return jnp.mean(jnp.square(preds - jnp_labels)) + \
         2 * jnp.linalg.norm(w_in[idx_neuron1, :] - w_in[idx_neuron2, :]) + \
         2 * jnp.linalg.norm(w_out[idx_neuron1] - w_out[idx_neuron2])

def regularized_loss_obj(weights, grad):
    network_size = int(len(weights) / 3)
    w_in = weights[0 : 2 * network_size].reshape(network_size, 2)
    w_out = weights[2 * network_size : ].reshape(network_size)
    idx_neuron1, idx_neuron2 = find_closest_neurons(w_in)

    loss_val = regularized_loss(jnp.array(weights),
                                idx_neuron1, idx_neuron2)
    if grad.size > 0:
        grad[:] = np.array(jgrad(regularized_loss)(jnp.array(weights),
                                      idx_neuron1, idx_neuron2),
              dtype=np.float64)
    return np.float64(loss_val)

def regularized_second_order_train(weights):
    lower_bound = weights - 15
    upper_bound = weights + 15
    opt = nlopt.opt(nlopt.LD_SLSQP, len(weights))
    opt.set_lower_bounds(lower_bound)
    opt.set_upper_bounds(upper_bound)
    opt.set_min_objective(regularized_loss_obj)
    opt.set_maxtime(100)
    opt.set_xtol_rel(1e-10)
    opt.set_initial_step(1e-32)
    final_weights = opt.optimize(weights)
    return final_weights

class BSolver:
    def __init__(self, third_order_derivs, H, zero_evector):
        self.third_order_derivs = third_order_derivs
        self.H = H
        self.zero_evector = zero_evector
    
    def compute_B_loss(self, u):
#         s = 0
#         for i in range(self.third_order_derivs.shape[0]):
#             for j in range(self.third_order_derivs.shape[1]):
#                 for k in range(self.third_order_derivs.shape[2]):
#                     s += self.third_order_derivs[i][j][k] * u[i] * u[j] * u[k]

        return jnp.abs((u @ self.third_order_derivs) @ u @ u)
        
    def compute_B_loss_obj(self, u, grad):
        loss_val = self.compute_B_loss(u) 
        if grad.size > 0:
            grad[:] = np.array(jgrad(self.compute_B_loss)(u),
                  dtype=np.float64)
        return np.float64(loss_val)

    def compute_B_constraint_norm_pos(self, u, grad):
        constraint_loss = lambda x : jnp.linalg.norm(x) - 1
        loss_val = constraint_loss(u)
        if grad.size > 0:
            grad[:] =  np.array(u / jnp.linalg.norm(u),
                  dtype=np.float64)
        return np.float64(loss_val)
    
    def compute_B_constraint_hessian(self, u, grad):
        constraint_loss = lambda x : x.T @ self.H @ x
        loss_val = constraint_loss(u)
        if grad.size > 0:
            grad[:] =  np.array(jgrad(constraint_loss)(u),
                  dtype=np.float64)
        return np.float64(loss_val)
    
    def compute_B_constraint_orthogonal(self, u, grad):
        constraint_loss = lambda x : jnp.dot(x, self.zero_evector)
        loss_val = constraint_loss(u)
        if grad.size > 0:
            grad[:] =  np.array(jgrad(constraint_loss)(u),
                  dtype=np.float64)
        return np.float64(loss_val)

    def compute_B(self, u):
        lower_bound = u - 15
        upper_bound = u + 15
        opt = nlopt.opt(nlopt.LD_SLSQP, len(u))
        opt.set_lower_bounds(lower_bound)
        opt.set_upper_bounds(upper_bound)
        opt.set_min_objective(self.compute_B_loss_obj)
        opt.set_maxtime(100)
        opt.set_xtol_rel(1e-10)
        opt.set_initial_step(1e-32)
        opt.add_equality_constraint(lambda x, grad: self.compute_B_constraint_norm_pos(x, grad), 0)
        opt.add_equality_constraint(lambda x, grad: self.compute_B_constraint_orthogonal(x, grad), 0)
        opt.add_inequality_constraint(lambda x, grad: self.compute_B_constraint_hessian(x, grad), 0)

        final_weights = opt.optimize(u)
        return final_weights
    
    # def find_maximum_by_perturbation(self, u):
        
class FourthOrderSolver:
    def __init__(self, fourth_order_derivs, H, zero_evector):
        self.fourth_order_derivs = fourth_order_derivs
        self.H = H
        self.zero_evector = zero_evector
    
    def constraint_norm_pos(self, u, grad):
        constraint_loss = lambda x : jnp.linalg.norm(x) - 1
        loss_val = constraint_loss(u)
        if grad.size > 0:
            grad[:] =  np.array(u / jnp.linalg.norm(u),
                  dtype=np.float64)
        return np.float64(loss_val)
    
    def constraint_hessian(self, u, grad):
        constraint_loss = lambda x : x.T @ self.H @ x
        loss_val = constraint_loss(u)
        if grad.size > 0:
            grad[:] =  np.array(jgrad(constraint_loss)(u),
                  dtype=np.float64)
        return np.float64(loss_val)
    
    def constraint_orthogonal(self, u, grad):
        constraint_loss = lambda x : jnp.dot(x, self.zero_evector)
        loss_val = constraint_loss(u)
        if grad.size > 0:
            grad[:] =  np.array(jgrad(constraint_loss)(u),
                  dtype=np.float64)
        return np.float64(loss_val)
    
    def compute_loss(self, u):
        return (u @ self.fourth_order_derivs) @ u @ u @ u
    
    def compute_loss_obj(self, u, grad):
        loss_val = self.compute_loss(u) 
        if grad.size > 0:
            grad[:] = np.array(jgrad(self.compute_loss)(u),
                  dtype=np.float64)
            print(grad)
        return np.float64(loss_val)
    
    def compute_min(self, u):
        lower_bound = u - 5
        upper_bound = u + 5
        opt = nlopt.opt(nlopt.LD_SLSQP, len(u))
        opt.set_lower_bounds(lower_bound)
        opt.set_upper_bounds(upper_bound)
        opt.set_min_objective(self.compute_loss_obj)
        opt.set_maxtime(100)
        opt.set_xtol_rel(1e-10)
        opt.set_initial_step(1e-32)
        opt.add_equality_constraint(lambda x, grad: self.constraint_norm_pos(x, grad), 0)
        opt.add_equality_constraint(lambda x, grad: self.constraint_orthogonal(x, grad), 0)
        opt.add_inequality_constraint(lambda x, grad: self.constraint_hessian(x, grad), 0)

        final_weights = opt.optimize(u)
        return final_weights