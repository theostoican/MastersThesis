import jax.numpy as jnp
from jax import nn as jax_nn
from jax.config import config; config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=32)

import torch
import torch.nn as nn

import numpy as np

def construct_dataset():
    data = []
    for y in np.arange(-5, 5.1, .25):
        for x in np.arange(-5, 5.1, .25):
            data.append([x, y])
    return data

def find_closest_neurons(w_in):
    min_dist = np.inf
    idx_neuron1 = None
    idx_neuron2 = None

    for i in range(len(w_in)):
      current_neuron = np.array(w_in[i])
      for j in range(i + 1, len(w_in)):
        potential_closest_neuron = np.array(w_in[j])
        if min_dist > np.linalg.norm(current_neuron - potential_closest_neuron):
          min_dist = np.linalg.norm(current_neuron - potential_closest_neuron)
          idx_neuron1 = i
          idx_neuron2 = j
        
    return idx_neuron1, idx_neuron2

class TeacherNetwork(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        D_in: input dimension
        H: dimension of hidden layer
        D_out: output dimension of the first layer
        """
        super(TeacherNetwork, self).__init__()
        self.linear1 = nn.Linear(2, 4, bias=False)
        self.linear2 = nn.Linear(4, 1, bias=False)
        self.linear1.weight = torch.nn.Parameter(torch.transpose(
            torch.DoubleTensor([[0.6, -0.5, -0.2, 0.1],
                                [0.5, 0.5, -0.6, -0.6]]), 0, 1))
        self.linear2.weight = torch.nn.Parameter(torch.transpose(
            torch.DoubleTensor([[1], [-1], [1], [-1]]), 0, 1))
    def forward(self, x):
        h_sigmoid = torch.sigmoid(self.linear1(x))
        y_pred = self.linear2(h_sigmoid)
        return y_pred

class TheoremChecker:
    def __init__(self, D_in, H_teacher, D_out):
        self.D_in = D_in
        self.H_teacher = H_teacher
        self.D_out = D_out
        
        teacher_model = TeacherNetwork()
        dataset = construct_dataset()
        
        self.jnp_inputs = jnp.array(dataset, dtype=jnp.float64)
        self.jnp_labels = jnp.array(teacher_model(
            torch.DoubleTensor(dataset)).detach(),
                                    dtype=jnp.float64)

    def predict(self, w_in, w_out):
        return w_out @ jnp.transpose(jax_nn.sigmoid(self.jnp_inputs @ jnp.transpose(w_in)))

    def check_si_saddle_failure_mode(self, w):
        network_size = int(len(w) / 3)
        w_in = w[0 : 2 * network_size].reshape(network_size, self.D_in)
        w_out = w[2 * network_size : ].reshape(1, network_size)
        preds = jnp.transpose(self.predict(w_in, w_out))
        
        e = preds - self.jnp_labels
        
        first_derivative_sigmoid = lambda x : jax_nn.sigmoid(x) * \
                                              (1 - jax_nn.sigmoid(x))
        second_derivative_sigmoid = lambda x : jax_nn.sigmoid(x) * \
                                               ((1 - jax_nn.sigmoid(x)) ** 2)-\
                                               (jax_nn.sigmoid(x) ** 2) *\
                                               (1 - jax_nn.sigmoid(x))
        
        neuron_to_be_duplicated = 0
        Y = jnp.zeros((2, 2))
        
        for idx, x in enumerate(self.jnp_inputs):
            Y += w_out[0][0] * second_derivative_sigmoid(jnp.dot(x, w_in[0])) * e[idx][0] * \
                 jnp.array(x).reshape(len(x), 1) @ jnp.array(x).reshape(1, len(x))

        Y /= len(self.jnp_inputs)
        
        U00 = 0
        U01 = 0
        
        for idx, x in enumerate(self.jnp_inputs):
            U00 += first_derivative_sigmoid(jnp.dot(x, w_in[0])) * e[idx][0] * x[0]
            U01 += first_derivative_sigmoid(jnp.dot(x, w_in[0])) * e[idx][0] * x[1]
            
        U00 /= len(self.jnp_inputs)
        U01 /= len(self.jnp_inputs)
        
        
        return Y, np.array([U00, U01])
