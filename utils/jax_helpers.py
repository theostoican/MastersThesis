import jax.numpy as jnp
from jax import nn as jax_nn
from jax.config import config; config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=32) 

from jax import jacfwd, jacrev
from jax import grad
import jax

import torch

from utils.datasets import construct_toy_dataset

def get_jnp_toy_dataset():
    dataset, labels = construct_toy_dataset()

    jnp_inputs = jnp.array(dataset, dtype=jnp.float64)
    jnp_labels = jnp.array(labels, dtype=jnp.float64)
    
    return jnp_inputs, jnp_labels

def predict(w_in, w_out):
    jnp_inputs, jnp_labels = get_jnp_toy_dataset()
    return w_out @ jnp.transpose(jax_nn.sigmoid(jnp_inputs @ jnp.transpose(w_in)))

def jax_loss(w):
    jnp_inputs, jnp_labels = get_jnp_toy_dataset()
    network_size = int(len(w) / 3)
    w_in = w[0 : 2 * network_size].reshape(network_size, 2)
    w_out = w[2 * network_size : ].reshape(1, network_size)
    preds = jnp.transpose(predict(w_in, w_out))
    return jnp.mean(jnp.square(preds - jnp_labels))

def jax_grad(w):
    return grad(jax_loss)(w)

def jax_hessian(w):
    H = jacfwd(jacrev(jax_loss))(w)
    H = (H + H.T) / 2
    return H

def jax_evals(w):
    H = jax_hessian(w)
    evals, evectors = jnp.linalg.eigh(H)
    
    return evals, evectors