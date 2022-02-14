# An Empirical Investigation of the Failure Mode of Training in Mildly Overparameterized NNs

For a successful training of a NN to a global minimum, the optimization algorithm should avoid getting trapped at a local minimum or a non-strict saddle. Recent work has shown that in very wide NNs, the optimization algorithm converges to a global minimum reliably. We identified a regime where the network is just expressive enough to achieve a zero-loss point. However, it fails to do so for a fraction of initializations. We are looking for a student who will investigate the failure mode of training. Is it due to (an abundance of) local minima, or is it due to non-strict saddles? We will provide a codebase in Julia at the beginning of this project. Some experience in coding and NN training are the prerequisites.

Seed papers:


[1] Şimşek Berfin et al., 2021, Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances

[2] Safran Itay, Shamir Ohad, 2017, Spurious Local Minima are Common in Two-Layer ReLU Neural Networks

[3] Zhang Yaoyu, 2021, Embedding Principle of Loss Landscape of Deep Neural Networks

