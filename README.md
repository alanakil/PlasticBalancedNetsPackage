<!-- ![Build Status](https://travis-ci.com/alanakil/PlasticBalancedNetsPackage.svg?branch=main) -->
<!-- ![codecov](https://codecov.io/gh/alanakil/PlasticBalancedNetsPackage/branch/main/graph/badge.svg) -->
[![PyPI version](https://badge.fury.io/py/plastic-balanced-network.svg)](https://badge.fury.io/py/plastic-balanced-network)
![Python version](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![GitHub pull requests](https://img.shields.io/github/issues-pr/alanakil/PlasticBalancedNetsPackage)
[![Downloads](https://pepy.tech/badge/plastic-balanced-network)](https://pepy.tech/project/plastic-balanced-network)
[![Downloads](https://static.pepy.tech/badge/plastic-balanced-network/month)](https://pepy.tech/project/plastic-balanced-network)
[![Downloads](https://static.pepy.tech/badge/plastic-balanced-network/week)](https://pepy.tech/project/plastic-balanced-network)
<!-- ![Conda Version](https://img.shields.io/conda/vn/conda-forge/PlasticBalancedNetsPackage) -->
<!-- ![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/PlasticBalancedNetsPackage.svg) -->
[![Docs Status](https://github.com/alanakil/PlasticBalancedNetsPackage/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/alanakil/PlasticBalancedNetsPackage/actions?query=workflow%3ADocs)
[![Build and Test](https://github.com/alanakil/PlasticBalancedNetsPackage/actions/workflows/ci.yml/badge.svg)](https://github.com/alanakil/PlasticBalancedNetsPackage/actions/workflows/ci.yml)

# PlasticBalancedNetsPackage

The package `plastic_balanced_network` can be used to simulate spiking neural networks in the balance regime and undergoing synaptic plasticity on any cell type pair. 

There is a great deal of flexibility to simulate a network with any combination of the following parameters:

(1) Total number of neurons.

(2) Fraction of E-I neurons.

(3) Probability of connection.

(4) Synaptic strengths.

(5) Total time of simulation.

(6) Input rate and correlations.

(7) Extra injected current.

(8) EIF neuron parameters.

(9) Plasticity parameters on any connection type and plasticity type (Hebbian, Kohonen, homeostatic inhibitory plasticities).

Installation: Run `pip install plastic_balanced_network` in the terminal. 
Alternatively, clone the repo and run `pip install -e .`

Import: `from plastic_balanced_network.network import PlasticNeuralNetwork`

You may also import other useful functions for analysis: `from plastic_balanced_network.helpers import compute_firing_rate, spike_count_cov, cov2corr, average_cov_corr_over_subpops`

Documentation: https://alanakil.github.io/PlasticBalancedNetsPackage/

Research Article: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008958

## Original Simulations
In addition to the packaged neural network, we also make available the original simulations of plastic balanced networks that were run in MATLAB. The results of these simulations were reported in Akil et al. 2021 ("Balanced networks under spike-timing dependent plasticity"). The exact same network is used in the MATLAB simulations and the Python package presented here.
Link to paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008958

We provide the MATLAB and Python code that was used for all simulations in Akil et al., 2021 in the folder named `original_simulations`.
In this code, we run several realizations of plastic balanced networks with varying: 

- Network size. To compare with theoretical predictions of rates, covariances, and synaptic weights.

- Input correlations. To assess the impact of increasing correlations in synaptic weights and rates.

- Initial connectivity. To show the emergence of a manifold of fixed points in weight space when only I->E synapses are plastic.

Please see more details in the paper Akil et al., 2021.

This codebase was developed by Robert Rosenbaum and Alan Akil and is currently maintained by Alan Akil. 
