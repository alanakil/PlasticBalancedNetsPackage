# PlasticBalancedNetsPackage

Here we provide code for a series of simulations of plastic balanced networks. The results were reported in Akil et al 2021 ("Balanced Networks Under Spike-Timing Dependent Plasticity"). Link to preprint: https://www.biorxiv.org/content/10.1101/2020.04.26.061515v1

The code used here to simulate a balanced network is almost the same as in https://github.com/alanakil/BalancedNetworkPackage. Please refer to that Package for more details on how simulations were run along with details about the neuron model, initial connectivity, etc.

We provide code to simulate balanced networks undergoing a number of STDP rules: Kohonen, Oja, classical Hebbian, and inhibitory STDP (as in Vogels et al 2011). 

In addition to that, we provide code where we run several realizations of plastic balanced networks with varying: 

- Network size. To compare with theoretical predictions of rates, covariances, and synaptic weights.

- Input correlations. To assess the impact of increasing correlations in synaptic weights and rates.

- Initial connectivity. To show the emergence of a manifold of fixed points in weight space when only I->E synapses are plastic.

Lastly, we also provided code that simulates optogenetic activation of subset of E neurons in a plastic balanced network.

