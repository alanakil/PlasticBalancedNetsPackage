"""
Functions to analyze activity of plastic balanced network.
"""
__author__ = "Alan Akil (alan.akil@yahoo.com)"
__date__ = "APRIL 2023"

# %%
import numpy as np
import time
import logging

# %%


def compute_firing_rate(s, T, N, frac_exc=0.8, dtRate=10, window_size=10):
    """
    Calculate the mean firing rate of E and I populations as a function of time.
    Arguments
    :param s: Matrix of covariances or correlations.
    :type s: np.ndarray
    :param T: Total time of simulation.
    :type T: int
    :param N: Total number of neurons.
    :type N: int
    :param frac_exc: Fraction of E neurons. Defaults to 0.8.
    :type frac_exc: float
    :param dtRate: Size of time bin to count spikes over. Defaults to 10 ms.
    :type dtRate: int
    :param windowsize: Size of window for moving average. Defaults to 10 bins.
    :type windowsize: int

    Returns
    :return: Time varying firing rate of E and I neurons, respectively (eRateT, iRateT).
    :rtype: tuple(np.ndarray,np.ndarray)
    """
    start_time = time.time()
    timeVector = np.arange(dtRate, T + dtRate, dtRate) / 1000

    hist, bin_edges = np.histogram(s[0, s[1, :] < frac_exc * N], bins=len(timeVector))
    eRateT = hist / (dtRate * frac_exc * N) * 1000
    hist, bin_edges = np.histogram(s[0, s[1, :] >= frac_exc * N], bins=len(timeVector))
    iRateT = hist / (dtRate * (1 - frac_exc) * N) * 1000

    # Smooth rates. We multiplied by 1000 to get them in units of Hz.
    eRateT = np.convolve(eRateT, np.ones(window_size) / window_size, mode="same")
    iRateT = np.convolve(iRateT, np.ones(window_size) / window_size, mode="same")

    elapsed_time = time.time() - start_time
    logging.info(f"Time for computing rates: {round(elapsed_time/60,2)} minutes.")

    return eRateT, iRateT, timeVector


# %%


def spike_count_cov(s, N, T1, T2, winsize=250):
    """
    Compute NxN spike count covariance matrix.
    s is a 2x(ns) matrix where ns is the number of spikes
    s(0,:) lists spike times
    and s(1,:) lists corresponding neuron indices
    Neuron indices are assumed to go from 0 to N-1

    Spikes are counts starting at time T1 and ending at
    time T2.

    winsize is the window size over which spikes are counted,
    so winsize is assumed to be much smaller than T2-T1

    Arguments
    :param s: Spike trains of all neurons.
    :type s: np.ndarray
    :param N: Total number of neurons
    :type N: int
    :param T1: Start time to count spikes for covariance calculation.
    :type T1: float or int
    :param T2: End time to count spikes for covariance calculation.
    :type T2: float or int
    :param winsize: Time window over which spikes are counted. Defaults to 250 ms.
    :type winsize: int

    Returns (as part of `self`)
    :return C: Full spike count covariance matrix.
    :rtype C: np.ndarray
    """
    start_time = time.time()
    #   Count only spikes between T1, T2
    s1 = s[:, (s[0, :] <= T2) & (s[1, :] >= T1)]

    #   Count only for neurons between 0, N
    s1 = s[:, (s[1, :] < N) & (s[1, :] >= 0)]

    #   Edges for histogram
    edgest = np.arange(T1, T2, winsize)
    edgesi = np.arange(0, N + 1)

    #   Get 2D histogram of spike indices and times
    counts, xedges, yedges = np.histogram2d(s1[0, :], s1[1, :], bins=(edgest, edgesi))
    #   Compute and return covariance N x N matrix
    C = np.array(np.cov(counts.transpose()))

    elapsed_time = time.time() - start_time
    logging.info(f"Time for computing covariance matrix: {round(elapsed_time/60,2)} minutes.")

    return C


def cov2corr(cov):
    """convert covariance matrix to correlation matrix

    Arguments
    :param cov: Covariance matrix.
    :type cov: np.ndarray

    Returns (as part of `self`)
    :return corr: Full spike count correlation matrix.
    :rtype corr: np.ndarray
    """
    start_time = time.time()
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)

    elapsed_time = time.time() - start_time
    logging.info(f"Time for converting covariance matrix into correlation matrix: {round(elapsed_time/60,2)} minutes.")
    return corr


def average_cov_corr_over_subpops(C, N, frac_exc=0.8):
    """
    Average covariances or correlations over subpopulations.
    Arguments
    :param C: Matrix of covariances or correlations.
    :type C: np.ndarray
    :param N: Total number of neurons.
    :type N: int
    :param frac_exc: Fraction of E neurons. Defaults to 0.8.
    :type frac_exc: float

    Returns
    :return mC: Mean spike count covariance matrix.
    :rtype mC: np.ndarray
    """
    start_time = time.time()
    # Get mean spike count covariances over each sub-pop
    II, JJ = np.meshgrid(np.arange(0, N), np.arange(0, N))
    mCee = np.nanmean(C[(II < frac_exc * N) & (JJ < II)])
    mCei = np.nanmean(C[(II < frac_exc * N) & (JJ >= frac_exc * N)])
    mCii = np.nanmean(C[(II > frac_exc * N) & (JJ > II)])

    # Mean-field spike count cov matrix
    # Compare this to the theoretical prediction
    mC = [[mCee, mCei], [mCei, mCii]]

    elapsed_time = time.time() - start_time
    logging.info(f"Time for averaging corrs or covs: {round(elapsed_time/60,2)} minutes.")
    return mC
