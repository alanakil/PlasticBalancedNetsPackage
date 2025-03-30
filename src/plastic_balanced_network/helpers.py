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

    Parameters
    ----------
    s: np.ndarray
        Spike trains of all neurons.
    T: int
        Total simulation time in milliseconds.
    N: int
        Total number of neurons.
    frac_exc: float
        Fraction of E neurons. Defaults to 0.8.
    dtRate: int
        Size of time bin to count spikes over. Defaults to 10 ms.
    window_size: int
        Size of window for moving average. Defaults to 10 bins.

    Returns
    ----------
    eRateT: np.ndarray
        Smoothed time varying firing rate of E neurons.
    iRateT: np.ndarray
        Smoothed time varying firing rate of I neurons.
    timeVector: np.ndarray
        Discretized time domain.
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
    logging.info(f"Time for computing rates: {round(elapsed_time / 60, 2)} minutes.")

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

    Parameters
    ----------
    s: np.ndarray
        Spike trains of all neurons.
    N: int
        Total number of neurons.
    T1: float or int
        Start time to count spikes for covariance calculation.
    T2: float or int
        End time to count spikes for covariance calculation.
    winsize: int
        Time window over which spikes are counted. Defaults to 250 ms.

    Returns
    ----------
    C: np.ndarray
        Full spike count covariance matrix.
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
    logging.info(f"Time for computing covariance matrix: {round(elapsed_time / 60, 2)} minutes.")

    return C


def cov2corr(cov):
    """convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov: np.ndarray
        Covariance matrix.

    Returns
    ----------
    corr: np.ndarray
        Correlation matrix.
    """
    start_time = time.time()
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)

    elapsed_time = time.time() - start_time
    logging.info(
        f"Time for converting covariance matrix into correlation matrix: {round(elapsed_time / 60, 2)} minutes."
    )
    return corr


def average_cov_corr_over_subpops(C, N, frac_exc=0.8):
    """
    Average covariances or correlations over subpopulations.

    Parameters
    ----------
    C: np.ndarray
        Matrix of covariances or correlations.
    N: int
        Total number of neurons.
    frac_exc: float
        Fraction of E neurons. Defaults to 0.8.

    Returns
    -------
    mC: np.ndarray
        Mean spike count covariance or correlation matrix.
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
    logging.info(f"Time for averaging corrs or covs: {round(elapsed_time / 60, 2)} minutes.")
    return mC
