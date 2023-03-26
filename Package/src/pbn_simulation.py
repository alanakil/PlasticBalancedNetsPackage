"""
This code simulates a plastic balanced network as defined in Akil et al, 2021. 
We build this as a Python package and allow for a great deal of flexibility constructing
the simulation. One may change (1) Synaptic weights, (2) Type of plasticity, 
(3) Connectivity, (4) Level of induced correlations, ...
Output data is saved in the `data/processed` folder. 
Please refer to the README.md under `Package` for detailed instructions on how to run this code.
"""
__author__ = "Alan Akil (alan.akil@yahoo.com)"
__date__ = "MARCH 2023"

#%%
# Load python packages.
import numpy as np
import random2
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from datetime import datetime as dtm
import datetime as dt
import os
from pathlib import Path

from plastic_balanced_network.helpers import plasticNeuralNetwork

#%%

PROJECTROOT = Path(__file__).parent.parent

DATA_DIR = os.path.join(PROJECTROOT, "data", "processed")
LOG_DIR = os.path.join(PROJECTROOT, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


#%%
todaysdate = dtm.today()
datetime_format = "%Y%b%d-%H%M"
datadatetime = todaysdate.strftime(datetime_format).upper()

start_time = time.time()

log_format = (
    "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
)
loglevel = "INFO"
loglevel = str(loglevel).replace('"', "")
levels = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARN,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
level = levels.get(loglevel)

_ = [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"pbn_{datadatetime}.log"),
    filemode="w",
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
    level=level,
)

logging.info("Start simulation of plastic balanced network.")

logging.info(f"Project root: {PROJECTROOT}.")


#%%
# Number of neurons in each population
N = int(5000)
frac_exc = 0.8
frac_ext = 0.2

# Define individual connection probabilities
p_ee = 0.1
p_ei = 0.1
p_ie = 0.1
p_ii = 0.1
p_ex = 0.1
p_ix = 0.1
# Recurrent net connection probabilities
P = np.array([[p_ee, p_ei], [p_ie, p_ii]])
# Ffwd connection probs
Px = np.array([[p_ex], [p_ix]])

# Timescale of correlation in ms
taujitter = 5
# Mean connection strengths between each cell type pair
jee = 25
jei = -150
jie = 112.5
jii = -250
jex = 180
jix = 135
Jm = np.array([[jee, jei], [jie, jii]]) / np.sqrt(N)
Jxm = np.array([[jex], [jix]]) / np.sqrt(N)

# Total_time (in ms) for sim
T = 5000

# Total_time discretization
dt = 0.1

# FFwd spike train rate (in kHz)
rx = 10 / 1000

# Extra stimulus: Istim is a Total_time-dependent stimulus
# it is delivered to all neurons with weights given by JIstim.
# Specifically, the stimulus to neuron j at Total_time index i is:
# Istim(i)*JIstim(j)
jestim = 0
jistim = 0

# Synaptic timescales in ms
taux = 10
taue = 8
taui = 4

# Generate FFwd spike trains
# Correlation of ffwd spike trains.
c = 0

# Neuron parameters
Cm = 1
gL = 1 / 15
EL = -72
Vth = -50
Vre = -75
DeltaT = 1
VT = -55

# Plasticity parameters
tauSTDP = 200  # ms

# EE hebb
Jmax_ee = 30 / np.sqrt(N)
eta_ee_hebb = 0 / 10**3  # Learning rate

# EE kohonen
beta = 2 / np.sqrt(N)
eta_ee_koh = 0 / 10**2  # Learning rate

# IE hebb
Jmax_ie_hebb = 125 / np.sqrt(N)
eta_ie_hebb = 0 / 10**3  # Learning rate

# IE homeo
Jnorm_ie = 200 / np.sqrt(N)
eta_ie_homeo = 0 / 10**3 / Jnorm_ie  # Learning rate
rho_ie = 0.020  # Target rate 20Hz
alpha_ie = 2 * rho_ie * tauSTDP

# EI homeo
Jnorm_ei = -200 / np.sqrt(N)
eta_ei = 0.015 / 10**3 / Jnorm_ei  # Learning rate
rho_ei = 0.010  # Target rate 10Hz
alpha_ei = 2 * rho_ei * tauSTDP

# II
Jnorm_ii = -300 / np.sqrt(N)
eta_ii = 0 / 10**3 / Jnorm_ii  # Learning rate
rho_ii = 0.020  # Target rate 20Hz
alpha_ii = 2 * rho_ii * tauSTDP

# Indices of neurons to record currents, voltages
numrecord = int(100)  # Number to record from each population
Irecord = np.array(
    [
        [
            random2.sample(list(np.arange(0, frac_exc * N)), numrecord),
            random2.sample(list(np.arange(frac_exc * N, N)), numrecord),
        ]
    ]
)
Ierecord = np.sort(Irecord[0, 0]).astype(int)
Iirecord = np.sort(Irecord[0, 1]).astype(int)
Ixrecord = np.sort(random2.sample(list(np.arange(0, frac_ext * N)), numrecord)).astype(
    int
)
Vrecord = (
    np.sort(
        [
            [
                random2.sample(
                    list(np.arange(0, frac_exc * N)), int(round(numrecord / 2))
                ),
                random2.sample(
                    list(np.arange(frac_exc * N, N)), int(round(numrecord / 2))
                ),
            ]
        ]
    )[0]
    .reshape(1, numrecord)
    .astype(int)[0]
)
del Irecord

# Number of time bins to average over when recording
nBinsRecord = 10
dtRecord = nBinsRecord * dt
# Number of synapses to be sampled
nJrecord0 = 1000

#%%
# Define the model.
nn = plasticNeuralNetwork(
    N,
    frac_exc,
    frac_ext,
    P,
    Px,
    taujitter,
    Jm,
    Jxm,
    T,
    dt,
    rx,
    jestim,
    jistim,
    taue,
    taui,
    taux,
    c,
    Cm,
    gL,
    EL,
    Vth,
    Vre,
    DeltaT,
    VT,
    tauSTDP,
    Jmax_ee,
    eta_ee_hebb,
    eta_ee_koh,
    beta,
    eta_ie_hebb,
    Jmax_ie_hebb,
    eta_ie_homeo,
    alpha_ie,
    alpha_ei,
    eta_ei,
    alpha_ii,
    eta_ii,
    nBinsRecord,
    dtRecord,
    Ierecord,
    Iirecord,
    Ixrecord,
    Vrecord,
    numrecord,
    nJrecord0,
)

#%%
# Initialize the connectivity
nn.connectivity()

#%%
# Generate Poisson ffwd spike trains
nn.ffwd_spikes()

#%%
# Simulate plastic network
(
    s,
    sx,
    JRec_ee,
    JRec_ie,
    JRec_ei,
    JRec_ii,
    IeRec,
    IiRec,
    IxRec,
    VRec,
    timeRecord,
) = nn.simulate()

# %% [markdown]
### Analysis of simulation

# %% [markdown]
# Raster plot of neurons firing.

#%%
# s(0,:) are the spike times
# s(1,:) are the associated neuron indices

fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)

sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 2.3})

plt.scatter(s[0, :] / 1000, s[1, :], s=0.02, color="black")
plt.xlabel("time (s)")
plt.ylabel("Neuron index")
plt.ylim((0, N))
plt.yticks((0, N))
plt.xlim((0, T / 1000 / 10))
plt.xticks((0, T / 1000 / 10))
plt.show()

# %% [markdown]
# Balance of mean currents.

# %%
# Start the figure.
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
# ax.spines['left'].set_color('black')
# ax.spines['left'].set_linewidth(3)

sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 0.3})

plt.plot(
    timeRecord / 1000,
    np.mean(IeRec, axis=0) + np.mean(IxRec, axis=0),
    color="blue",
    label=r"e+x",
)
plt.plot(timeRecord / 1000, np.mean(IiRec, axis=0), color="red", label=r"i")
plt.plot(
    timeRecord / 1000,
    np.mean(IeRec, axis=0) + np.mean(IxRec, axis=0) + np.mean(IiRec, axis=0),
    color="black",
    label=r"e+i+x",
)

plt.xlabel("Time (s)")
plt.ylabel("Input")

plt.xlim((0, T / 1000))

leg = plt.legend(loc="upper left", fontsize=18, frameon="none", markerscale=1)
leg.get_frame().set_linewidth(0.0)

sns.despine()

plt.show()

# %% [markdown]
# Time course of mean synaptic weight.

# %%
# Start the figure.
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)

sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})

if eta_ee_hebb != 0:
    plt.plot(timeRecord / 1000, np.mean(JRec_ee, 0), color="darkgrey", label=r"$EE$")
if eta_ee_koh != 0:
    plt.plot(timeRecord / 1000, np.mean(JRec_ee, 0), color="darkgrey", label=r"$EE$")
if eta_ie_hebb != 0:
    plt.plot(timeRecord / 1000, np.mean(JRec_ie, 0), color="pink", label=r"$IE$")
if eta_ie_homeo != 0:
    plt.plot(timeRecord / 1000, np.mean(JRec_ie, 0), color="pink", label=r"$IE$")
if eta_ei != 0:
    plt.plot(timeRecord / 1000, np.mean(JRec_ei, 0), color="darkgreen", label=r"$EI$")
if eta_ii != 0:
    plt.plot(timeRecord / 1000, np.mean(JRec_ii, 0), color="darkviolet", label=r"$II$")

leg = plt.legend(loc="center left", fontsize=18, frameon="none", markerscale=1)
leg.get_frame().set_linewidth(0.0)

plt.xlabel("Time (s)")
plt.ylabel("Syn. weight")
plt.xticks((0, T / 1000))
plt.xlim((0, T / 1000))
sns.despine()

plt.show()

# %% [markdown]
# Distribution synaptic weight over connections.

# %%
# Start the figure.
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)

sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})


if eta_ee_hebb != 0:
    sns.histplot(
        JRec_ee[:, -1],
        color="darkgrey",
        stat="density",
        element="step",
        fill=False,
        bins=100,
    )
if eta_ee_koh != 0:
    sns.histplot(
        JRec_ee[:, -1],
        color="darkgrey",
        stat="density",
        element="step",
        fill=False,
        bins=100,
    )
if eta_ie_hebb != 0:
    sns.histplot(
        JRec_ie[:, -1],
        color="pink",
        stat="density",
        element="step",
        fill=False,
        bins=100,
    )
if eta_ie_homeo != 0:
    sns.histplot(
        JRec_ie[:, -1],
        color="pink",
        stat="density",
        element="step",
        fill=False,
        bins=100,
    )
if eta_ei != 0:
    sns.histplot(
        JRec_ei[:, -1],
        color="darkgreen",
        stat="density",
        element="step",
        fill=False,
        bins=100,
    )
if eta_ii != 0:
    sns.histplot(
        JRec_ii[:, -1],
        color="darkviolet",
        stat="density",
        element="step",
        fill=False,
        bins=100,
    )

plt.xlabel("Syn. weight")
plt.ylabel("Count")
plt.yticks(())

sns.despine()
plt.show()

# %% [markdown]
# Time course of firing rates.

#%%
# Compute histogram of rates (over time)
dtRate = 100  # ms
timeVector = np.arange(dtRate, T + dtRate, dtRate) / 1000
hist, bin_edges = np.histogram(s[0, s[1, :] < frac_exc * N], bins=len(timeVector))
eRateT = hist / (dtRate * frac_exc * N) * 1000

hist, bin_edges = np.histogram(s[0, s[1, :] >= frac_exc * N], bins=len(timeVector))
iRateT = hist / (dtRate * (1 - frac_exc) * N) * 1000

# Slide a window over the rates to smooth them.
window = 5
Num_points = int(len(eRateT) - window)
eRate_New = np.zeros((Num_points, 1))
iRate_New = np.zeros((Num_points, 1))
for i in range(Num_points):
    eRate_New[i, 0] = np.mean(eRateT[i : i + window])
    iRate_New[i, 0] = np.mean(iRateT[i : i + window])

# Start the figure.
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)

sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})

# Plot time-dependent rates
plt.plot(np.linspace(0, T / 1000, num=Num_points), eRate_New, color="blue", label=r"e")
plt.plot(np.linspace(0, T / 1000, num=Num_points), iRate_New, color="red", label=r"i")

plt.ylabel("rate (Hz)")
plt.xlabel("time (s)")
plt.xticks((0, T / 1000))
plt.xlim((0, T / 1000))

leg = plt.legend(loc="upper right", fontsize=18, frameon="none", markerscale=1)
leg.get_frame().set_linewidth(0.0)

sns.despine()
plt.show()

# %% [markdown]
# Define functions to compute empirical spike train covariances and correlations.

# %%

#  Compute spike count covariance matrix.
#  s is a 2x(ns) matrix where ns is the number of spikes
#  s(0,:) lists spike times
#  and s(1,:) lists corresponding neuron indices
#  Neuron indices are assumed to go from 1 to N

#  Spikes are counts starting at time T1 and ending at
#  time T2.

#  winsize is the window size over which spikes are counted,
#  so winsize is assumed to be much smaller than T2-T1

#  Covariances are only computed between neurons whose
#  indices are listed in the vector Inds. If Inds is not
#  passed in then all NxN covariances are computed.


def SpikeCountCov(s, N, T1, T2, winsize):

    Inds = np.arange(0, N)

    #   Count only spikes between T1, T2
    s1 = s[:, (s[0, :] <= T2) & (s[1, :] >= T1)]

    #   Count only for neurons between 0, N
    s1 = s[:, (s[1, :] < N) & (s[1, :] >= 0)]

    #   Edges for histogram
    edgest = np.arange(T1, T2, winsize)
    edgesi = np.arange(0, N + 1)

    #   Get 2D histogram of spike indices and times
    counts, xedges, yedges = np.histogram2d(s1[0, :], s1[1, :], bins=(edgest, edgesi))

    #   Compute and return covariance matrix
    return np.array(np.cov(counts.transpose()))


def cov2corr(cov):
    """convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.

    """
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    return corr


# %% [markdown]
# Apply functions above to compute covariances and correlations.

#%%

## All the code below computes spike count covariances and correlations

# Compute spike count covariances over windows of size
# winsize starting at time T1 and ending at time T2.
winsize = 250  # ms
T1 = T / 2  # ms
T2 = T  # ms
# Do computation
C = SpikeCountCov(s, N, T1, T2, winsize)


# Get mean spike count covariances over each sub-pop
II, JJ = np.meshgrid(np.arange(0, N), np.arange(0, N))
mCee = np.nanmean(C[(II < frac_exc * N) & (JJ < II)])
mCei = np.nanmean(C[(II < frac_exc * N) & (JJ >= frac_exc * N)])
mCii = np.nanmean(C[(II > frac_exc * N) & (JJ > II)])

# Mean-field spike count cov matrix
# Compare this to the theoretical prediction
mC = [[mCee, mCei], [mCei, mCii]]

# Compute spike count correlations
# This takes a while, so make it optional
ComputeSpikeCountCorrs = 1
if ComputeSpikeCountCorrs:

    #    Get correlation matrix from cov matrix
    start_time = time.time()
    R = cov2corr(C)
    elapsed_time = time.time() - start_time
    print(elapsed_time / 60, "minutes")

    mRee = np.nanmean(R[(II < frac_exc * N) & (JJ < II)])
    mRei = np.nanmean(R[(II < frac_exc * N) & (JJ >= frac_exc * N)])
    mRii = np.nanmean(R[(II > frac_exc * N) & (JJ > II)])

    # Mean-field spike count correlation matrix
    mR = [[mRee, mRei], [mRei, mRii]]
    print("mR =", mR)

# %% [markdown]
# Plot distributions of EE, EI, II correlations.

# %%
# Start the figure.
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)

sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})

sns.histplot(
    R[(II < frac_exc * N) & (JJ < II)],
    bins=100,
    kde=False,
    stat="density",
    element="step",
    fill=False,
)
sns.histplot(
    R[(II < frac_exc * N) & (JJ >= frac_exc * N)],
    bins=100,
    kde=False,
    stat="density",
    element="step",
    fill=False,
)
sns.histplot(
    R[(II > frac_exc * N) & (JJ > II)],
    bins=100,
    kde=False,
    stat="density",
    element="step",
    fill=False,
)

plt.xlabel("spike count corr")
plt.ylabel("Count")

sns.despine()
plt.show()


# %%
