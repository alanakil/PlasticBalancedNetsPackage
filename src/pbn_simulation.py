"""
This code is a sample simulation of a plastic balanced network as defined in Akil et al, 2021.
The purpose of this code is to demonstrate how the `plastic_balanced_network` package can be used.

We allow for a great deal of flexibility constructing the simulation/s.
One may easily change the following parameters:
(1) Total number of neurons.
(2) Fraction of E-I neurons.
(3) Probability of connection.
(4) Synaptic strengths.
(5) Total time of simulation.
(6) Input rate and correlations.
(7) Extra injected current.
(8) EIF neuron parameters.
(9) Plasticity parameters on any connection type.

Output data is saved in the `data/processed` folder.
Logs are saved in `logs` folder.
Please refer to the README.md for detailed instructions on how to run this code.
"""
__author__ = "Alan Akil (alan.akil@yahoo.com)"
__date__ = "APRIL 2023"

# %%
# Load python packages.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from datetime import datetime as dtm
import os
from pathlib import Path

from plastic_balanced_network.network import (
    PlasticNeuralNetwork,
    compute_firing_rate,
    spike_count_cov,
    cov2corr,
    average_cov_corr_over_subpops,
)

# %%
# Construct a str containing the datetime when the simulation is run.
todaysdate = dtm.today()
datetime_format = "%Y%b%d-%H%M"
datadatetime = todaysdate.strftime(datetime_format).upper()

# Define paths: project root, data directory, and logs directory.
PROJECTROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECTROOT, "data", "processed")
LOG_DIR = os.path.join(PROJECTROOT, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
DATA_FILE_PATH = f"{DATA_DIR}/pbn_data_{datadatetime}.npz"

# %%
# Set up logging.
log_format = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
# Use loglevel to filter out undesired logs.
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

logging.info(f"Project root: {PROJECTROOT}.")
logging.info(f"Data directory: {DATA_DIR}.")
logging.info(f"Logs directory: {LOG_DIR}.")


# %%
# Define all input variables for the network simulation.
logging.info("Define input variables for plastic balanced network simulation.")

# Total number of neurons.
N = int(5000)
# Total time (in ms) for simulation.
T = 5000

# FFwd spike train rate (in kHz).
rx = 10 / 1000
# Correlation of ffwd spike trains.
cx = 0.1

# EE hebb
eta_ee_hebb = 0 / 10**3  # Learning rate, if zero then no plasticity.
# EE kohonen
eta_ee_koh = 0 / 10**2  # Learning rate, if zero then no plasticity.
beta = 2
# IE hebb
eta_ie_hebb = 0 / 10**3  # Learning rate, if zero then no plasticity.
# IE homeostatic
eta_ie_homeo = 0 / 10**3  # Learning rate, if zero then no plasticity.
rho_ie = 0.020  # Target rate 20Hz
# EI homeostatic
eta_ei = 0.015 / 10**3  # Learning rate, if zero then no plasticity.
rho_ei = 0.010  # Target rate 10Hz
# II homeostatic
eta_ii = 0.015 / 10**3  # Learning rate, if zero then no plasticity.
rho_ii = 0.020  # Target rate 20Hz

# Set the random seed.
np.random.seed(31415)

# %%
# Define the model.
pnn = PlasticNeuralNetwork(
    N,
    T,
)

# %%
# Initialize the connectivity.
pnn.connectivity()

# %%
# Generate Poisson ffwd spike trains.
pnn.ffwd_spikes(T, cx, rx)

# %%
# Simulate plastic network.
# Note that spike trains are recorded in s as follows:
# s(0,:) are the spike times
# s(1,:) are the associated neuron indices
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
) = pnn.simulate(
    eta_ee_hebb=eta_ee_hebb,
    eta_ee_koh=eta_ee_koh,
    eta_ie_homeo=eta_ie_homeo,
    eta_ie_hebb=eta_ie_hebb,
    eta_ei=eta_ei,
    eta_ii=eta_ii,
)

# %%
# If one uses default parameters from the pnn class, we can easily access them like this:
frac_exc = pnn.frac_exc
frac_ext = pnn.frac_ext
jee = pnn.jee
jie = pnn.jie
jei = pnn.jei
jii = pnn.jii
jex = pnn.jex
jix = pnn.jix
p_ee = pnn.p_ee
p_ie = pnn.p_ie
p_ei = pnn.p_ei
p_ii = pnn.p_ii
p_ex = pnn.p_ex
p_ix = pnn.p_ix
tauSTDP = pnn.tauSTDP
beta = pnn.beta
jmax_ee = pnn.jmax_ee
jmax_ie_hebb = pnn.jmax_ie_hebb
dt = pnn.dt

# %% [markdown]
## Save and load relevant data variables for analysis and plotting.

# %%
# Save data containing relevant variables.
np.savez(
    DATA_FILE_PATH,  # File name
    s=s,
    sx=sx,
    JRec_ee=JRec_ee,
    JRec_ie=JRec_ie,
    JRec_ei=JRec_ei,
    JRec_ii=JRec_ii,
    IeRec=IeRec,
    IiRec=IiRec,
    IxRec=IxRec,
    VRec=VRec,
    timeRecord=timeRecord,
    N=N,
    frac_exc=frac_exc,
    frac_ext=frac_ext,
    p_ee=p_ee,
    p_ie=p_ie,
    p_ei=p_ei,
    p_ii=p_ii,
    p_ex=p_ex,
    p_ix=p_ix,
    jee=jee,
    jie=jie,
    jei=jei,
    jii=jii,
    jex=jex,
    jix=jix,
    T=T,
    dt=dt,
    cx=cx,
    rx=rx,
    tauSTDP=tauSTDP,
    jmax_ee=jmax_ee,
    eta_ee_hebb=eta_ee_hebb,
    beta=beta,
    eta_ee_koh=eta_ee_koh,
    jmax_ie_hebb=jmax_ie_hebb,
    eta_ie_hebb=eta_ie_hebb,
    eta_ie_homeo=eta_ie_homeo,
    rho_ie=rho_ie,
    eta_ei=eta_ei,
    rho_ei=rho_ei,
    eta_ii=eta_ii,
    rho_ii=rho_ii,
)

# %%
# Load data from previous runs.
data = np.load(DATA_FILE_PATH)
# loop through the variables and set them as local variables with the same name as the key
for key, _value in data.items():
    exec(f"{key} = value")

# %% [markdown]
### Analysis of simulation

# %% [markdown]
# Raster plot of neurons firing.

# %%
# Raster plot.
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
plt.xlim((0, T / 1000))
plt.xticks((0, T / 1000))
plt.show()

# %% [markdown]
# Balance of mean currents.

# %%
# Plot input currents.
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
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
# Time course of mean synaptic weight.
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
# Distribution synaptic weight over connections.
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

sns.despine()
plt.show()

# %% [markdown]
# Time course of firing rates.

# %%
# Compute smoothed histogram of rates (over time)
eRateT, iRateT, timeVector = compute_firing_rate(s, T, N, frac_exc=0.8, dtRate=10, window_size=10)

# Start the figure.
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})

# Plot time-dependent rates
plt.plot(timeVector, eRateT, color="blue", label=r"e")
plt.plot(timeVector, iRateT, color="red", label=r"i")

plt.ylabel("rate (Hz)")
plt.xlabel("time (s)")
plt.xticks((0, T / 1000))
plt.xlim((0, T / 1000))

leg = plt.legend(loc="upper right", fontsize=18, frameon="none", markerscale=1)
leg.get_frame().set_linewidth(0.0)

sns.despine()
plt.show()

# %% [markdown]
# Compute empirical spike train covariances and correlations.

# %%
# Compute spike count covariances over windows of size
# winsize starting at time T1 and ending at time T2.
winsize = 250  # ms
T1 = T / 2  # ms
T2 = T  # ms
# Do computation
C = spike_count_cov(s, N, T1, T2, winsize)

mC = average_cov_corr_over_subpops(C, N, frac_exc)

# Compute spike count correlations
# Get correlation matrix from cov matrix
start_time = time.time()
R = cov2corr(C)
elapsed_time = time.time() - start_time
print(elapsed_time / 60, "minutes")

mR = average_cov_corr_over_subpops(R, N, frac_exc)
print("mR =", mR)

# %% [markdown]
# Plot distributions of EE, EI, IE, II correlations.

# %%
# Plot distributions of EE, EI, IE, II correlations.
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})

II, JJ = np.meshgrid(np.arange(0, N), np.arange(0, N))

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
