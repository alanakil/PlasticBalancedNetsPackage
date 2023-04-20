"""
This code is a sample simulation of a plastic balanced network as defined in Akil et al., 2021. 
The purpose of this code is to demonstrate how the `plastic_balanced_network` package can be used.
In particular, we here show how to simulate a PBN as N gets large and compare that to the theoretical predictions derived in Akil et al., 2021.

Output data is saved in the `data/processed` folder. 
Logs are saved in `logs` folder.

Before running, ensure that the package is installed in your virtual environment. See README.md for details.

"""
__author__ = "Alan Akil (alan.akil@yahoo.com)"
__date__ = "APRIL 2023"

#%%
# Load python packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime as dtm
import os
from pathlib import Path

from plastic_balanced_network.helpers import (
    plasticNeuralNetwork,
    compute_firing_rate,
    spike_count_cov,
    cov2corr,
    average_cov_corr_over_subpops,
)

#%%
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
DATA_FILE_PATH = f"{DATA_DIR}/pbn_data_{datadatetime}.csv"

#%%
# Set up logging.
log_format = (
    "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
)
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


#%%
# Define all input variables for the network simulation.
logging.info("Define input variables for plastic balanced network simulation.")

# Total number of neurons.
N_vector = np.array([500, 1000, 2000, 5000])
# Total time (in ms) for simulation.
T = int(10000)
# FFwd spike train rate (in kHz).
rx = 10 / 1000
# Correlation of ffwd spike trains.
cx = 0
# Fraction of E and X neurons.
frac_exc = 0.8
frac_ext = 0.2

# EE hebb
eta_ee_hebb = 0 / 10**3  # Learning rate, if zero then no plasticity.
jmax_ee = 30
# EE kohonen
eta_ee_koh = 0 / 10**2  # Learning rate, if zero then no plasticity.
beta = 2
# IE hebb
eta_ie_hebb = 0 / 10**3  # Learning rate, if zero then no plasticity.
jmax_ie_hebb = 125
# IE homeostatic
eta_ie_homeo = 0 / 10**3  # Learning rate, if zero then no plasticity.
rho_ie = 0.020  # Target rate 20Hz
# EI homeostatic
eta_ei = 0 / 10**3  # Learning rate, if zero then no plasticity.
rho_ei = 0.010  # Target rate 10Hz
# II homeostatic
eta_ii = 0 / 10**3  # Learning rate, if zero then no plasticity.
rho_ii = 0.020  # Target rate 20Hz

# Time interval where covs and corrs are computed.
T1 = T / 2  # ms
T2 = T  # ms

# Set the random seed.
np.random.seed(31415)

#%%
# Loop over N. Compute relevant variables and only save those.

results_df = pd.DataFrame(
    np.nan,
    index=range(len(N_vector)),
    columns=[
        "N",
        "frac_exc",
        "frac_ext",
        "T",
        "cx",
        "rx",
        "eta_ee_hebb",
        "jmax_ee",
        "eta_ee_koh",
        "beta",
        "eta_ie_homeo",
        "rho_ie",
        "eta_ie_hebb",
        "jmax_ie_hebb",
        "eta_ei",
        "rho_ei",
        "eta_ii",
        "rho_ii",
        "eRate",
        "iRate",
        "mCee",
        "mCei",
        "mCii",
        "mRee",
        "mRei",
        "mRii",
        "mJee",
        "mJie",
        "mJei",
        "mJii",
    ],
)

results_df["N"] = N_vector
results_df["frac_exc"] = frac_exc
results_df["frac_ext"] = frac_ext
results_df["T"] = T
results_df["cx"] = cx
results_df["rx"] = rx
results_df["eta_ee_hebb"] = eta_ee_hebb
results_df["jmax_ee"] = jmax_ee
results_df["eta_ee_koh"] = eta_ee_koh
results_df["eta_ie_hebb"] = eta_ie_hebb
results_df["jmax_ie_hebb"] = jmax_ie_hebb
results_df["eta_ie_homeo"] = eta_ie_homeo
results_df["eta_ei"] = eta_ei
results_df["eta_ii"] = eta_ii
results_df["rho_ie"] = rho_ie
results_df["rho_ei"] = rho_ei
results_df["rho_ii"] = rho_ii

results_df = results_df.set_index("N")

for N in N_vector:

    pnn = plasticNeuralNetwork(N, T)
    pnn.connectivity()
    pnn.ffwd_spikes(T, cx, rx)

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
        jmax_ee=jmax_ee,
        eta_ee_koh=eta_ee_koh,
        eta_ie_homeo=eta_ie_homeo,
        jmax_ie_hebb=jmax_ie_hebb,
        eta_ie_hebb=eta_ie_hebb,
        eta_ei=eta_ei,
        eta_ii=eta_ii,
        rho_ie=rho_ie,
        rho_ei=rho_ei,
        rho_ii=rho_ii,
    )

    # Compute relevant variables and save.
    eRateT, iRateT, timeVector = compute_firing_rate(s, T, N)
    # Average rates over the second half of the simulation (when at steady state).
    results_df.loc[N, "eRate"] = np.mean(eRateT[len(eRateT) // 2 :])
    results_df.loc[N, "iRate"] = np.mean(iRateT[len(iRateT) // 2 :])

    # Covs and Corrs
    C = spike_count_cov(s, N, T1, T2)
    R = cov2corr(C)
    mC = average_cov_corr_over_subpops(C, N, frac_exc)
    mR = average_cov_corr_over_subpops(R, N, frac_exc)

    results_df.loc[N, "mCee"] = mC[0][0]
    results_df.loc[N, "mCei"] = mC[0][1]
    results_df.loc[N, "mCii"] = mC[1][1]
    results_df.loc[N, "mRee"] = mR[0][0]
    results_df.loc[N, "mRei"] = mR[0][1]
    results_df.loc[N, "mRii"] = mR[1][1]

    # Avg weights over connections and time.
    JRec_ee = np.mean(JRec_ee, 0)
    JRec_ie = np.mean(JRec_ie, 0)
    JRec_ei = np.mean(JRec_ei, 0)
    JRec_ii = np.mean(JRec_ii, 0)
    results_df.loc[N, "mJee"] = np.mean(JRec_ee[len(JRec_ee) // 2 :])
    results_df.loc[N, "mJie"] = np.mean(JRec_ie[len(JRec_ie) // 2 :])
    results_df.loc[N, "mJei"] = np.mean(JRec_ei[len(JRec_ei) // 2 :])
    results_df.loc[N, "mJii"] = np.mean(JRec_ii[len(JRec_ii) // 2 :])


#%% [markdown]
# Save and load relevant data variables for analysis and plotting.

# %%
results_df.to_csv(DATA_FILE_PATH)

#%% [markdown]
# Load relevant data variables for analysis and plotting.
# We'll focus on plotting relevant quantities like rates, covariances, correlations and synaptic weights over N.
# They should converge to a fixed point if appropriate conditions are chosen (see Akil et al. 2021).

#%%
# Load data.
results_df = pd.read_csv("../data/processed/pbn_data_2023APR20-1604.csv")

#%%
# Let's start with Rates.

fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})

plt.plot(results_df["N"], results_df["eRate"], label="e")
plt.plot(results_df["N"], results_df["iRate"], label="i")

plt.xlabel("N")
plt.ylabel("Rates (Hz)")
plt.xscale("log")
leg = plt.legend(loc="upper right", fontsize=18, frameon="none", markerscale=1)
leg.get_frame().set_linewidth(0.0)

sns.despine()
plt.show()

# %%
# Covariances.

fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})

plt.plot(results_df["N"], results_df["mCee"], label="ee")
plt.plot(results_df["N"], results_df["mCei"], label="ei")
plt.plot(results_df["N"], results_df["mCii"], label="ii")

plt.xlabel("N")
plt.ylabel("Mean Covariances")
plt.xscale("log")
leg = plt.legend(loc="upper right", fontsize=18, frameon="none", markerscale=1)
leg.get_frame().set_linewidth(0.0)

sns.despine()
plt.show()


# %%
# Correlations.

fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})

plt.plot(results_df["N"], results_df["mRee"], label="ee")
plt.plot(results_df["N"], results_df["mRei"], label="ei")
plt.plot(results_df["N"], results_df["mRii"], label="ii")

plt.xlabel("N")
plt.ylabel("Mean Correlations")
plt.xscale("log")
leg = plt.legend(loc="upper right", fontsize=18, frameon="none", markerscale=1)
leg.get_frame().set_linewidth(0.0)

sns.despine()
plt.show()


# %%
# Syanptic weights.
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
sns.set()
sns.set_style("whitegrid")
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk", font_scale=1.9, rc={"lines.linewidth": 3.3})

if eta_ee_hebb != 0:
    plt.plot(results_df["N"], results_df["mJee"], label="ee")
if eta_ee_koh != 0:
    plt.plot(results_df["N"], results_df["mJee"], label="ee")
if eta_ie_hebb != 0:
    plt.plot(results_df["N"], results_df["mJie"], label="ie")
if eta_ie_homeo != 0:
    plt.plot(results_df["N"], results_df["mJie"], label="ie")
if eta_ei != 0:
    plt.plot(results_df["N"], results_df["mJei"], label="ei")
if eta_ii != 0:
    plt.plot(results_df["N"], results_df["mJii"], label="ii")

plt.xlabel("Syn. weight")
plt.ylabel("Count")
leg = plt.legend(loc="upper right", fontsize=18, frameon="none", markerscale=1)
leg.get_frame().set_linewidth(0.0)

sns.despine()
plt.show()

# %%
