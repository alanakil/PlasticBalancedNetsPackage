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
import random2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from datetime import datetime as dtm
import datetime as dt
import os
from pathlib import Path

from plastic_balanced_network.helpers import plasticNeuralNetwork, SpikeCountCov, cov2corr, average_cov_corr_over_subpops

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
N_vector = np.array([100, 500, 1000, 2000, 5000, 10000])
# Fraction of excitatory neurons.
frac_exc = 0.8
# Extra fraction of neurons (external).
frac_ext = 0.2

# Define individual connection probabilities.
p_ee = 0.1
p_ei = 0.1
p_ie = 0.1
p_ii = 0.1
p_ex = 0.1
p_ix = 0.1
# Recurrent net connection probabilities.
P = np.array([[p_ee, p_ei], [p_ie, p_ii]])
# Ffwd connection probs.
Px = np.array([[p_ex], [p_ix]])

# Mean connection strengths between each cell type pair.
jee = 25
jei = -150
jie = 112.5
jii = -250
jex = 180
jix = 135

# Total time (in ms) for simulation.
T = 100000

# Step size for discretization.
dt = 0.1

# FFwd spike train rate (in kHz).
rx = 10 / 1000
# Correlation of ffwd spike trains.
cx = 0.1
# Timescale of correlation in ms. Jitter spike trains in external layer by taujitter.
taujitter = 5

# Extra stimulus: Istim is a Total_time-dependent stimulus
# it is delivered to all neurons with weights given by Jstim.
# Specifically, the stimulus to neuron j at Total_time index i is:
# Istim(i)*Jstim(j)
jestim = 0
jistim = 0

# Synaptic timescales in ms.
taux = 10
taue = 8
taui = 4

# Neuron parameters.
Cm = 1
gL = 1 / 15
EL = -72
Vth = -50
Vre = -75
DeltaT = 1
VT = -55

# Plasticity parameters.
tauSTDP = 200  # ms

eta_ee_hebb = 0 / 10**3  # Learning rate, if zero then no plasticity.
eta_ee_koh = 0 / 10**2  # Learning rate, if zero then no plasticity.
eta_ie_hebb = 0 / 10**3  # Learning rate, if zero then no plasticity.
rho_ie = 0.020  # Target rate 20Hz
alpha_ie = 2 * rho_ie * tauSTDP
rho_ei = 0.010  # Target rate 10Hz
alpha_ei = 2 * rho_ei * tauSTDP
rho_ii = 0.020  # Target rate 20Hz
alpha_ii = 2 * rho_ii * tauSTDP

# Number of neurons to record from each population.
numrecord = int(10)  
# Number of time bins to average over when recording currents and voltages.
nBinsRecord = 10
# Number of synapses to be sampled per cell type pair that is plastic.
nJrecord0 = 10

winsize = 250  # ms
T1 = T / 2  # ms
T2 = T  # ms
# Set the random seed.
np.random.seed(31415)

#%% 
# Loop over N. Compute relevant variables and only save those.
# We will want to save the value of N, frac_exc, eRate, iRate, mC, mR, mean weights (if plastic).

results_df = pd.DataFrame(
    np.nan,
    index=range(len(N_vector)),
    columns=["N", "frac_exc", "frac_ext", "T", "dt", "cx", "rx", 
             "eta_ee_hebb", "Jmax_ee", "eta_ee_koh", "beta", "eta_ie_homeo",
             "alpha_ie", "eta_ie_hebb", "Jmax_ie_hebb", "eta_ei", "alpha_ei",
             "eta_ii", "alpha_ii"
             "eRate", "iRate", "mCee", "mCei", "mCii", "mRee", "mRei", "mRii",
             "mJee", "mJie", "mJei", "mJii"]
    )

results_df["N"] = N_vector
results_df["frac_exc"] = frac_exc
results_df["frac_ext"] = frac_ext
results_df["T"] = T
results_df["dt"] = dt
results_df["cx"] = cx
results_df["rx"] = rx
results_df["eta_ee_hebb"] = eta_ee_hebb
results_df["eta_ee_koh"] = eta_ee_koh
results_df["eta_ie_hebb"] = eta_ie_hebb
results_df["alpha_ie"] = alpha_ie
results_df["alpha_ei"] = alpha_ei
results_df["alpha_ii"] = alpha_ii

results_df = results_df.set_index("N")

for N in N_vector:
    
    Jm = np.array([[jee, jei], [jie, jii]]) / np.sqrt(N)
    Jxm = np.array([[jex], [jix]]) / np.sqrt(N)

    # EE hebb
    Jmax_ee = 30 / np.sqrt(N)
    results_df.loc[N, "Jmax_ee"] = Jmax_ee
    # EE kohonen
    beta = 2 / np.sqrt(N)
    results_df.loc[N, "beta"] = beta
    # IE hebb
    Jmax_ie_hebb = 125 / np.sqrt(N)
    results_df.loc[N, "Jmax_ie_hebb"] = Jmax_ie_hebb
    # IE homeostatic
    Jnorm_ie = 200 / np.sqrt(N)
    eta_ie_homeo = 0 / 10**3 / Jnorm_ie  # Learning rate, if zero then no plasticity.
    results_df.loc[N, "eta_ie_homeo"] = eta_ie_homeo
    # EI homeostatic
    Jnorm_ei = -200 / np.sqrt(N)
    eta_ei = 0 / 10**3 / Jnorm_ei  # Learning rate, if zero then no plasticity.
    results_df.loc[N, "eta_ei"] = eta_ei
    # II homeostatic
    Jnorm_ii = -300 / np.sqrt(N)
    eta_ii = 0 / 10**3 / Jnorm_ii  # Learning rate, if zero then no plasticity.
    results_df.loc[N, "eta_ii"] = eta_ii

    pnn = plasticNeuralNetwork(
        N,
        frac_exc,
        frac_ext,
        T,
        dt,
        jestim,
        jistim,
        nBinsRecord,
    )
    
    pnn.connectivity(Jm, Jxm, P, Px, nJrecord0)

    pnn.ffwd_spikes(cx, rx, taujitter, T)

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
        Cm,
        gL,
        VT,
        Vre,
        Vth,
        EL,
        DeltaT,
        taue,
        taui,
        taux,
        tauSTDP,
        numrecord,
        eta_ee_hebb,
        Jmax_ee,
        eta_ee_koh,
        beta,
        eta_ie_homeo,
        alpha_ie,
        eta_ie_hebb,
        Jmax_ie_hebb,
        eta_ei,
        alpha_ei,
        eta_ii,
        alpha_ii,
        dt,
        nBinsRecord,
    )

    # Compute relevant variables and save.
    # Compute histogram of rates (over time)
    dtRate = 100  # ms
    timeVector = np.arange(dtRate, T + dtRate, dtRate) / 1000
    hist, bin_edges = np.histogram(s[0, s[1, :] < frac_exc * N], bins=len(timeVector))
    eRateT = hist / (dtRate * frac_exc * N) * 1000
    hist, bin_edges = np.histogram(s[0, s[1, :] >= frac_exc * N], bins=len(timeVector))
    iRateT = hist / (dtRate * (1 - frac_exc) * N) * 1000

    eRate = np.mean(eRateT[len(eRateT)//2:])
    iRate = np.mean(iRateT[len(iRateT)//2:])

    results_df.loc[N, "eRate"] = eRate
    results_df.loc[N, "iRate"] = iRate

    # Covs and Corrs
    C = SpikeCountCov(s, N, T1, T2, winsize)
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
    results_df.loc[N, "mJee"] = np.mean(JRec_ee[len(JRec_ee)//2:])
    results_df.loc[N, "mJie"] = np.mean(JRec_ie[len(JRec_ie)//2:])
    results_df.loc[N, "mJei"] = np.mean(JRec_ei[len(JRec_ei)//2:])
    results_df.loc[N, "mJii"] = np.mean(JRec_ii[len(JRec_ii)//2:])


#%% [markdown]
## Save and load relevant data variables for analysis and plotting.

# %%
results_df.to_csv(DATA_FILE_PATH)

#%%
