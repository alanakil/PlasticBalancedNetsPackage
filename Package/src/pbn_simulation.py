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
import random
import matplotlib.pyplot as plt
import math
import seaborn as sns
import time
import logging
import time
import sys
from datetime import datetime as dtm
import datetime as dt
import os
from pathlib import Path

#%%
from pbn.helpers import plasticNeuralNetwork

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

log_format = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
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
    filemode = "w",
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
    level = level,
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
P=np.array([[p_ee, p_ei], [p_ie, p_ii]])
# Ffwd connection probs
Px=np.array([[p_ex],[p_ix]])

# Timescale of correlation in ms
taujitter=5
# Mean connection strengths between each cell type pair
jee = 25
jei = -150
jie = 112.5
jii = -250
jex = 180
jix = 135
Jm=np.array([[jee, jei],[jie, jii]])/np.sqrt(N)
Jxm=np.array([[jex],[jix]])/np.sqrt(N)

# Total_time (in ms) for sim
T=5000

# Total_time discretization
dt=.1

# FFwd spike train rate (in kHz)
rx=10/1000

# Extra stimulus: Istim is a Total_time-dependent stimulus
# it is delivered to all neurons with weights given by JIstim.
# Specifically, the stimulus to neuron j at Total_time index i is:
# Istim(i)*JIstim(j)
jestim=0
jistim=0

# Synaptic timescales in ms
taux=10
taue=8
taui=4

# Generate FFwd spike trains 
# Correlation of ffwd spike trains.
c=0.1

# Neuron parameters
Cm=1
gL=1/15
EL=-72
Vth=-50
Vre=-75
DeltaT=1
VT=-55

# Plasticity parameters
tauSTDP=200 # ms

#EE hebb
Jmax_ee = 30/np.sqrt(N)
eta_ee_hebb= 0/10**3 # Learning rate 

#EE kohonen
beta = 2/np.sqrt(N)
eta_ee_koh= 0/10**2 # Learning rate 

#IE hebb
Jmax_ie_hebb = 125/np.sqrt(N)
eta_ie_hebb= 0/10**3 # Learning rate 

#IE homeo
Jnorm_ie = 200/np.sqrt(N)
eta_ie_homeo = 0/10**3 /Jnorm_ie # Learning rate 
rho_ie=0.020 # Target rate 20Hz
alpha_ie=2*rho_ie*tauSTDP

#EI homeo
Jnorm_ei = -200/np.sqrt(N)
eta_ei=0/10**3 /Jnorm_ei # Learning rate 
rho_ei=0.010 # Target rate 10Hz
alpha_ei=2*rho_ei*tauSTDP

#II
Jnorm_ii = -300/np.sqrt(N)
eta_ii = 0/10**3 /Jnorm_ii # Learning rate 
rho_ii = 0.020 # Target rate 20Hz
alpha_ii = 2*rho_ii*tauSTDP

# Indices of neurons to record currents, voltages
numrecord = int(100)  # Number to record from each population
Irecord = np.array([[random.sample(list(np.arange(0,frac_exc*N)), numrecord), random.sample(list(np.arange(frac_exc*N,N)), numrecord) ]])
Ierecord = np.sort(Irecord[0,0]).astype(int)
Iirecord = np.sort(Irecord[0,1]).astype(int)
Ixrecord = np.sort(random.sample(list(np.arange(0,frac_ext*N)), numrecord)).astype(int)
Vrecord = np.sort( [[random.sample(list(np.arange(0,frac_exc*N)), int(round(numrecord/2)) ), random.sample( list(np.arange(frac_exc*N,N)), int(round(numrecord/2)) ) ]])[0].reshape(1, numrecord).astype(int)[0]
del Irecord

# Number of time bins to average over when recording
nBinsRecord=10
dtRecord=nBinsRecord*dt

#%%
# Define the model.
nn = plasticNeuralNetwork(N, frac_exc, frac_ext, P, Px, taujitter, Jm, Jxm, T, dt, rx, jestim, 
                jistim, taue, taui, taux, c, Cm, gL, EL, Vth, Vre, DeltaT, VT, tauSTDP, 
                Jmax_ee, eta_ee_hebb, eta_ee_koh, beta, eta_ie_hebb, Jmax_ie_hebb, 
                eta_ie_homeo, alpha_ie, alpha_ei, eta_ei, alpha_ii, eta_ii, nBinsRecord,
                dtRecord, Ierecord, Iirecord, Ixrecord, Vrecord, numrecord)

#%%
# Initialize the connectivity
nn.connectivity()

#%%
# Generate Poisson ffwd spike trains
nn.ffwd_spikes()

#%%
# Simulate plastic network
s, JRec_ee, JRec_ie, JRec_ei, JRec_ii, IeRec, IiRec, IxRec, VRec = nn.simulate()

# %% [markdown]
# Analysis of simulation
