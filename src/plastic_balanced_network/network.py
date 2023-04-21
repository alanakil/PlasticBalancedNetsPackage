"""
Plastic balanced network class. Includes functions to initialize, build connectivity, simulate, etc.
"""
__author__ = "Alan Akil (alan.akil@yahoo.com)"
__date__ = "APRIL 2023"

# %%
import numpy as np
import random2
import math
import time
import logging
import sys


# %%
class PlasticNeuralNetwork:
    """PlasticNeuralNetwork is a class that builds a neural network with
    correlated or uncorrelated firing as well as with plastic or static
    synaptic weights on any connection.
    It contains functions to define the connectivity, simulate feedforward external
    spike trains, and to simulate the recurrent network's firing.

    Arguments
    :param N: Total number of neurons in recurrent neural network.
    :type N: int
    :param T: Total time of simulation in milliseconds.
    :type T: int
    :param frac_exc: Fraction of excitatory neurons. Defaults to 0.8.
    :type frac_exc: float or int
    :param frac_ext: Fraction of excitatory neurons in external layer. Defaults to 0.2.
    :type frac_ext: float or int
    :param dt: Time bin size in milliseconds for time discretization. Defaults to 0.1 ms.
    :type dt: float or int
    :param jestim: Added constant current to excitatory neurons. Defaults to 0.
    :type jestim: float or int
    :param jistim: Added constant current to inhibitory neurons. Defaults to 0.
    :type jistim: float or int
    :param nBinsRecord: Number of bins to record average and record over. Defaults to 10.
    :type nBinsRecord: int

    Returns as part of self.
    :return: Number of excitatory neurons (Ne); Number of inhibitory neurons (Ni); Number of external neurons (Nx);
    Total number of discretized time points (Nt); Time vector of added constant stimulatiom (Istim);
    Weight coupling for Istim (Jstim); Maximum number of spikes to terminate pathologic behavior (maxns);
    Discretized recorded time domain (timeRecord); Number of points in discretized recorded time domain (Ntrec).
    :rtype: tuple(int, int, int, int, np.ndarray, np.ndarray, float or int, np.ndarray, int)
    """

    def __init__(
        self,
        N,
        T,
        frac_exc=0.8,
        frac_ext=0.2,
        dt=0.1,
        jestim=0,
        jistim=0,
        nBinsRecord=10,
    ):
        # None error.
        if N is None:
            err = ValueError(
                """ERROR: N cannot be None. Pick an integer number of total neurons.
                A number in the order of 10^3 is a good place to start.
                """
            )
            logging.exception(err)
            raise err
        if T is None:
            err = ValueError(
                """ERROR: T cannot be None. Pick an integer number of total simulation time.
                A number in the order of 10^3 is a good place to start for a quick simulation.
                """
            )
            logging.exception(err)
            raise err
        # Type tests.
        if not isinstance(N, (int, np.integer)):
            err = TypeError("ERROR: N is not int")
            logging.exception(err)
            raise err
        if not isinstance(frac_exc, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: frac_exc is not one of these - int, float")
            logging.exception(err)
            raise err
        if not isinstance(frac_ext, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: frac_ext is not one of these - int, float")
            logging.exception(err)
            raise err
        if not isinstance(T, (int, np.integer)):
            err = TypeError("ERROR: T is not int")
            logging.exception(err)
            raise err
        if not isinstance(dt, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: dt is not one of these - int, float")
            logging.exception(err)
            raise err
        if not isinstance(jestim, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jestim is not one of these - int, float")
            logging.exception(err)
            raise err
        if not isinstance(jistim, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jistim is not one of these - int, float")
            logging.exception(err)
            raise err
        if not isinstance(nBinsRecord, (int, np.integer)):
            err = TypeError("ERROR: nBinsRecord is not int")
            logging.exception(err)
            raise err
        # Value tests.
        if N < 1:
            err = ValueError("ERROR: N has to be greater than or equal to 1.")
            logging.exception(err)
            raise err
        if (frac_exc > 1) | (frac_exc < 0):
            err = ValueError("ERROR: frac_exc hast be between 0 and 1.")
            logging.exception(err)
            raise err
        if (frac_ext > 1) | (frac_ext < 0):
            err = ValueError("ERROR: frac_ext hast be between 0 and 1.")
            logging.exception(err)
            raise err
        if T < 1:
            err = ValueError("ERROR: T has to be at least 1 millisecond.")
            logging.exception(err)
            raise err
        if dt <= 0:
            err = ValueError("ERROR: dt has to be positive.")
            logging.exception(err)
            raise err
        if nBinsRecord <= 0:
            err = ValueError("ERROR: nBinsRecord has to be positive.")
            logging.exception(err)
            raise err

        # Start the simulation.
        start_time = time.time()

        self.N = N
        self.frac_exc = frac_exc
        self.frac_ext = frac_ext
        self.Ne = int(round(frac_exc * N))
        self.Ni = int(round((1 - frac_exc) * N))
        self.Nx = int(round(frac_ext * N))
        self.Nt = round(T / dt)
        self.total_time = np.arange(dt, T + dt, dt)  # Time discretized domain.
        self.Istim = np.zeros(len(self.total_time))
        self.Istim[self.total_time > T / 2] = 0
        self.Jstim = np.sqrt(N) * np.hstack((jestim * np.ones((1, self.Ne)), jistim * np.ones((1, self.Ni))))
        self.maxns = round(0.05 * N * T)  # Max num of spikes (50 Hz).
        dtRecord = nBinsRecord * dt
        self.timeRecord = np.arange(dtRecord, T + dtRecord, dtRecord)
        self.Ntrec = len(self.timeRecord)
        self.dt = dt
        self.nBinsRecord = nBinsRecord

        logging.info(f"Simulating a network of {self.N} neurons.")
        logging.info(f"{self.Ne} neurons are excitatory.")
        logging.info(f"{self.Ni} neurons are inhibitory.")
        logging.info(f"The external layer, X, provides input from {self.Nx} excitatory neurons.")
        logging.info(f"The network will be simulated for {T/1000} seconds.")

        elapsed_time = time.time() - start_time
        logging.info(f"Time for initializing the class: {round(elapsed_time/60,2)} minutes.")

    def connectivity(
        self,
        jee=25,
        jie=112.5,
        jei=-150,
        jii=-250,
        jex=180,
        jix=135,
        p_ee=0.1,
        p_ie=0.1,
        p_ei=0.1,
        p_ii=0.1,
        p_ex=0.1,
        p_ix=0.1,
        nJrecord0=100,
    ):
        """
        Create connectivity matrix and arrays to record individual weights of all four connections.

        Arguments
        :param jee: Unscaled coupling strength from E to E neurons. Defaults to 25.
        :type jee: float
        :param jie: Unscaled coupling strength from E to I neurons. Defaults to 112.5.
        :type jie: float
        :param jei: Unscaled coupling strength from I to E neurons. Defaults to -150.
        :type jei: float
        :param jii: Unscaled coupling strength from I to I neurons. Defaults to -250.
        :type jii: float
        :param jex: Unscaled coupling strength from X to E neurons. Defaults to 180.
        :type jex: float
        :param jix: Unscaled coupling strength from X to I neurons. Defaults to 135.
        :type jix: float
        :param p_ee: Probability of connection from E to E neurons. Defaults to 0.1.
        :type p_ee: float
        :param p_ie: Probability of connection from E to I neurons. Defaults to 0.1.
        :type p_ie: float
        :param p_ei: Probability of connection from I to E neurons. Defaults to 0.1.
        :type p_ei: float
        :param p_ii: Probability of connection from I to I neurons. Defaults to 0.1.
        :type p_ii: float
        :param p_ex: Probability of connection from X to E neurons. Defaults to 0.1.
        :type p_ex: float
        :param p_ix: Probability of connection from X to I neurons. Defaults to 0.1.
        :type p_ix: float
        :param nJrecord0: Count of synaptic weights recorded. Relevant when network is plastic.
        :type nJrecord0: int

        Returns as part of self.
        :return: Recurrent connectivity matrix (J); External feedforward connectivity matrix (Jx);
        Indices of recorded EE synaptic weights (Jrecord_ee);
        Indices of recorded IE synaptic weights (Jrecord_ie); Indices of recorded EI synaptic weights (Jrecord_ei);
        Indices of recorded II synaptic weights (Jrecord_ii); Number of recorded EE synaptic weights (numrecordJ_ee);
        Number of recorded IE synaptic weights (numrecordJ_ie); Number of recorded EI synaptic weights (numrecordJ_ei);
        Number of recorded II synaptic weights (numrecordJ_ii).
        :rtype: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,  int, int, int, int)
        """
        # Test types
        if not isinstance(jee, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jee is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(jie, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jie is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(jei, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jei is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(jii, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jii is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(jex, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jex is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(jix, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jix is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(p_ee, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: p_ee is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(p_ie, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: p_ie is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(p_ei, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: p_ei is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(p_ii, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: p_ii is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(p_ex, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: p_ex is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(p_ix, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: p_ix is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(nJrecord0, (int, np.integer)):
            err = TypeError("ERROR: nJrecord0 is not int.")
            logging.exception(err)
            raise err
        # Value tests.
        if jee <= 0:
            err = ValueError("ERROR: jee, EE syn strength, has to be positive.")
            logging.exception(err)
            raise err
        if jie <= 0:
            err = ValueError("ERROR: jie, IE syn strength, has to be positive.")
            logging.exception(err)
            raise err
        if jei >= 0:
            err = ValueError("ERROR: jei, EI syn strength, has to be negative.")
            logging.exception(err)
            raise err
        if jii >= 0:
            err = ValueError("ERROR: jii, II syn strength, has to be negative.")
            logging.exception(err)
            raise err
        if jex <= 0:
            err = ValueError("ERROR: jex, EX syn strength, has to be positive.")
            logging.exception(err)
            raise err
        if jix <= 0:
            err = ValueError("ERROR: jix, IX syn strength, has to be positive.")
            logging.exception(err)
            raise err
        if (p_ee > 1) | (p_ee < 0):
            err = ValueError("ERROR: p_ee, EE Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (p_ie > 1) | (p_ie < 0):
            err = ValueError("ERROR: p_ie, IE Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (p_ei > 1) | (p_ei < 0):
            err = ValueError("ERROR: p_ei, EI Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (p_ii > 1) | (p_ii < 0):
            err = ValueError("ERROR: p_ii, II Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (p_ex > 1) | (p_ex < 0):
            err = ValueError("ERROR: p_ex, EX Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (p_ix > 1) | (p_ix < 0):
            err = ValueError("ERROR: p_ix, IX Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if nJrecord0 <= 0:
            err = ValueError("ERROR: nJrecord0, number of syn weights recorded, has to be positive.")
            logging.exception(err)
            raise err

        # Make defaults accessible.
        self.jee = jee
        self.jie = jie
        self.jei = jei
        self.jii = jii
        self.jex = jex
        self.jix = jix
        self.p_ee = p_ee
        self.p_ie = p_ie
        self.p_ei = p_ei
        self.p_ii = p_ii
        self.p_ex = p_ex
        self.p_ix = p_ix

        # Start the simulation.
        start_time = time.time()

        # Recurrent net connection probabilities.
        P = np.array([[p_ee, p_ei], [p_ie, p_ii]])
        # Ffwd connection probs.
        Px = np.array([[p_ex], [p_ix]])
        # Define mean field matrices.
        Jm = np.array([[jee, jei], [jie, jii]]) / np.sqrt(self.N)
        Jxm = np.array([[jex], [jix]]) / np.sqrt(self.N)

        # Define connectivity
        self.J = np.vstack(
            (
                np.hstack(
                    (
                        np.array(Jm[0, 0] * np.random.binomial(1, P[0, 0], (self.Ne, self.Ne))),
                        np.array(Jm[0, 1] * np.random.binomial(1, P[0, 1], (self.Ne, self.Ni))),
                    )
                ),
                np.hstack(
                    (
                        np.array(Jm[1, 0] * np.random.binomial(1, P[1, 0], (self.Ni, self.Ne))),
                        np.array(Jm[1, 1] * np.random.binomial(1, P[1, 1], (self.Ni, self.Ni))),
                    )
                ),
            )
        )

        self.Jx = np.vstack(
            (
                np.array(Jxm[0, 0] * np.random.binomial(1, Px[0, 0], (self.Ne, self.Nx))),
                np.array(Jxm[1, 0] * np.random.binomial(1, Px[1, 0], (self.Ni, self.Nx))),
            )
        )

        logging.info(f"Connectivity matrices J and Jx were built successfully.")

        # Define variables to record changes in weights.
        # Synaptic weights EE to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        IIJJ_rec = np.argwhere(self.J[0 : self.Ne, 0 : self.Ne])  # Find non-zero E to E weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        sampled_indices = np.random.choice(len(IIJJ_rec[:, 0]), size=nJrecord0, replace=False)
        II = II[sampled_indices]
        JJ = JJ[sampled_indices]
        self.Jrecord_ee = np.array([II, JJ])  # Record these
        self.numrecordJ_ee = len(JJ)

        # Synaptic weights IE to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        IIJJ_rec = np.argwhere(self.J[self.Ne : self.N, 0 : self.Ne])  # Find non-zero E to I weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        sampled_indices = np.random.choice(len(IIJJ_rec[:, 0]), size=nJrecord0, replace=False)
        II = II[sampled_indices]
        JJ = JJ[sampled_indices]
        self.Jrecord_ie = np.array([II + self.Ne, JJ])  # Record these
        self.numrecordJ_ie = len(JJ)

        # Synaptic weights EI to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        IIJJ_rec = np.argwhere(self.J[0 : self.Ne, self.Ne : self.N])  # Find non-zero I to E weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        sampled_indices = np.random.choice(len(IIJJ_rec[:, 0]), size=nJrecord0, replace=False)
        II = II[sampled_indices]
        JJ = JJ[sampled_indices]
        self.Jrecord_ei = np.array([II, JJ + self.Ne])  # Record these
        self.numrecordJ_ei = len(JJ)

        # Synaptic weights II to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        IIJJ_rec = np.argwhere(self.J[self.Ne : self.N, self.Ne : self.N])  # Find non-zero I to I weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        sampled_indices = np.random.choice(len(IIJJ_rec[:, 0]), size=nJrecord0, replace=False)
        II = II[sampled_indices]
        JJ = JJ[sampled_indices]
        self.Jrecord_ii = np.array([II + self.Ne, JJ + self.Ne])  # Record these
        self.numrecordJ_ii = len(JJ)

        elapsed_time = time.time() - start_time
        logging.info(f"Time for building connectivity matrix: {round(elapsed_time/60,2)} minutes.")

    def ffwd_spikes(self, T, cx=0.1, rx=10 / 1000, taujitter=5):
        """
        Create all spike trains of the Poisson feedforward, external layer.

        Arguments
        :param T: Total time of simulation.
        :type T: int
        :param cx: Value of mean correlation between feedforward Poisson spike trains. Defaults to 0.1.
        :type cx: float or int
        :param rx: Rate of feedforward Poisson neurons in Hz. Defaults to 0.01 kHz.
        :type rx: float or int
        :param taujitter: Spike trains are jittered by taujitter milliseconds to avoid perfect synchrony. Defaults to 5 ms.
        :type taujitter: float or int

        Returns (as part of `self`)
        :return: Feedforward, Poisson spike trains recorded as spike time and neuron index (sx);
        Total number of spikes in sx (nspikeX).
        :rtype: tuple(np.ndarray, int)
        """
        # None errors.
        if T is None:
            err = ValueError(
                """ERROR: T cannot be None. Pick an integer number of total simulation time.
                A number in the order of 10^3 is a good place to start for a quick simulation.
                """
            )
            logging.exception(err)
            raise err
        # Type errors
        if not isinstance(cx, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: cx is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(rx, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: rx is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(taujitter, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: taujitter is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(T, (int, np.integer)):
            err = TypeError("ERROR: T is not int")
            logging.exception(err)
            raise err
        # Value tests.
        if T < 1:
            err = ValueError("ERROR: T has to be greater than 1 ms.")
            logging.exception(err)
            raise err
        if (cx > 1) | (cx < 0):
            err = ValueError("ERROR: cx, input corrs, have to be between 0 and 1.")
            logging.exception(err)
            raise err
        if rx <= 0:
            err = ValueError("ERROR: rx, input rate, has to be greater than 0.")
            logging.exception(err)
            raise err
        if taujitter <= 0:
            err = ValueError("ERROR: taujitter has to be greater than 0.")
            logging.exception(err)
            raise err

        # Start the simulation.
        start_time = time.time()

        if cx < 1e-5:  # If uncorrelated
            self.nspikeX = np.random.poisson(self.Nx * rx * T)
            st = np.random.uniform(0, 1, (1, self.nspikeX)) * T
            self.sx = np.zeros((2, len(st[0])))
            self.sx[0, :] = np.sort(st)[0]  # spike time
            self.sx[1, :] = np.random.randint(1, self.Nx, (1, len(st[0])))  # neuron index that spiked
            logging.info(f"Uncorrelated ffwd spike trains (cx={cx} and rate={rx} kHz) were generated successfully.")
        else:  # If correlated
            rm = rx / cx  # Firing rate of mother process
            nstm = np.random.poisson(rm * T)  # Number of mother spikes
            stm = np.random.uniform(0, 1, (nstm, 1)) * T  # spike times of mother process
            maxnsx = int(T * rx * self.Nx * 1.2)  # Max num spikes
            sx = np.zeros((2, maxnsx))
            ns = 0
            for j in np.arange(1, self.Nx, 1):  # For each ffwd spike train
                ns0 = np.random.binomial(nstm, cx)  # Number of spikes for this spike train
                st = random2.sample(list(stm[:, 0]), ns0)  # Sample spike times randomly
                st = st + taujitter * np.random.normal(0, 1, size=len(st))  # jitter spike times
                st = st[(st > 0) & (st < T)]  # Get rid of out-of-bounds times
                ns0 = len(st)  # Re-compute spike count
                sx[0, ns + 1 : ns + ns0 + 1] = st  # Set the spike times and indices
                sx[1, ns + 1 : ns + ns0 + 1] = j
                ns = ns + ns0

            # Get rid of padded zeros
            sx = sx[:, sx[0, :] > 0]

            # Sort by spike time
            I = np.argsort(sx[0, :])
            self.sx = sx[:, I]
            self.nspikeX = len(sx[0, :])

            logging.info(f"Correlated ffwd spike trains (cx={cx} and rate={rx} kHz) were generated successfully.")

        elapsed_time = time.time() - start_time
        logging.info(f"Time for generating feedforward Poisson spike trains: {round(elapsed_time/60,2)} minutes.")

    def simulate(
        self,
        Cm=1,
        gL=1 / 15,
        VT=-55,
        Vre=-75,
        Vth=-50,
        EL=-72,
        DeltaT=1,
        taue=8,
        taui=4,
        taux=10,
        tauSTDP=200,
        numrecord=100,
        eta_ee_hebb=0,
        jmax_ee=30,
        eta_ee_koh=0,
        beta=2,
        eta_ie_homeo=0,
        rho_ie=0.020,
        eta_ie_hebb=0,
        jmax_ie_hebb=125,
        eta_ei=0,
        rho_ei=0.010,
        eta_ii=0,
        rho_ii=0.020,
    ):
        """
        Execute Network simulation.

        Arguments
        :param Cm: Membrane capacitance. Defaults to 1.
        :type Cm: float or int
        :param gL: Leak conductance. Defaults to 1/15.
        :type gL: float or int
        :param VT: Threshold in EIF neuron. Defaults to -55.
        :type VT: float or int
        :param Vre: Reset voltage. Defaults to -75.
        :type Vre: float or int
        :param Vth: Hard threshold that determines when a spike happened. Defaults to -50.
        :type Vth: float or int
        :param EL: Resting potential. Defaults to -72.
        :type EL: float or int
        :param DeltaT: EIF neuron parameter. Determines the shape of the rise to spike. Defaults to 1.
        :type DeltaT: float or int
        :param taue: Timescale of excitatory neurons in milliseconds. Defaults to 8 ms.
        :type taue: float or int
        :param taui: Timescale of inhibitory neurons in milliseconds. Defaults to 4 ms.
        :type taui: float or int
        :param taux: Timescale of external neurons in milliseconds. Defaults to 10 ms.
        :type taux: float or int
        :param tauSTDP: Timescale of eligibility trace used for STDP. Defaults to 200 ms.
        :type tauSTDP: float or int
        :param numrecord: Number of neurons to record currents and voltage from. Defaults to 100.
        :type numrecord: int
        :param eta_ee_hebb: Learning rate of EE Hebbian STDP. Defaults to 0.
        Pick a value in the approximate order of 10^-3 or lower as a start point.
        :type eta_ee_hebb: float or int
        :param Jmax_ee: Hard constraint on EE Hebbian STDP. Defaults to 30/np.sqrt(N).
        :type Jmax_ee: float or int
        :param eta_ee_koh: Learning rate of Kohonen STDP. Defaults to 0.
        Pick a value in the approximate order of 10^-3 or lower as a start point.
        :type eta_ee_koh: float or int
        :param beta: Parameter for Kohonen STDP. Defaults to 2/np.sqrt(N).
        :type beta: float or int
        :param eta_ie_homeo: Learning rate of iSTDP. Defaults to 0.
        Pick a value in the approximate order of 10^-3 or lower as a start point.
        :type eta_ie_homeo: float or int
        :param rho_ie: Target rate of I neurons in iSTDP. Defaults to 0.020 kHz.
        :type rho_ie: float or int
        :param eta_ie_hebb: Learning rate of IE Hebbian STDP. Defaults to 0.
        Pick a value in the approximate order of 10^-3 as a start point.
        :type eta_ie_hebb: float or int
        :param Jmax_ie_hebb: Hard constraint on IE Hebbian STDP. Defaults to 125/np.sqrt(N).
        :type Jmax_ie_hebb: float or int
        :param eta_ei: Learning rate of iSTDP. Defaults to 0.
        Pick a value in the approximate order of 10^-3 or lower as a start point.
        :type eta_ei: float or int
        :param rho_ei: Parameter that determines target rate in iSTDP. Defaults to 0.010 kHz.
        :type rho_ei: float or int
        :param eta_ii: Learning rate of iSTDP. Defaults to 0.
        Pick a value in the approximate order of 10^-3 or lower as a start point.
        :type eta_ii: float or int
        :param rho_ii: Parameter that determines target rate in iSTDP. Defaults to 0.020 kHz.
        :type rho_ii: float or int

        Returns
        :return: A tuple containing spike train of all neurons in recurrent neural network (s);
        spike train of all feedforward neurons (sx),
        matrices of neurons (rows) by time bins (cols) for EE, EI, IE,
        and II recorded weights (JRec_ee, JRec_ie, JRec_ei, JRec_ii);
        matrices of neurons (rows) by time bins (cols) for E, I, and X input currents (IeRec, IiRec, IxRec);
        matrix of neurons (rows) by time bins (cols) for recurrent network voltages (VRec);
        and discretized recorded time domain (timeRecord).
        :rtype: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray)

        """
        # Type errors.
        if not isinstance(Cm, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: Cm is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(gL, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: gL is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(VT, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: VT is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(Vre, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: Vre is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(Vth, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: Vth is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(EL, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: EL is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(DeltaT, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: DeltaT is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(taue, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: taue is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(taui, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: taui is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(taux, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: taux is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(tauSTDP, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: tauSTDP is not int nor float.")
            logging.exception(err)
            raise err
        if not isinstance(eta_ee_hebb, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: eta_ee_hebb is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(jmax_ee, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jmax_ee is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(eta_ee_koh, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: eta_ee_koh is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(beta, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: beta is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(eta_ie_homeo, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: eta_ie_homeo is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(rho_ie, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: rho_ie is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(eta_ie_hebb, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: eta_ie_hebb is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(jmax_ie_hebb, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: jmax_ie_hebb is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(eta_ei, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: eta_ei is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(rho_ei, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: rho_ei is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(eta_ii, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: eta_ii is not int,float.")
            logging.exception(err)
            raise err
        if not isinstance(rho_ii, (float, np.floating, int, np.integer)):
            err = TypeError("ERROR: rho_ii is not int,float.")
            logging.exception(err)
            raise err
        # Value tests.
        if Cm <= 0:
            err = ValueError("ERROR: Cm has to be greater than zero.")
            logging.exception(err)
            raise err
        if gL <= 0:
            err = ValueError("ERROR: gL has to be greater than zero.")
            logging.exception(err)
            raise err
        if VT > 0:
            err = ValueError("ERROR: VT has to be less than zero.")
            logging.exception(err)
            raise err
        if Vre > 0:
            err = ValueError("ERROR: Vre has to be less than zero.")
            logging.exception(err)
            raise err
        if Vth > 0:
            err = ValueError("ERROR: Vth has to be less than zero.")
            logging.exception(err)
            raise err
        if EL > 0:
            err = ValueError("ERROR: EL has to be less than zero.")
            logging.exception(err)
            raise err
        if DeltaT <= 0:
            err = ValueError("ERROR: DeltaT has to be greater than zero.")
            logging.exception(err)
            raise err
        if taue <= 0:
            err = ValueError("ERROR: taue has to be greater than zero.")
            logging.exception(err)
            raise err
        if taui <= 0:
            err = ValueError("ERROR: taui has to be greater than zero.")
            logging.exception(err)
            raise err
        if taux <= 0:
            err = ValueError("ERROR: taux has to be greater than zero.")
            logging.exception(err)
            raise err
        if tauSTDP <= 0:
            err = ValueError("ERROR: tauSTDP has to be greater than zero.")
            logging.exception(err)
            raise err
        if numrecord <= 0:
            err = ValueError("ERROR: numrecord has to be greater than zero.")
            logging.exception(err)
            raise err
        if eta_ee_hebb < 0:
            err = ValueError("ERROR: eta_ee_hebb has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if jmax_ee <= 0:
            err = ValueError("ERROR: Jmax_ee has to be greater than zero.")
            logging.exception(err)
            raise err
        if eta_ee_koh < 0:
            err = ValueError("ERROR: eta_ee_koh has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if beta <= 0:
            err = ValueError("ERROR: beta has to be greater than zero.")
            logging.exception(err)
            raise err
        if eta_ie_homeo < 0:
            err = ValueError("ERROR: eta_ie_homeo has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if rho_ie <= 0:
            err = ValueError("ERROR: rho_ie has to be greater than zero.")
            logging.exception(err)
            raise err
        if eta_ie_hebb < 0:
            err = ValueError("ERROR: eta_ie_hebb has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if jmax_ie_hebb <= 0:
            err = ValueError("ERROR: jmax_ie_hebb has to be greater than zero.")
            logging.exception(err)
            raise err
        if eta_ei < 0:
            err = ValueError("ERROR: eta_ei has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if rho_ei <= 0:
            err = ValueError("ERROR: rho_ei has to be greater than zero.")
            logging.exception(err)
            raise err
        if eta_ii < 0:
            err = ValueError("ERROR: eta_ii has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if rho_ii <= 0:
            err = ValueError("ERROR: rho_ii has to be greater than zero.")
            logging.exception(err)
            raise err
        # We cannot allow for two plasticity rules to act on the same synapse type at the same time.
        if (eta_ee_hebb != 0) & (eta_ee_koh != 0):
            err = ValueError("ERROR: You cannot have both Kohonen and Hebb STDP's at the same time on EE connections!")
            logging.exception(err)
            raise err
        if (eta_ie_hebb != 0) & (eta_ie_homeo != 0):
            err = ValueError(
                "ERROR: You cannot have both Homeostatic and Hebb STDP's at the same time on IE connections!"
            )
            logging.exception(err)
            raise err

        # Make defaults accessible.
        self.tauSTDP = tauSTDP
        self.beta = beta
        self.jmax_ee = jmax_ee
        self.jmax_ie_hebb = jmax_ie_hebb
        # Initialize some variables
        alpha_ie = 2 * rho_ie * tauSTDP
        alpha_ei = 2 * rho_ei * tauSTDP
        alpha_ii = 2 * rho_ii * tauSTDP
        beta = beta / np.sqrt(self.N)
        Jmax_ee = jmax_ee / np.sqrt(self.N)
        Jmax_ie_hebb = jmax_ie_hebb / np.sqrt(self.N)
        Jnorm_ie = 112.5 / np.sqrt(self.N)
        Jnorm_ei = -150 / np.sqrt(self.N)
        Jnorm_ii = -250 / np.sqrt(self.N)
        # Random initial voltages
        V0 = np.random.uniform(0, 1, (1, self.N)) * (VT - Vre) + Vre
        V = V0
        # Initialize current vectors.
        Ie = np.zeros((1, self.N))
        Ii = np.zeros((1, self.N))
        Ix = np.zeros((1, self.N))
        # Initialize eligibility traces.
        x = np.zeros((1, self.N))
        # Sample neurons to record from.
        Irecord = np.array(
            [
                [
                    random2.sample(list(np.arange(0, self.Ne)), numrecord),
                    random2.sample(list(np.arange(self.Ne, self.N)), numrecord),
                ]
            ]
        )
        Ierecord = np.sort(Irecord[0, 0]).astype(int)
        Iirecord = np.sort(Irecord[0, 1]).astype(int)
        Ixrecord = np.sort(random2.sample(list(np.arange(0, self.Ne)), numrecord)).astype(int)
        Vrecord = (
            np.sort(
                [
                    [
                        random2.sample(list(np.arange(0, self.Ne)), int(round(numrecord / 2))),
                        random2.sample(list(np.arange(self.Ne, self.N)), int(round(numrecord / 2))),
                    ]
                ]
            )[0]
            .reshape(1, numrecord)
            .astype(int)[0]
        )
        del Irecord
        # Initialize recorded currents vectors.
        IeRec = np.zeros((numrecord, self.Ntrec))
        IiRec = np.zeros((numrecord, self.Ntrec))
        IxRec = np.zeros((numrecord, self.Ntrec))
        VRec = np.zeros((numrecord, self.Ntrec))
        JRec_ee = np.zeros((self.numrecordJ_ee, self.Ntrec))
        JRec_ie = np.zeros((self.numrecordJ_ie, self.Ntrec))
        JRec_ei = np.zeros((self.numrecordJ_ei, self.Ntrec))
        JRec_ii = np.zeros((self.numrecordJ_ii, self.Ntrec))
        # Initial spike related variables.
        iFspike = 0
        # s(0,:) are the spike times
        # s(1,:) are the associated neuron indices
        s = np.zeros((2, self.maxns))
        nspike = 0

        logging.info(f"We will record currents and membrane potential from {numrecord} E and {numrecord} I neurons.")

        # If there is EE Hebbian plasticity
        if eta_ee_hebb != 0:
            logging.info("EE connections will evolve according to Hebbian STDP.")
            logging.info(f"We will record {self.numrecordJ_ee} plastic EE weights.")
        # If there is EE Kohonen plasticity
        if eta_ee_koh != 0:
            logging.info("EE connections will evolve according to Kohonen's rule.")
            logging.info(f"We will record {self.numrecordJ_ee} plastic EE weights.")
        # If there is IE *Homeo* plasticity
        if eta_ie_homeo != 0:
            logging.info("IE connections will evolve according to homeostatic STDP.")
            logging.info(f"We will record {self.numrecordJ_ie} plastic IE weights.")
        # If there is IE *Hebbian* plasticity
        if eta_ie_hebb != 0:
            logging.info("IE connections will evolve according to Hebbian STDP.")
            logging.info(f"We will record {self.numrecordJ_ie} plastic IE weights.")
        # If there is EI plasticity
        if eta_ei != 0:
            logging.info("EI connections will evolve according to homeosatic iSTDP.")
            logging.info(f"We will record {self.numrecordJ_ei} plastic EI weights.")
        # If there is II plasticity
        if eta_ii != 0:
            logging.info("II connections will evolve according to homeostatic iSTDP.")
            logging.info(f"We will record {self.numrecordJ_ii} plastic II weights.")

        # Start the simulation.
        start_time = time.time()

        for i in range(len(self.total_time)):
            # Propagate ffwd spikes
            while (self.sx[0, iFspike] <= self.total_time[i]) & (iFspike < self.nspikeX - 1):
                jpre = int(self.sx[1, iFspike])
                Ix += self.Jx[:, jpre] / taux
                iFspike += 1

            # Euler update to V
            V += (
                self.dt
                / Cm
                * (self.Istim[i] * self.Jstim + Ie + Ii + Ix + gL * (EL - V) + gL * DeltaT * np.exp((V - VT) / DeltaT))
            )

            # Find which neurons spiked
            Ispike = np.argwhere(V >= Vth)[:, 1]

            # If there are spikes
            if len(Ispike) != 0:
                # Store spike times and neuron indices
                if nspike + len(Ispike) <= self.maxns:
                    s[0, nspike + 1 : nspike + len(Ispike) + 1] = self.total_time[i]
                    s[1, nspike + 1 : nspike + len(Ispike) + 1] = Ispike
                else:
                    logging.error("Stopped simulation. Too many spikes.")
                    sys.exit(1)

                # Update synaptic currents
                Ie += np.sum(self.J[:, Ispike[Ispike <= self.Ne]], 1) / taue
                Ii += np.sum(self.J[:, Ispike[Ispike > self.Ne]], 1) / taui

                # If there is EE Hebbian plasticity
                if eta_ee_hebb != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to E after a pre spike
                    self.J[0 : self.Ne, Ispike[Ispike <= self.Ne]] -= np.tile(
                        eta_ee_hebb * (x[0, 0 : self.Ne]),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (self.J[0 : self.Ne, Ispike[Ispike <= self.Ne]])
                    # E to E after a post spike
                    self.J[Ispike[Ispike < self.Ne], 0 : self.Ne] += (
                        np.tile(
                            eta_ee_hebb * x[0, 0 : self.Ne].transpose(),
                            (np.count_nonzero(Ispike < self.Ne), 1),
                        )
                        * Jmax_ee
                        * (self.J[Ispike[Ispike < self.Ne], 0 : self.Ne] != 0)
                    )

                # If there is EE Kohonen plasticity
                if eta_ee_koh != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to E after a pre spike
                    self.J[0 : self.Ne, Ispike[Ispike <= self.Ne]] += np.tile(
                        beta * eta_ee_koh * (x[0, 0 : self.Ne]),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (self.J[0 : self.Ne, Ispike[Ispike <= self.Ne]] != 0)
                    # E to E after a post spike
                    self.J[Ispike[Ispike < self.Ne], 0 : self.Ne] -= (
                        eta_ee_koh * self.J[Ispike[Ispike < self.Ne], 0 : self.Ne]
                    )

                # If there is IE *Homeo* plasticity
                if eta_ie_homeo != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to I after a pre spike
                    self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]] -= np.tile(
                        eta_ie_homeo / Jnorm_ie * (x[0, self.Ne : self.N] - alpha_ie),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]])
                    # E to I after a post spike
                    self.J[Ispike[Ispike > self.Ne], 0 : self.Ne] -= np.tile(
                        eta_ie_homeo / Jnorm_ie * x[0, 0 : self.Ne].transpose(),
                        (np.count_nonzero(Ispike > self.Ne), 1),
                    ) * (self.J[Ispike[Ispike > self.Ne], 0 : self.Ne])

                # If there is IE *Hebbian* plasticity
                if eta_ie_hebb != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to I after a pre spike
                    self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]] -= np.tile(
                        eta_ie_hebb * (x[0, self.Ne : self.N]),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]] != 0)
                    # E to I after a post spike
                    self.J[Ispike[Ispike > self.Ne], 0 : self.Ne] += (
                        np.tile(
                            eta_ie_hebb * x[0, 0 : self.Ne].transpose(),
                            (np.count_nonzero(Ispike > self.Ne), 1),
                        )
                        * Jmax_ie_hebb
                        * (self.J[Ispike[Ispike > self.Ne], 0 : self.Ne] != 0)
                    )

                # If there is EI plasticity
                if eta_ei != 0:
                    # Update synaptic weights according to plasticity rules
                    # I to E after an I spike
                    self.J[0 : self.Ne, Ispike[Ispike >= self.Ne]] -= np.tile(
                        eta_ei / Jnorm_ei * (x[0, 0 : self.Ne] - alpha_ei),
                        (np.count_nonzero(Ispike >= self.Ne), 1),
                    ).transpose() * (self.J[0 : self.Ne, Ispike[Ispike >= self.Ne]])
                    # I to E after an E spike
                    self.J[Ispike[Ispike < self.Ne], self.Ne : self.N] -= np.tile(
                        eta_ei / Jnorm_ei * x[0, self.Ne : self.N].transpose(),
                        (np.count_nonzero(Ispike < self.Ne), 1),
                    ) * (self.J[Ispike[Ispike < self.Ne], self.Ne : self.N])

                # If there is II plasticity
                if eta_ii != 0:
                    # Update synaptic weights according to plasticity rules
                    # I to E after an I spike
                    self.J[self.Ne : self.N, Ispike[Ispike >= self.Ne]] -= np.tile(
                        eta_ii / Jnorm_ii * (x[0, self.Ne : self.N] - alpha_ii),
                        (np.count_nonzero(Ispike >= self.Ne), 1),
                    ).transpose() * (self.J[self.Ne : self.N, Ispike[Ispike >= self.Ne]])
                    # I to E after an E spike
                    self.J[Ispike[Ispike > self.Ne], self.Ne : self.N] -= np.tile(
                        eta_ii / Jnorm_ii * x[0, self.Ne : self.N].transpose(),
                        (np.count_nonzero(Ispike > self.Ne), 1),
                    ) * (self.J[Ispike[Ispike > self.Ne], self.Ne : self.N])

                # Update rate estimates for plasticity rules
                x[0, Ispike] += 1

                # Update cumulative number of spikes
                nspike += len(Ispike)

            # Euler update to synaptic currents
            Ie -= self.dt * Ie / taue
            Ii -= self.dt * Ii / taui
            Ix -= self.dt * Ix / taux

            # Update time-dependent firing rates for plasticity
            x[0 : self.Ne] -= self.dt * x[0 : self.Ne] / tauSTDP
            x[self.Ne : self.N] -= self.dt * x[self.Ne : self.N] / tauSTDP

            # This makes plots of V(t) look better.
            # All action potentials reach Vth exactly.
            # This has no real effect on the network sims
            V[0, Ispike] = Vth

            # Store recorded variables
            ii = int(math.floor(i / self.nBinsRecord))
            IeRec[:, ii] += Ie[0, Ierecord]
            IiRec[:, ii] += Ii[0, Iirecord]
            IxRec[:, ii] += Ix[0, Ixrecord]
            VRec[:, ii] += V[0, Vrecord]
            JRec_ee[:, ii] += self.J[self.Jrecord_ee[0, :], self.Jrecord_ee[1, :]]
            JRec_ie[:, ii] += self.J[self.Jrecord_ie[0, :], self.Jrecord_ie[1, :]]
            JRec_ei[:, ii] += self.J[self.Jrecord_ei[0, :], self.Jrecord_ei[1, :]]
            JRec_ii[:, ii] += self.J[self.Jrecord_ii[0, :], self.Jrecord_ii[1, :]]

            # Reset mem pot.
            V[0, Ispike] = Vre

        IeRec = IeRec / self.nBinsRecord  # Normalize recorded variables by # bins
        IiRec = IiRec / self.nBinsRecord
        IxRec = IxRec / self.nBinsRecord
        VRec = VRec / self.nBinsRecord
        JRec_ee = JRec_ee * np.sqrt(self.N) / self.nBinsRecord
        JRec_ie = JRec_ie * np.sqrt(self.N) / self.nBinsRecord
        JRec_ei = JRec_ei * np.sqrt(self.N) / self.nBinsRecord
        JRec_ii = JRec_ii * np.sqrt(self.N) / self.nBinsRecord

        s = s[:, 0:nspike]  # Get rid of padding in s

        logging.info(f"The plastic balanced network has been simulated successfully.")

        elapsed_time = time.time() - start_time
        logging.info(f"Time for simulation: {round(elapsed_time/60,2)} minutes.")

        return (
            s,
            self.sx,
            JRec_ee,
            JRec_ie,
            JRec_ei,
            JRec_ii,
            IeRec,
            IiRec,
            IxRec,
            VRec,
            self.timeRecord,
        )


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
    timeVector = np.arange(dtRate, T + dtRate, dtRate) / 1000

    hist, bin_edges = np.histogram(s[0, s[1, :] < frac_exc * N], bins=len(timeVector))
    eRateT = hist / (dtRate * frac_exc * N) * 1000
    hist, bin_edges = np.histogram(s[0, s[1, :] >= frac_exc * N], bins=len(timeVector))
    iRateT = hist / (dtRate * (1 - frac_exc) * N) * 1000

    # Smooth rates. We multiplied by 1000 to get them in units of Hz.
    eRateT = np.convolve(eRateT, np.ones(window_size) / window_size, mode="same")
    iRateT = np.convolve(iRateT, np.ones(window_size) / window_size, mode="same")

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
    return np.array(np.cov(counts.transpose()))


def cov2corr(cov):
    """convert covariance matrix to correlation matrix

    Arguments
    :param cov: Covariance matrix.
    :type cov: np.ndarray

    Returns (as part of `self`)
    :return corr: Full spike count correlation matrix.
    :rtype corr: np.ndarray
    """
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
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
    # Get mean spike count covariances over each sub-pop
    II, JJ = np.meshgrid(np.arange(0, N), np.arange(0, N))
    mCee = np.nanmean(C[(II < frac_exc * N) & (JJ < II)])
    mCei = np.nanmean(C[(II < frac_exc * N) & (JJ >= frac_exc * N)])
    mCii = np.nanmean(C[(II > frac_exc * N) & (JJ > II)])

    # Mean-field spike count cov matrix
    # Compare this to the theoretical prediction
    mC = [[mCee, mCei], [mCei, mCii]]
    return mC
