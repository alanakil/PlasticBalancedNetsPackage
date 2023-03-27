"""
Helper functions for main code.
"""
__author__ = "Alan Akil (alan.akil@yahoo.com)"
__date__ = "MARCH 2023"

#%%
import numpy as np
import random2
import math
import time
import logging

#%%
#%%
class plasticNeuralNetwork:
    def __init__(
        self,
        N,
        frac_exc,
        frac_ext,
        T,
        dt,
        jestim,
        jistim,
        nBinsRecord,
    ):
        """
        plasticNeuralNetwork is a class that builds a neural network with
        correlated or uncorrelated firing as well as with plastic or static
        synaptic weights on any connection.
        It contains functions to define the connectivity, simulate feedforward external
        spike trains, and to simulate the recurrent network's firing.

        Inputs
        N : Total number of neurons in recurrent neural network.
        frac_exc : Fraction of excitatory neurons. Typically 0.8.
        frac_ext : Fraction of excitatory neurons in external layer. Typically 0.2.
        T : Total time of simulation in milliseconds.
        dt : Time bin size for time discretization.
        jestim : Added constant current to excitatory neurons.
        jistim : Added constant current to inhibitory neurons.
        nBinsRecord : Number of bins to record average and record over.

        Returns
        Ne : Number of excitatory neurons.
        Ni : Number of inhibitory neurons.
        Nx : Number of external neurons.
        Nt : Total number of discretized time points.
        Istim : Time vector of added constant stimulation.
        Jstim : Weight coupling for Istim.
        maxns : Maximum number of spikes to terminate pathologic behavior.
        timeRecord : Discretized recorded time domain.
        Ntrec : Number of points in discretized recorded time domain.
        """
        # Type tests.
        if type(N) not in [int]:
            err = TypeError("ERROR: N is not int")
            logging.exception(err)
            raise err
        if type(frac_exc) not in [int, float]:
            err = TypeError("ERROR: frac_exc is not one of these - int, float")
            logging.exception(err)
            raise err
        if type(frac_ext) not in [int, float]:
            err = TypeError("ERROR: frac_exc is not one of these - int, float")
            logging.exception(err)
            raise err
        if type(T) not in [int]:
            err = TypeError("ERROR: T is not int")
            logging.exception(err)
            raise err
        if type(dt) not in [int, float]:
            err = TypeError("ERROR: dt is not one of these - int, float")
            logging.exception(err)
            raise err
        if type(jestim) not in [int, float]:
            err = TypeError("ERROR: jestim is not one of these - int, float")
            logging.exception(err)
            raise err
        if type(jistim) not in [int, float]:
            err = TypeError("ERROR: jistim is not one of these - int, float")
            logging.exception(err)
            raise err
        if type(nBinsRecord) not in [int]:
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

        self.N = N
        self.Ne = int(round(frac_exc * N))
        self.Ni = int(round((1 - frac_exc) * N))
        self.Nx = int(round(frac_ext * N))
        self.Nt = round(T / dt)
        self.total_time = np.arange(dt, T + dt, dt)  # Time discretized domain.
        self.Istim = np.zeros(len(self.total_time))
        self.Istim[self.total_time > T / 2] = 0
        self.Jstim = np.sqrt(N) * np.hstack(
            (jestim * np.ones((1, self.Ne)), jistim * np.ones((1, self.Ni)))
        )
        self.maxns = round(0.05 * N * T)  # Max num of spikes (50 Hz).
        dtRecord = nBinsRecord * dt
        self.timeRecord = np.arange(dtRecord, T + dtRecord, dtRecord)
        self.Ntrec = len(self.timeRecord)

    def connectivity(self, Jm, Jxm, P, Px, nJrecord0):
        """
        Create connectivity matrix and arrays to record individual weights of all four connections.
        Jm : Mean field matrix J for recurrent connections. It contains avg value of connection for recurrent connections.
        Jxm : Mean field matrix Jx for feedforward connections. It contains avg value of connection for feedforward connections.
        P : Matrix containing probability of connection for each pair of populations.
        Px : Matrix containing probability of connection for each pair of populations from external to recurrent.
        nJrecord0 : Count of synaptic weights recorded. Relevant when network is plastic.

        Returns (as part of `self`)
        J : Recurrent connectivity matrix.
        Jx : External feedforward connectivity matrix.
        Jrecord_ee : Indices of recorded EE synaptic weights.
        Jrecord_ie : Indices of recorded IE synaptic weights.
        Jrecord_ei : Indices of recorded EI synaptic weights.
        Jrecord_ii : Indices of recorded II synaptic weights.
        numrecordJ_ee : Number of recorded EE synaptic weights.
        numrecordJ_ie : Number of recorded IE synaptic weights.
        numrecordJ_ei : Number of recorded EI synaptic weights.
        numrecordJ_ii : Number of recorded II synaptic weights.
        """
        # Test types
        if type(Jm) not in [np.ndarray]:
            err = TypeError("ERROR: Jm is not np.array.")
            logging.exception(err)
            raise err
        if type(Jxm) not in [np.ndarray]:
            err = TypeError("ERROR: Jm is not np.array.")
            logging.exception(err)
            raise err
        if type(P) not in [np.ndarray]:
            err = TypeError("ERROR: P is not np.array.")
            logging.exception(err)
            raise err
        if type(Px) not in [np.ndarray]:
            err = TypeError("ERROR: Px is not np.array.")
            logging.exception(err)
            raise err
        if type(nJrecord0) not in [int]:
            err = TypeError("ERROR: nJrecord0 is not int.")
            logging.exception(err)
            raise err
        # Value tests.
        if Jm[0,0] <= 0:
            err = ValueError("ERROR: Jm[0,0], EE syn strength, has to be positive.")
            logging.exception(err)
            raise err
        if Jm[1,0] <= 0:
            err = ValueError("ERROR: Jm[1,0], IE syn strength, has to be positive.")
            logging.exception(err)
            raise err
        if Jm[0,1] >= 0:
            err = ValueError("ERROR: Jm[0,1], EI syn strength, has to be negative.")
            logging.exception(err)
            raise err
        if Jm[1,1] >= 0:
            err = ValueError("ERROR: Jm[1,1], II syn strength, has to be negative.")
            logging.exception(err)
            raise err
        if Jxm[0,0] <= 0:
            err = ValueError("ERROR: Jxm[0,0], EX syn strength, has to be positive.")
            logging.exception(err)
            raise err
        if Jxm[1,0] <= 0:
            err = ValueError("ERROR: Jxm[1,0], IX syn strength, has to be positive.")
            logging.exception(err)
            raise err
        if (P[0,0] > 1) | (P[0,0] < 0):
            err = ValueError("ERROR: P[0,0], EE Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (P[1,0] > 1) | (P[1,0] < 0):
            err = ValueError("ERROR: P[1,0], IE Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (P[0,1] > 1) | (P[0,1] < 0):
            err = ValueError("ERROR: P[0,1], EI Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (P[1,1] > 1) | (P[1,1] < 0):
            err = ValueError("ERROR: P[1,1], II Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (Px[0,0] > 1) | (Px[0,0] < 0):
            err = ValueError("ERROR: Px[0,0], EX Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if (Px[1,0] > 1) | (Px[1,0] < 0):
            err = ValueError("ERROR: P[0,0], IX Prob of connection, has to be between 0 and 1.")
            logging.exception(err)
            raise err
        if nJrecord0 <= 0:
            err = ValueError("ERROR: nJrecord0, number of syn weights recorded, has to be positive.")
            logging.exception(err)
            raise err

        # Define connectivity
        self.J = np.vstack(
            (
                np.hstack(
                    (
                        np.array(
                            Jm[0, 0]
                            * np.random.binomial(1, P[0, 0], (self.Ne, self.Ne))
                        ),
                        np.array(
                            Jm[0, 1]
                            * np.random.binomial(1, P[0, 1], (self.Ne, self.Ni))
                        ),
                    )
                ),
                np.hstack(
                    (
                        np.array(
                            Jm[1, 0]
                            * np.random.binomial(1, P[1, 0], (self.Ni, self.Ne))
                        ),
                        np.array(
                            Jm[1, 1]
                            * np.random.binomial(1, P[1, 1], (self.Ni, self.Ni))
                        ),
                    )
                ),
            )
        )

        self.Jx = np.vstack(
            (
                np.array(
                    Jxm[0, 0] * np.random.binomial(1, Px[0, 0], (self.Ne, self.Nx))
                ),
                np.array(
                    Jxm[1, 0] * np.random.binomial(1, Px[1, 0], (self.Ni, self.Nx))
                ),
            )
        )

        # Define variables to record changes in weights.
        # Synaptic weights EE to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        IIJJ_rec = np.argwhere(
            self.J[0 : self.Ne, 0 : self.Ne]
        )  # Find non-zero E to E weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        sampled_indices = np.random.choice(
            len(IIJJ_rec[:, 0]), size=nJrecord0, replace=False
        )
        II = II[sampled_indices]
        JJ = JJ[sampled_indices]
        self.Jrecord_ee = np.array([II, JJ])  # Record these
        self.numrecordJ_ee = len(JJ)

        # Synaptic weights IE to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        IIJJ_rec = np.argwhere(
            self.J[self.Ne : self.N, 0 : self.Ne]
        )  # Find non-zero E to I weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        sampled_indices = np.random.choice(
            len(IIJJ_rec[:, 0]), size=nJrecord0, replace=False
        )
        II = II[sampled_indices]
        JJ = JJ[sampled_indices]
        self.Jrecord_ie = np.array([II + self.Ne, JJ])  # Record these
        self.numrecordJ_ie = len(JJ)

        # Synaptic weights EI to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        IIJJ_rec = np.argwhere(
            self.J[0 : self.Ne, self.Ne : self.N]
        )  # Find non-zero I to E weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        sampled_indices = np.random.choice(
            len(IIJJ_rec[:, 0]), size=nJrecord0, replace=False
        )
        II = II[sampled_indices]
        JJ = JJ[sampled_indices]
        self.Jrecord_ei = np.array([II, JJ + self.Ne])  # Record these
        self.numrecordJ_ei = len(JJ)

        # Synaptic weights II to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        IIJJ_rec = np.argwhere(
            self.J[self.Ne : self.N, self.Ne : self.N]
        )  # Find non-zero I to I weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        sampled_indices = np.random.choice(
            len(IIJJ_rec[:, 0]), size=nJrecord0, replace=False
        )
        II = II[sampled_indices]
        JJ = JJ[sampled_indices]
        self.Jrecord_ii = np.array([II + self.Ne, JJ + self.Ne])  # Record these
        self.numrecordJ_ii = len(JJ)

        return None

    def ffwd_spikes(self, cx, rx, taujitter, T):
        """
        Create all spike trains of the Poisson feedforward, external layer.
        cx : Value of mean correlation between feedforward Poisson spike trains.
        rx : Rate of feedforward Poisson neurons in Hz.
        taujitter : Spike trains are jittered by taujitter milliseconds to avoid perfect synchrony.
        T : Total time of simulation.
        
        Returns (as part of `self`)
        sx : Feedforward, Poisson spike trains recorded as spike time and neuron index.
        nspikeX : Total number of spikes in sx.
        """
        # Type errors
        if type(cx) not in [int, float]:
            err = TypeError("ERROR: cx is not int nor float.")
            logging.exception(err)
            raise err
        if type(rx) not in [int, float]:
            err = TypeError("ERROR: rx is not int nor float.")
            logging.exception(err)
            raise err
        if type(taujitter) not in [int, float]:
            err = TypeError("ERROR: taujitter is not int nor float.")
            logging.exception(err)
            raise err
        if type(T) not in [int]:
            err = TypeError("ERROR: T is not int")
            logging.exception(err)
            raise err
        # Value tests.
        if (cx > 1) | (cx <= 0):
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
        if T < 1:
            err = ValueError("ERROR: T has to be greater than 0.")
            logging.exception(err)
            raise err

        if cx < 1e-5:  # If uncorrelated
            self.nspikeX = np.random.poisson(self.Nx * rx * T)
            st = np.random.uniform(0, 1, (1, self.nspikeX)) * T
            self.sx = np.zeros((2, len(st[0])))
            self.sx[0, :] = np.sort(st)[0]  # spike time
            self.sx[1, :] = np.random.randint(
                1, self.Nx, (1, len(st[0]))
            )  # neuron index that spiked
        else:  # If correlated
            rm = rx / cx  # Firing rate of mother process
            nstm = np.random.poisson(rm * T)  # Number of mother spikes
            stm = (
                np.random.uniform(0, 1, (nstm, 1)) * T
            )  # spike times of mother process
            maxnsx = int(T * rx * self.Nx * 1.2)  # Max num spikes
            sx = np.zeros((2, maxnsx))
            ns = 0
            for j in np.arange(1, self.Nx, 1):  # For each ffwd spike train
                ns0 = np.random.binomial(
                    nstm, cx
                )  # Number of spikes for this spike train
                st = random2.sample(list(stm[:, 0]), ns0)  # Sample spike times randomly
                st = st + taujitter * np.random.normal(
                    0, 1, size=len(st)
                )  # jitter spike times
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

        return None

    def simulate(
        self,
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
    ):
        """
        Execute Network simulation.
        Cm : Membrane capacitance.
        gL : Leak conductance.
        VT : Threshold in EIF neuron.
        Vre : Reset voltage.
        Vth : Hard threshold that determines when a spike happened.
        EL : Resting potential.
        DeltaT : EIF neuron parameter. Determines the shape of the rise to spike.
        taue : Timescale of excitatory neurons in milliseconds.
        taui : Timescale of inhibitory neurons in milliseconds.
        taux  : Timescale of external neurons in milliseconds.
        tauSTDP :  Timescale of eligibility trace used for STDP.
        numrecord : Number of neurons to record currents and voltage from.
        eta_ee_hebb : Learning rate of EE Hebbian STDP.
        Jmax_ee : Hard constraint on EE Hebbian STDP.
        eta_ee_koh : Learning rate of Kohonen STDP.
        beta : Parameter for Kohonen STDP.
        eta_ie_homeo : Learning rate of iSTDP.
        alpha_ie : Parameter that determines target rate in iSTDP.
        eta_ie_hebb : Learning rate of IE Hebbian STDP.
        Jmax_ie_hebb : Hard constraint on IE Hebbian STDP.
        eta_ei : Learning rate of iSTDP.
        alpha_ei : Parameter that determines target rate in iSTDP.
        eta_ii : Learning rate of iSTDP.
        alpha_ii : Parameter that determines target rate in iSTDP.
        dt : Time bin size in ms.
        nBinsRecord : Number of bins to record average and record over.

        Returns
        s : Spike trains of all neurons in recurrent network, recorded by neuron index and spike time.
        sx : Same as sx coming from `ffwd_spikes()`.
        JRec_ee : Matrix of neurons (rows) by time bins (cols) for EE recorded weights.
        JRec_ie : Matrix of neurons (rows) by time bins (cols) for IE recorded weights.
        JRec_ei : Matrix of neurons (rows) by time bins (cols) for EI recorded weights.
        JRec_ii : Matrix of neurons (rows) by time bins (cols) for II recorded weights.
        IeRec : Matrix of neurons (rows) by time bins (cols) for E input currents.
        IiRec : Matrix of neurons (rows) by time bins (cols) for I input currents.
        IxRec : Matrix of neurons (rows) by time bins (cols) for X input currents.
        VRec : Matrix of neurons (rows) by time bins (cols) for recurrent network voltages.
        timeRecord : Discretized, recorded time domain.
        """
        # Type errors.
        if type(Cm) not in [int,float]:
            err = TypeError("ERROR: Cm is not int nor float.")
            logging.exception(err)
            raise err
        if type(gL) not in [int,float]:
            err = TypeError("ERROR: gL is not int nor float.")
            logging.exception(err)
            raise err
        if type(VT) not in [int,float]:
            err = TypeError("ERROR: VT is not int nor float.")
            logging.exception(err)
            raise err
        if type(Vre) not in [int,float]:
            err = TypeError("ERROR: Vre is not int nor float.")
            logging.exception(err)
            raise err
        if type(Vth) not in [int,float]:
            err = TypeError("ERROR: Vth is not int nor float.")
            logging.exception(err)
            raise err
        if type(EL) not in [int,float]:
            err = TypeError("ERROR: EL is not int nor float.")
            logging.exception(err)
            raise err
        if type(DeltaT) not in [int,float]:
            err = TypeError("ERROR: DeltaT is not int nor float.")
            logging.exception(err)
            raise err
        if type(taue) not in [int,float]:
            err = TypeError("ERROR: taue is not int nor float.")
            logging.exception(err)
            raise err
        if type(taui) not in [int,float]:
            err = TypeError("ERROR: taui is not int nor float.")
            logging.exception(err)
            raise err
        if type(taux) not in [int,float]:
            err = TypeError("ERROR: taux is not int nor float.")
            logging.exception(err)
            raise err
        if type(tauSTDP) not in [int,float]:
            err = TypeError("ERROR: tauSTDP is not int nor float.")
            logging.exception(err)
            raise err
        if type(eta_ee_hebb) not in [int,float]:
            err = TypeError("ERROR: eta_ee_hebb is not int,float.")
            logging.exception(err)
            raise err
        if type(Jmax_ee) not in [int,float]:
            err = TypeError("ERROR: Jmax_ee is not int,float.")
            logging.exception(err)
            raise err
        if type(eta_ee_koh) not in [int,float]:
            err = TypeError("ERROR: eta_ee_koh is not int,float.")
            logging.exception(err)
            raise err
        if type(beta) not in [int,float]:
            err = TypeError("ERROR: beta is not int,float.")
            logging.exception(err)
            raise err
        if type(eta_ie_homeo) not in [int,float]:
            err = TypeError("ERROR: eta_ie_homeo is not int,float.")
            logging.exception(err)
            raise err
        if type(alpha_ie) not in [int,float]:
            err = TypeError("ERROR: alpha_ie is not int,float.")
            logging.exception(err)
            raise err
        if type(eta_ie_hebb) not in [int,float]:
            err = TypeError("ERROR: eta_ie_hebb is not int,float.")
            logging.exception(err)
            raise err
        if type(Jmax_ie_hebb) not in [int,float]:
            err = TypeError("ERROR: Jmax_ie_hebb is not int,float.")
            logging.exception(err)
            raise err
        if type(eta_ei) not in [int,float]:
            err = TypeError("ERROR: eta_ei is not int,float.")
            logging.exception(err)
            raise err
        if type(alpha_ei) not in [int,float]:
            err = TypeError("ERROR: alpha_ei is not int,float.")
            logging.exception(err)
            raise err
        if type(eta_ii) not in [int,float]:
            err = TypeError("ERROR: eta_ii is not int,float.")
            logging.exception(err)
            raise err
        if type(alpha_ii) not in [int,float]:
            err = TypeError("ERROR: alpha_ii is not int,float.")
            logging.exception(err)
            raise err
        if type(dt) not in [int,float]:
            err = TypeError("ERROR: dt is not int,float.")
            logging.exception(err)
            raise err
        if type(nBinsRecord) not in [int]:
            err = TypeError("ERROR: nBinsRecord is not int.")
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
        if Jmax_ee < 0:
            err = ValueError("ERROR: Jmax_ee has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if eta_ee_koh < 0:
            err = ValueError("ERROR: eta_ee_koh has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if beta < 0:
            err = ValueError("ERROR: beta has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if eta_ie_homeo < 0:
            err = ValueError("ERROR: eta_ie_homeo has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if alpha_ie < 0:
            err = ValueError("ERROR: alpha_ie has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if eta_ie_hebb < 0:
            err = ValueError("ERROR: eta_ie_hebb has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if Jmax_ie_hebb < 0:
            err = ValueError("ERROR: Jmax_ie_hebb has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if eta_ei > 0:
            err = ValueError("ERROR: eta_ei has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if alpha_ei <= 0:
            err = ValueError("ERROR: alpha_ei has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if eta_ii > 0:
            err = ValueError("ERROR: eta_ii has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if alpha_ii <= 0:
            err = ValueError("ERROR: alpha_ii has to be greater than or equal to zero.")
            logging.exception(err)
            raise err
        if  dt <= 0:
            err = ValueError("ERROR: dt has to be greater than zero.")
            logging.exception(err)
            raise err
        if nBinsRecord <= 0:
            err = ValueError("ERROR: nBinsRecord has to be greater than zero.")
            logging.exception(err)
            raise err

        # Initialize some variables
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
        Ixrecord = np.sort(random2.sample(list(np.arange(0, self.Ne)), numrecord)).astype(
            int
        )
        Vrecord = (
            np.sort(
                [
                    [
                        random2.sample(
                            list(np.arange(0, self.Ne)), int(round(numrecord / 2))
                        ),
                        random2.sample(
                            list(np.arange(self.Ne, self.N)), int(round(numrecord / 2))
                        ),
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
        s = np.zeros((2, self.maxns))
        nspike = 0
        TooManySpikes = 0

        # Start the simulation.
        start_time = time.time()

        for i in range(len(self.total_time)):
            # Propagate ffwd spikes
            while (self.sx[0, iFspike] <= self.total_time[i]) & (
                iFspike < self.nspikeX - 1
            ):
                jpre = int(self.sx[1, iFspike])
                Ix += self.Jx[:, jpre] / taux
                iFspike += 1

            # Euler update to V
            V += (
                dt
                / Cm
                * (
                    self.Istim[i] * self.Jstim
                    + Ie
                    + Ii
                    + Ix
                    + gL * (EL - V)
                    + gL * DeltaT * np.exp((V - VT) / DeltaT)
                )
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
                    TooManySpikes = 1
                    break

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
                    ).transpose() * (
                        self.J[0 : self.Ne, Ispike[Ispike <= self.Ne]] != 0
                    )
                    # E to E after a post spike
                    self.J[Ispike[Ispike < self.Ne], 0 : self.Ne] -= (
                        eta_ee_koh * self.J[Ispike[Ispike < self.Ne], 0 : self.Ne]
                    )

                # If there is IE *Homeo* plasticity
                if eta_ie_homeo != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to I after a pre spike
                    self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]] -= np.tile(
                        eta_ie_homeo * (x[0, self.Ne : self.N] - alpha_ie),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (
                        self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]]
                    )
                    # E to I after a post spike
                    self.J[Ispike[Ispike > self.Ne], 0 : self.Ne] -= np.tile(
                        eta_ie_homeo * x[0, 0 : self.Ne].transpose(),
                        (np.count_nonzero(Ispike > self.Ne), 1),
                    ) * (self.J[Ispike[Ispike > self.Ne], 0 : self.Ne])

                # If there is IE *Hebbian* plasticity
                if eta_ie_hebb != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to I after a pre spike
                    self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]] -= np.tile(
                        eta_ie_hebb * (x[0, self.Ne : self.N]),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (
                        self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]] != 0
                    )
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
                        eta_ei * (x[0, 0 : self.Ne] - alpha_ei),
                        (np.count_nonzero(Ispike >= self.Ne), 1),
                    ).transpose() * (self.J[0 : self.Ne, Ispike[Ispike >= self.Ne]])
                    # I to E after an E spike
                    self.J[Ispike[Ispike < self.Ne], self.Ne : self.N] -= np.tile(
                        eta_ei * x[0, self.Ne : self.N].transpose(),
                        (np.count_nonzero(Ispike < self.Ne), 1),
                    ) * (self.J[Ispike[Ispike < self.Ne], self.Ne : self.N])

                # If there is II plasticity
                if eta_ii != 0:
                    # Update synaptic weights according to plasticity rules
                    # I to E after an I spike
                    self.J[self.Ne : self.N, Ispike[Ispike >= self.Ne]] -= np.tile(
                        eta_ii * (x[0, self.Ne : self.N] - alpha_ii),
                        (np.count_nonzero(Ispike >= self.Ne), 1),
                    ).transpose() * (
                        self.J[self.Ne : self.N, Ispike[Ispike >= self.Ne]]
                    )
                    # I to E after an E spike
                    self.J[Ispike[Ispike > self.Ne], self.Ne : self.N] -= np.tile(
                        eta_ii * x[0, self.Ne : self.N].transpose(),
                        (np.count_nonzero(Ispike > self.Ne), 1),
                    ) * (self.J[Ispike[Ispike > self.Ne], self.Ne : self.N])

                # Update rate estimates for plasticity rules
                x[0, Ispike] += 1

                # Update cumulative number of spikes
                nspike += len(Ispike)

            # Euler update to synaptic currents
            Ie -= dt * Ie / taue
            Ii -= dt * Ii / taui
            Ix -= dt * Ix / taux

            # Update time-dependent firing rates for plasticity
            x[0 : self.Ne] -= dt * x[0 : self.Ne] / tauSTDP
            x[self.Ne : self.N] -= dt * x[self.Ne : self.N] / tauSTDP

            # This makes plots of V(t) look better.
            # All action potentials reach Vth exactly.
            # This has no real effect on the network sims
            V[0, Ispike] = Vth

            # Store recorded variables
            ii = int(math.floor(i / nBinsRecord))
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

        elapsed_time = time.time() - start_time
        logging.info(f"Time for simulation: {round(elapsed_time/60,2)} minutes.")

        IeRec = IeRec / nBinsRecord  # Normalize recorded variables by # bins
        IiRec = IiRec / nBinsRecord
        IxRec = IxRec / nBinsRecord
        VRec = VRec / nBinsRecord
        JRec_ee = JRec_ee * np.sqrt(self.N) / nBinsRecord
        JRec_ie = JRec_ie * np.sqrt(self.N) / nBinsRecord
        JRec_ei = JRec_ei * np.sqrt(self.N) / nBinsRecord
        JRec_ii = JRec_ii * np.sqrt(self.N) / nBinsRecord

        s = s[:, 0:nspike]  # Get rid of padding in s

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
