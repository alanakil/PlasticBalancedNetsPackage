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
    ):

        self.N = N
        self.Ne = int(round(frac_exc * N))
        self.Ni = int(round((1 - frac_exc) * N))
        self.Nx = int(round(frac_ext * N))
        self.rx = rx
        self.P = P
        self.Px = Px
        self.taujitter = taujitter
        self.Jm = Jm
        self.Jxm = Jxm
        self.T = T
        self.dt = dt
        self.Nt = round(T / dt)
        self.total_time = np.arange(dt, T + dt, dt)  # Time discretized domain.
        self.Istim = np.zeros(len(self.total_time))
        self.Istim[self.total_time > T / 2] = 0
        self.Jstim = np.sqrt(N) * np.hstack(
            (jestim * np.ones((1, self.Ne)), jistim * np.ones((1, self.Ni)))
        )
        self.taue = taue
        self.taui = taui
        self.taux = taux
        self.c = c
        self.maxns = round(0.05 * N * T)  # Max num of spikes (50 Hz).
        self.Cm = Cm
        self.gL = gL
        self.EL = EL
        self.Vth = Vth
        self.Vre = Vre
        self.DeltaT = DeltaT
        self.VT = VT
        self.tauSTDP = tauSTDP
        self.Jmax_ee = Jmax_ee
        self.Jmax_ie_hebb = Jmax_ie_hebb
        self.eta_ee_hebb = eta_ee_hebb
        self.eta_ee_koh = eta_ee_koh
        self.eta_ei = eta_ei
        self.eta_ie_hebb = eta_ie_hebb
        self.eta_ie_homeo = eta_ie_homeo
        self.eta_ii = eta_ii
        self.alpha_ei = alpha_ei
        self.alpha_ie = alpha_ie
        self.alpha_ii = alpha_ii
        self.beta = beta
        self.nBinsRecord = nBinsRecord
        self.dtRecord = dtRecord
        self.timeRecord = np.arange(dtRecord, T + dtRecord, dtRecord)
        self.Ntrec = len(self.timeRecord)
        self.Ierecord = Ierecord
        self.Iirecord = Iirecord
        self.Ixrecord = Ixrecord
        self.Vrecord = Vrecord
        self.numrecord = numrecord

    def connectivity(self):
        """
        Create connectivity matrix and arrays to record individual weights of all four connections.
        """
        # Define connectivity
        self.J = np.vstack(
            (
                np.hstack(
                    (
                        np.array(
                            self.Jm[0, 0]
                            * np.random.binomial(1, self.P[0, 0], (self.Ne, self.Ne))
                        ),
                        np.array(
                            self.Jm[0, 1]
                            * np.random.binomial(1, self.P[0, 1], (self.Ne, self.Ni))
                        ),
                    )
                ),
                np.hstack(
                    (
                        np.array(
                            self.Jm[1, 0]
                            * np.random.binomial(1, self.P[1, 0], (self.Ni, self.Ne))
                        ),
                        np.array(
                            self.Jm[1, 1]
                            * np.random.binomial(1, self.P[1, 1], (self.Ni, self.Ni))
                        ),
                    )
                ),
            )
        )

        self.Jx = np.vstack(
            (
                np.array(
                    self.Jxm[0, 0]
                    * np.random.binomial(1, self.Px[0, 0], (self.Ne, self.Nx))
                ),
                np.array(
                    self.Jxm[1, 0]
                    * np.random.binomial(1, self.P[1, 0], (self.Ni, self.Nx))
                ),
            )
        )

        # Define variables to record changes in weights.
        # Synaptic weights EE to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        nJrecord0 = 1000  # Number to record
        IIJJ_rec = np.argwhere(
            self.J[0 : self.Ne, 0 : self.Ne]
        )  # Find non-zero I to E weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        III = random2.sample(list(II), nJrecord0)  # Choose some at random to record
        II = II[III]
        JJ = JJ[III]
        self.Jrecord_ee = np.array([II, JJ])  # Record these
        self.numrecordJ_ee = len(JJ)

        # Synaptic weights IE to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        nJrecord0 = 1000  # Number to record
        IIJJ_rec = np.argwhere(
            self.J[self.Ne + 1 : self.N, 0 : self.Ne]
        )  # Find non-zero I to E weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        III = random2.sample(list(II), nJrecord0)  # Choose some at random to record
        II = II[III]
        JJ = JJ[III]
        self.Jrecord_ie = np.array([II + self.Ne, JJ])  # Record these
        self.numrecordJ_ie = len(JJ)

        # Synaptic weights EI to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        nJrecord0 = 1000  # Number to record
        IIJJ_rec = np.argwhere(
            self.J[0 : self.Ne, self.Ne + 1 : self.N]
        )  # Find non-zero I to E weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        III = random2.sample(list(II), nJrecord0)  # Choose some at random to record
        II = II[III]
        JJ = JJ[III]
        self.Jrecord_ei = np.array([II, JJ + self.Ne])  # Record these
        self.numrecordJ_ei = len(JJ)

        # Synaptic weights II to record.
        # The first row of Jrecord is the postsynaptic indices
        # The second row is the presynaptic indices
        nJrecord0 = 1000  # Number to record
        IIJJ_rec = np.argwhere(
            self.J[self.Ne + 1 : self.N, self.Ne + 1 : self.N]
        )  # Find non-zero I to E weights
        II = IIJJ_rec[:, 0]
        JJ = IIJJ_rec[:, 1]
        III = random2.sample(list(II), nJrecord0)  # Choose some at random to record
        II = II[III]
        JJ = JJ[III]
        self.Jrecord_ii = np.array([II + self.Ne, JJ + self.Ne])  # Record these
        self.numrecordJ_ii = len(JJ)
        return None

    def ffwd_spikes(self):
        """
        Create all spike trains of the Poisson feedforward, external layer.
        """
        if self.c < 1e-5:  # If uncorrelated
            nspikeX = np.random.poisson(self.Nx * self.rx * self.T)
            st = np.random.uniform(0, 1, (1, nspikeX)) * self.T
            sx = np.zeros((2, len(st[0])))
            sx[0, :] = np.sort(st)[0]  # spike time
            sx[1, :] = np.random.randint(
                1, self.Nx, (1, len(st[0]))
            )  # neuron index that spiked
        else:  # If correlated
            rm = self.rx / self.c  # Firing rate of mother process
            nstm = np.random.poisson(rm * self.T)  # Number of mother spikes
            stm = (
                np.random.uniform(0, 1, (nstm, 1)) * self.T
            )  # spike times of mother process
            maxnsx = int(self.T * self.rx * self.Nx * 1.2)  # Max num spikes
            sx = np.zeros((2, maxnsx))
            ns = 0
            for j in np.arange(1, self.Nx, 1):  # For each ffwd spike train
                ns0 = np.random.binomial(
                    nstm, self.c
                )  # Number of spikes for this spike train
                st = random2.sample(list(stm[:, 0]), ns0)  # Sample spike times randomly
                st = st + self.taujitter * np.random.normal(
                    0, 1, size=len(st)
                )  # jitter spike times
                st = st[(st > 0) & (st < self.T)]  # Get rid of out-of-bounds times
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

    def simulate(self):
        """
        Execute Network simulation.
        """
        # Initialize some variables
        # Random initial voltages
        V0 = np.random.uniform(0, 1, (1, self.N)) * (self.VT - self.Vre) + self.Vre
        V = V0
        # Initialize current vectors.
        Ie = np.zeros((1, self.N))
        Ii = np.zeros((1, self.N))
        Ix = np.zeros((1, self.N))
        # Initialize eligibility traces.
        x = np.zeros((1, self.N))
        # Initialize recorded currents vectors.
        IeRec = np.zeros((self.numrecord, self.Ntrec))
        IiRec = np.zeros((self.numrecord, self.Ntrec))
        IxRec = np.zeros((self.numrecord, self.Ntrec))
        VRec = np.zeros((self.numrecord, self.Ntrec))
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
                Ix += self.Jx[:, jpre] / self.taux
                iFspike += 1

            # Euler update to V
            V += (
                self.dt
                / self.Cm
                * (
                    self.Istim[i] * self.Jstim
                    + Ie
                    + Ii
                    + Ix
                    + self.gL * (self.EL - V)
                    + self.gL * self.DeltaT * np.exp((V - self.VT) / self.DeltaT)
                )
            )

            # Find which neurons spiked
            Ispike = np.argwhere(V >= self.Vth)[:, 1]

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
                Ie += np.sum(self.J[:, Ispike[Ispike <= self.Ne]], 1) / self.taue
                Ii += np.sum(self.J[:, Ispike[Ispike > self.Ne]], 1) / self.taui

                # If there is EE Hebbian plasticity
                if self.eta_ee_hebb != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to E after a pre spike
                    self.J[0 : self.Ne, Ispike[Ispike <= self.Ne]] -= np.tile(
                        self.eta_ee_hebb * (x[0, 0 : self.Ne]),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (self.J[0 : self.Ne, Ispike[Ispike <= self.Ne]])
                    # E to E after a post spike
                    self.J[Ispike[Ispike < self.Ne], 0 : self.Ne] += (
                        np.tile(
                            self.eta_ee_hebb * x[0, 0 : self.Ne].transpose(),
                            (np.count_nonzero(Ispike < self.Ne), 1),
                        )
                        * self.Jmax_ee
                        * (self.J[Ispike[Ispike < self.Ne], 0 : self.Ne] != 0)
                    )

                # If there is EE Kohonen plasticity
                if self.eta_ee_koh != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to E after a pre spike
                    self.J[0 : self.Ne, Ispike[Ispike <= self.Ne]] += np.tile(
                        self.beta * self.eta_ee_koh * (x[0, 0 : self.Ne]),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (
                        self.J[0 : self.Ne, Ispike[Ispike <= self.Ne]] != 0
                    )
                    # E to E after a post spike
                    self.J[Ispike[Ispike < self.Ne], 0 : self.Ne] -= (
                        self.eta_ee_koh * self.J[Ispike[Ispike < self.Ne], 0 : self.Ne]
                    )

                # If there is IE *Homeo* plasticity
                if self.eta_ie_homeo != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to I after a pre spike
                    self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]] -= np.tile(
                        self.eta_ie_homeo * (x[0, self.Ne : self.N] - self.alpha_ie),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (
                        self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]]
                    )
                    # E to I after a post spike
                    self.J[Ispike[Ispike > self.Ne], 0 : self.Ne] -= np.tile(
                        self.eta_ie_homeo * x[0, 0 : self.Ne].transpose(),
                        (np.count_nonzero(Ispike > self.Ne), 1),
                    ) * (self.J[Ispike[Ispike > self.Ne], 0 : self.Ne])

                # If there is IE *Hebbian* plasticity
                if self.eta_ie_hebb != 0:
                    # Update synaptic weights according to plasticity rules
                    # E to I after a pre spike
                    self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]] -= np.tile(
                        self.eta_ie_hebb * (x[0, self.Ne : self.N]),
                        (np.count_nonzero(Ispike <= self.Ne), 1),
                    ).transpose() * (
                        self.J[self.Ne : self.N, Ispike[Ispike <= self.Ne]] != 0
                    )
                    # E to I after a post spike
                    self.J[Ispike[Ispike > self.Ne], 0 : self.Ne] += (
                        np.tile(
                            self.eta_ie_hebb * x[0, 0 : self.Ne].transpose(),
                            (np.count_nonzero(Ispike > self.Ne), 1),
                        )
                        * self.Jmax_ie_hebb
                        * (self.J[Ispike[Ispike > self.Ne], 0 : self.Ne] != 0)
                    )

                # If there is EI plasticity
                if self.eta_ei != 0:
                    # Update synaptic weights according to plasticity rules
                    # I to E after an I spike
                    self.J[0 : self.Ne, Ispike[Ispike >= self.Ne]] -= np.tile(
                        self.eta_ei * (x[0, 0 : self.Ne] - self.alpha_ei),
                        (np.count_nonzero(Ispike >= self.Ne), 1),
                    ).transpose() * (self.J[0 : self.Ne, Ispike[Ispike >= self.Ne]])
                    # I to E after an E spike
                    self.J[Ispike[Ispike < self.Ne], self.Ne : self.N] -= np.tile(
                        self.eta_ei * x[0, self.Ne : self.N].transpose(),
                        (np.count_nonzero(Ispike < self.Ne), 1),
                    ) * (self.J[Ispike[Ispike < self.Ne], self.Ne : self.N])

                # If there is II plasticity
                if self.eta_ii != 0:
                    # Update synaptic weights according to plasticity rules
                    # I to E after an I spike
                    self.J[self.Ne : self.N, Ispike[Ispike >= self.Ne]] -= np.tile(
                        self.eta_ii * (x[0, self.Ne : self.N] - self.alpha_ii),
                        (np.count_nonzero(Ispike >= self.Ne), 1),
                    ).transpose() * (
                        self.J[self.Ne : self.N, Ispike[Ispike >= self.Ne]]
                    )
                    # I to E after an E spike
                    self.J[Ispike[Ispike > self.Ne], self.Ne : self.N] -= np.tile(
                        self.eta_ii * x[0, self.Ne : self.N].transpose(),
                        (np.count_nonzero(Ispike > self.Ne), 1),
                    ) * (self.J[Ispike[Ispike > self.Ne], self.Ne : self.N])

                # Update rate estimates for plasticity rules
                x[0, Ispike] += 1

                # Update cumulative number of spikes
                nspike += len(Ispike)

            # Euler update to synaptic currents
            Ie -= self.dt * Ie / self.taue
            Ii -= self.dt * Ii / self.taui
            Ix -= self.dt * Ix / self.taux

            # Update time-dependent firing rates for plasticity
            x[0 : self.Ne] -= self.dt * x[0 : self.Ne] / self.tauSTDP
            x[self.Ne : self.N] -= self.dt * x[self.Ne : self.N] / self.tauSTDP

            # This makes plots of V(t) look better.
            # All action potentials reach Vth exactly.
            # This has no real effect on the network sims
            V[0, Ispike] = self.Vth

            # Store recorded variables
            ii = int(math.floor(i / self.nBinsRecord))
            IeRec[:, ii] += Ie[0, self.Ierecord]
            IiRec[:, ii] += Ii[0, self.Iirecord]
            IxRec[:, ii] += Ix[0, self.Ixrecord]
            VRec[:, ii] += V[0, self.Vrecord]
            JRec_ee[:, ii] += self.J[self.Jrecord_ee[0, :], self.Jrecord_ee[1, :]]
            JRec_ie[:, ii] += self.J[self.Jrecord_ie[0, :], self.Jrecord_ie[1, :]]
            JRec_ei[:, ii] += self.J[self.Jrecord_ei[0, :], self.Jrecord_ei[1, :]]
            JRec_ii[:, ii] += self.J[self.Jrecord_ii[0, :], self.Jrecord_ii[1, :]]

            # Reset mem pot.
            V[0, Ispike] = self.Vre

        elapsed_time = time.time() - start_time
        logging.info(f"Time for simulation: {round(elapsed_time/60,2)} minutes.")

        IeRec = IeRec / self.nBinsRecord  # Normalize recorded variables by # bins
        IiRec = IiRec / self.nBinsRecord
        IxRec = IxRec / self.nBinsRecord
        VRec = VRec / self.nBinsRecord
        JRec_ee = JRec_ee * np.sqrt(self.N)
        JRec_ie = JRec_ie * np.sqrt(self.N)
        JRec_ei = JRec_ei * np.sqrt(self.N)
        JRec_ii = JRec_ii * np.sqrt(self.N)

        s = s[:, 0:nspike]  # Get rid of padding in s

        return s, JRec_ee, JRec_ie, JRec_ei, JRec_ii, IeRec, IiRec, IxRec, VRec, self.timeRecord
