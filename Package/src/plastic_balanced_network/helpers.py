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
        dtRecord,
    ):

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
        self.timeRecord = np.arange(dtRecord, T + dtRecord, dtRecord)
        self.Ntrec = len(self.timeRecord)

    def connectivity(self, Jm, Jxm, P, Px, nJrecord0):
        """
        Create connectivity matrix and arrays to record individual weights of all four connections.
        """
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
        """
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
        Ierecord,
        Iirecord,
        Ixrecord,
        Vrecord,
    ):
        """
        Execute Network simulation.
        """
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
