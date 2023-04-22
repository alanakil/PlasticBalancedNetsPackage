### Test functions in plastic_balanced_network/

# %%
import pytest
import numpy as np
import unittest
from numpy.testing import assert_array_equal
from scipy.stats import expon, kstest

from src.plastic_balanced_network.network import PlasticNeuralNetwork

# %%


class Test__init__(unittest.TestCase):  # noqa: N801
    def test_N_type_str(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.__init__(
                self,
                N="1",
                frac_exc=0.2,
                frac_ext=0.2,
                T=100,
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_frac_exc_type_str(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc="0.2",
                frac_ext=0.2,
                T=100,
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_frac_ext_type_str(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext="0.2",
                T=100,
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_T_type_str(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext=0.2,
                T="100",
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_dt_type_str(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext=0.2,
                T=100,
                dt="0.1",
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_jestim_type_str(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext=0.2,
                T=100,
                dt=0.1,
                jestim="1",
                jistim=1,
                nBinsRecord=100,
            )

    def test_jistim_type_str(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext=0.2,
                T=100,
                dt=0.1,
                jestim=1,
                jistim="1",
                nBinsRecord=100,
            )

    def test_nBinsRecord_type_str(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext=0.2,
                T=100,
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord="100",
            )

    def test_N_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.__init__(
                self,
                N=-1,
                frac_exc=0.2,
                frac_ext=0.2,
                T=100,
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_frac_exc_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=2,
                frac_ext=0.2,
                T=100,
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_frac_ext_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext=-0.2,
                T=100,
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_T_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext=0.2,
                T=-100,
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_dt_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext=0.2,
                T=100,
                dt=-0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=100,
            )

    def test_nBinsRecord_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.__init__(
                self,
                N=10,
                frac_exc=0.2,
                frac_ext=0.2,
                T=100,
                dt=0.1,
                jestim=1,
                jistim=1,
                nBinsRecord=-100,
            )

    def test_initialization(self):  # noqa: N802
        """
        Test initialization of PlasticNeuralNetwork object.
        """
        N = 1000
        frac_exc = 0.8
        frac_ext = 0.2
        T = 1000
        dt = 0.1
        jestim = 0.0
        jistim = 0.0
        nBinsRecord = 10

        pnn = PlasticNeuralNetwork(N, T, frac_exc, frac_ext, dt, jestim, jistim, nBinsRecord)

        self.assertEqual(pnn.N, N)
        self.assertEqual(pnn.Ne, int(round(frac_exc * N)))
        self.assertEqual(pnn.Ni, int(round((1 - frac_exc) * N)))
        self.assertEqual(pnn.Nx, int(round(frac_ext * N)))
        self.assertEqual(pnn.Nt, round(T / dt))
        assert_array_equal(pnn.total_time, np.arange(dt, T + dt, dt))
        assert_array_equal(pnn.Istim, np.zeros(len(np.arange(dt, T + dt, dt))))
        assert_array_equal(
            pnn.Jstim, np.sqrt(N) * np.hstack((jestim * np.ones((1, pnn.Ne)), jistim * np.ones((1, pnn.Ni))))
        )
        self.assertEqual(pnn.maxns, round(0.05 * N * T))
        assert_array_equal(pnn.timeRecord, np.arange(nBinsRecord * dt, T + nBinsRecord * dt, nBinsRecord * dt))
        self.assertEqual(pnn.Ntrec, len(np.arange(nBinsRecord * dt, T + nBinsRecord * dt, nBinsRecord * dt)))


# %%
class Test_connectivity(unittest.TestCase):  # noqa: N801
    def test_jee_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee="25",
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
            )

    def test_jie_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie="112.5",
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
            )

    def test_jei_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei="-150",
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
            )

    def test_jii_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii="-250",
                jex=180,
                jix=135,
                p_ee=0.1,
                p_ie=0.1,
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_jex_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex="180",
                jix=135,
                p_ee=0.1,
                p_ie=0.1,
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_jix_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex=180,
                jix="135",
                p_ee=0.1,
                p_ie=0.1,
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_p_ee_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex=180,
                jix=135,
                p_ee="0.1",
                p_ie=0.1,
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_p_ie_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex=180,
                jix=135,
                p_ee=0.1,
                p_ie="0.1",
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_p_ei_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex=180,
                jix=135,
                p_ee=0.1,
                p_ie=0.1,
                p_ei="0.1",
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_p_ii_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
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
                p_ii="0.1",
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_p_ex_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
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
                p_ex="0.1",
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_p_ix_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
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
                p_ix="0.1",
                nJrecord0=100,
            )

    def test_nJrecord0_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.connectivity(
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
                nJrecord0=[100],
            )

    # Value tests.
    def test_jee_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=-25,
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
            )

    def test_jie_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=-112.5,
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
            )

    def test_jei_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=150,
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
            )

    def test_jii_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=250,
                jex=180,
                jix=135,
                p_ee=0.1,
                p_ie=0.1,
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_jex_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex=-180,
                jix=135,
                p_ee=0.1,
                p_ie=0.1,
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_jix_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex=180,
                jix=-135,
                p_ee=0.1,
                p_ie=0.1,
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_Pee_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex=180,
                jix=135,
                p_ee=-0.1,
                p_ie=0.1,
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_Pie_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex=180,
                jix=135,
                p_ee=0.1,
                p_ie=-0.1,
                p_ei=0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_Pei_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
                self,
                jee=25,
                jie=112.5,
                jei=-150,
                jii=-250,
                jex=180,
                jix=135,
                p_ee=0.1,
                p_ie=0.1,
                p_ei=-0.1,
                p_ii=0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_Pii_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
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
                p_ii=-0.1,
                p_ex=0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_Pex_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
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
                p_ex=-0.1,
                p_ix=0.1,
                nJrecord0=100,
            )

    def test_Pix_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
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
                p_ix=-0.1,
                nJrecord0=100,
            )

    def test_nJrecord0_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.connectivity(
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
                nJrecord0=-100,
            )

    def test_connectivity_outcomes(self):  # noqa: N802
        """
        Test connectivity of PlasticNeuralNetwork object.
        """
        np.random.seed(314)
        N = 1000
        frac_exc = 0.8
        frac_ext = 0.2
        T = 10
        dt = 0.1
        jestim = 0.0
        jistim = 0.0
        nBinsRecord = 10
        jee = 25
        jei = -150
        jie = 112.5
        jii = -250
        jex = 180
        jix = 135
        Jm = np.array([[jee, jei], [jie, jii]]) / np.sqrt(N)
        Jxm = np.array([[jex], [jix]]) / np.sqrt(N)
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
        nJrecord0 = 1000

        pnn = PlasticNeuralNetwork(N, T, frac_exc, frac_ext, dt, jestim, jistim, nBinsRecord)
        pnn.connectivity(jee, jie, jei, jii, jex, jix, p_ee, p_ie, p_ei, p_ii, p_ex, p_ix, nJrecord0)

        np.random.seed(314)
        assert_array_equal(
            pnn.J,
            np.vstack(
                (
                    np.hstack(
                        (
                            np.array(Jm[0, 0] * np.random.binomial(1, P[0, 0], (pnn.Ne, pnn.Ne))),
                            np.array(Jm[0, 1] * np.random.binomial(1, P[0, 1], (pnn.Ne, pnn.Ni))),
                        )
                    ),
                    np.hstack(
                        (
                            np.array(Jm[1, 0] * np.random.binomial(1, P[1, 0], (pnn.Ni, pnn.Ne))),
                            np.array(Jm[1, 1] * np.random.binomial(1, P[1, 1], (pnn.Ni, pnn.Ni))),
                        )
                    ),
                )
            ),
        )
        assert_array_equal(
            pnn.Jx,
            np.vstack(
                (
                    np.array(Jxm[0, 0] * np.random.binomial(1, Px[0, 0], (pnn.Ne, pnn.Nx))),
                    np.array(Jxm[1, 0] * np.random.binomial(1, Px[1, 0], (pnn.Ni, pnn.Nx))),
                )
            ),
        )
        self.assertEqual(len(pnn.Jrecord_ee[0, :]), nJrecord0)
        self.assertEqual(len(pnn.Jrecord_ei[0, :]), nJrecord0)
        self.assertEqual(len(pnn.Jrecord_ie[0, :]), nJrecord0)
        self.assertEqual(len(pnn.Jrecord_ii[0, :]), nJrecord0)
        self.assertEqual(pnn.numrecordJ_ee, nJrecord0)
        self.assertEqual(pnn.numrecordJ_ei, nJrecord0)
        self.assertEqual(pnn.numrecordJ_ie, nJrecord0)
        self.assertEqual(pnn.numrecordJ_ii, nJrecord0)


# %%


class Test_ffwd_spikes:  # noqa: N801
    def test_cx_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.ffwd_spikes(self, T=10000, cx="0.1", rx=10 / 1000, taujitter=5)

    def test_rx_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.ffwd_spikes(self, T=10000, cx=0.1, rx=[10 / 1000], taujitter=5)

    def test_taujitter_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.ffwd_spikes(self, T=10000, cx=0.1, rx=10 / 1000, taujitter="5")

    def test_T_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.ffwd_spikes(self, T="10000", cx=0.1, rx=10 / 1000, taujitter=5)

    # Value tests
    def test_cx_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.ffwd_spikes(self, T=10000, cx=-0.1, rx=10 / 1000, taujitter=5)

    def test_rx_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.ffwd_spikes(
                self,
                T=10000,
                cx=0.1,
                rx=-10 / 1000,
                taujitter=5,
            )

    def test_taujitter_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.ffwd_spikes(
                self,
                T=10000,
                cx=0.1,
                rx=10 / 1000,
                taujitter=-5,
            )

    def test_T_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.ffwd_spikes(
                self,
                T=-10000,
                cx=0.1,
                rx=10 / 1000,
                taujitter=5,
            )

    def test_ffwd_spikes_outcomes(self):  # noqa: N802
        """
        Test ffwd spike generation of PlasticNeuralNetwork object.
        """
        N = 1000
        frac_exc = 0.8
        frac_ext = 0.2
        T = 10000
        dt = 0.1
        jestim = 0.0
        jistim = 0.0
        nBinsRecord = 10
        rx = 10 / 1000
        cx = 0
        taujitter = 5

        # Fix seed for reproducibility.
        np.random.seed(314)

        pnn = PlasticNeuralNetwork(N, T, frac_exc, frac_ext, dt, jestim, jistim, nBinsRecord)
        pnn.ffwd_spikes(T, cx, rx, taujitter)

        # External ffwd spike trains should be Poisson. So the interspike interval (i.e, ISI) should be Exp-distributed.

        # inter-arrival times of events. WLOG Pick the first neuron in the layer.
        inter_arrival_times = pnn.sx[0, pnn.sx[1, :] == 1]
        # fit the inter-arrival times to an exponential distribution
        fit_params = expon.fit(inter_arrival_times)
        # perform the KS test
        ks_stat, p_value = kstest(inter_arrival_times, "expon", fit_params)

        # assert p-value is above significance level (e.g. 0.05)
        assert p_value > 0.05


# %%


class Test_simulate(unittest.TestCase):  # noqa: N801
    def test_Cm_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm="1",
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_gL_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL="1/15",
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_VT_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT="-55",
                Vre=-75,
                Vth=-50,
                EL=-72,
                DeltaT=1,
                taue=8,
                taui=4,
                taux=10,
                tauSTDP=200,
                numrecord=100,
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_Vre_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre="-75",
                Vth=-50,
                EL=-72,
                DeltaT=1,
                taue=8,
                taui=4,
                taux=10,
                tauSTDP=200,
                numrecord=100,
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_Vth_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth="-50",
                EL=-72,
                DeltaT=1,
                taue=8,
                taui=4,
                taux=10,
                tauSTDP=200,
                numrecord=100,
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_EL_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth=-50,
                EL="-72",
                DeltaT=1,
                taue=8,
                taui=4,
                taux=10,
                tauSTDP=200,
                numrecord=100,
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_DeltaT_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth=-50,
                EL=-72,
                DeltaT="1",
                taue=8,
                taui=4,
                taux=10,
                tauSTDP=200,
                numrecord=100,
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_taue_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth=-50,
                EL=-72,
                DeltaT=1,
                taue="8",
                taui=4,
                taux=10,
                tauSTDP=200,
                numrecord=100,
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_taui_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth=-50,
                EL=-72,
                DeltaT=1,
                taue=8,
                taui="4",
                taux=10,
                tauSTDP=200,
                numrecord=100,
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_taux_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                taux="10",
                tauSTDP=200,
                numrecord=100,
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_tauSTDP_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                tauSTDP="200",
                numrecord=100,
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_numrecord_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                numrecord="100",
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_eta_ee_hebb_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb="1",
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_Jmax_ee_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee="10",
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_eta_ee_koh_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh="1",
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_beta_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta="1",
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_eta_ie_homeo_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo="1",
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_rho_ie_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie="1",
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_eta_ie_hebb_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb="1",
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_jmax_ie_hebb_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb="1",
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_eta_ei_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei="-1",
                rho_ei=1,
                eta_ii=-1,
                rho_ii=1,
            )

    def test_rho_ei_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei="1",
                eta_ii=-1,
                rho_ii=1,
            )

    def test_eta_ii_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii="-1",
                rho_ii=1,
            )

    def test_rho_ii_type(self):  # noqa: N802
        with pytest.raises(TypeError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                rho_ie=1,
                eta_ie_hebb=1,
                jmax_ie_hebb=1,
                eta_ei=-1,
                rho_ei=1,
                eta_ii=-1,
                rho_ii="1",
            )

    def test_Cm_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=-1,
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
            )

    def test_gL_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=-1 / 15,
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
            )

    def test_VT_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=55,
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
            )

    def test_Vre_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=75,
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
            )

    def test_Vth_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth=50,
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
            )

    def test_EL_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth=-50,
                EL=72,
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
            )

    def test_DeltaT_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth=-50,
                EL=-72,
                DeltaT=-1,
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
            )

    def test_taue_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth=-50,
                EL=-72,
                DeltaT=1,
                taue=-8,
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
            )

    def test_taui_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1 / 15,
                VT=-55,
                Vre=-75,
                Vth=-50,
                EL=-72,
                DeltaT=1,
                taue=8,
                taui=-4,
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
            )

    def test_taux_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                taux=-10,
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
            )

    def test_tauSTDP_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                tauSTDP=0,
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
            )

    def test_numrecord_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                numrecord=-100,
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
            )

    def test_eta_ee_hebb_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=-10,
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
            )

    def test_jmax_ee_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                jmax_ee=-30,
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
            )

    def test_eta_ee_koh_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_koh=-10,
                beta=2,
                eta_ie_homeo=0,
                rho_ie=0.020,
                eta_ie_hebb=0,
                jmax_ie_hebb=125,
                eta_ei=0,
                rho_ei=0.010,
                eta_ii=0,
                rho_ii=0.020,
            )

    def test_beta_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                beta=0,
                eta_ie_homeo=0,
                rho_ie=0.020,
                eta_ie_hebb=0,
                jmax_ie_hebb=125,
                eta_ei=0,
                rho_ei=0.010,
                eta_ii=0,
                rho_ii=0.020,
            )

    def test_eta_ie_homeo_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                eta_ie_homeo=-10,
                rho_ie=0.020,
                eta_ie_hebb=0,
                jmax_ie_hebb=125,
                eta_ei=0,
                rho_ei=0.010,
                eta_ii=0,
                rho_ii=0.020,
            )

    def test_rho_ie_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                rho_ie=-0.020,
                eta_ie_hebb=0,
                jmax_ie_hebb=125,
                eta_ei=0,
                rho_ei=0.010,
                eta_ii=0,
                rho_ii=0.020,
            )

    def test_eta_ie_hebb_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                eta_ie_hebb=-10,
                jmax_ie_hebb=125,
                eta_ei=0,
                rho_ei=0.010,
                eta_ii=0,
                rho_ii=0.020,
            )

    def test_jmax_ie_hebb_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                jmax_ie_hebb=-125,
                eta_ei=0,
                rho_ei=0.010,
                eta_ii=0,
                rho_ii=0.020,
            )

    def test_eta_ei_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                eta_ei=-10,
                rho_ei=0.010,
                eta_ii=0,
                rho_ii=0.020,
            )

    def test_rho_ei_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                rho_ei=-0.010,
                eta_ii=0,
                rho_ii=0.020,
            )

    def test_eta_ii_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                eta_ii=-10,
                rho_ii=0.020,
            )

    def test_rho_ii_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                rho_ii=-0.020,
            )

    def test_double_stdp_ee_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                eta_ee_hebb=1,
                jmax_ee=30,
                eta_ee_koh=1,
                beta=2,
                eta_ie_homeo=0,
                rho_ie=0.020,
                eta_ie_hebb=0,
                jmax_ie_hebb=125,
                eta_ei=0,
                rho_ei=0.010,
                eta_ii=0,
                rho_ii=-0.020,
            )

    def test_double_stdp_ie_value(self):  # noqa: N802
        with pytest.raises(ValueError):
            PlasticNeuralNetwork.simulate(
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
                eta_ie_homeo=1,
                rho_ie=0.020,
                eta_ie_hebb=1,
                jmax_ie_hebb=125,
                eta_ei=0,
                rho_ei=0.010,
                eta_ii=0,
                rho_ii=-0.020,
            )

    def test_simulate_outcomes(self):  # noqa: N802
        """
        Test network simulation of PlasticNeuralNetwork object.
        """
        N = int(5000)
        # Total_time (in ms) for sim
        T = 5000

        # EI homeo
        eta_ei = 0.015 / 10**3  # Learning rate
        # II
        eta_ii = 0.015 / 10**3  # Learning rate

        # Indices of neurons to record currents, voltages
        numrecord = int(100)  # Number to record from each population
        # Number of synapses to be sampled
        nJrecord0 = 200

        # Set the random seed
        np.random.seed(31415)

        # Define the model.
        pnn = PlasticNeuralNetwork(N=N, T=T)
        # Initialize the connectivity
        pnn.connectivity(nJrecord0=nJrecord0)
        # Generate Poisson ffwd spike trains
        pnn.ffwd_spikes(T=T)
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
        ) = pnn.simulate(
            numrecord=numrecord,
            eta_ei=eta_ei,
            eta_ii=eta_ii,
        )

        self.assertEqual(len(JRec_ee[:, 0]), nJrecord0)
        self.assertEqual(len(JRec_ei[:, 0]), nJrecord0)
        self.assertEqual(len(JRec_ie[:, 0]), nJrecord0)
        self.assertEqual(len(JRec_ii[:, 0]), nJrecord0)

        self.assertEqual(len(IeRec[:, 0]), numrecord)
        self.assertEqual(len(IeRec[0, :]), pnn.Ntrec)

        self.assertEqual(len(IiRec[:, 0]), numrecord)
        self.assertEqual(len(IiRec[0, :]), pnn.Ntrec)

        self.assertEqual(len(IxRec[:, 0]), numrecord)
        self.assertEqual(len(IxRec[0, :]), pnn.Ntrec)

        self.assertEqual(len(JRec_ee[0, :]), pnn.Ntrec)
        self.assertEqual(len(JRec_ei[0, :]), pnn.Ntrec)
        self.assertEqual(len(JRec_ie[0, :]), pnn.Ntrec)
        self.assertEqual(len(JRec_ii[0, :]), pnn.Ntrec)

        print(IeRec[IeRec < 0])

        assert np.all(s >= 0), "Spike times and neuron indices should be non-negative"
        assert np.all(JRec_ee >= 0), "EE weights should be positive"
        assert np.all(JRec_ei <= 0), "EI weights should be negative"
        assert np.all(JRec_ie >= 0), "IE weights should be positive"
        assert np.all(JRec_ii <= 0), "II weights should be negative"
        assert np.all(VRec < 0), "Memberane potential should always be negative"


# %%
