### Test functions in plastic_balanced_network/

#%%
import pytest
import numpy as np
import unittest
from numpy.testing import assert_array_equal
from scipy.stats import expon, kstest

from src.plastic_balanced_network.helpers import plasticNeuralNetwork

#%%

class Test__init__(unittest.TestCase):
    def test_N_type_str(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.__init__(
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
    def test_frac_exc_type_str(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.__init__(
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
    def test_frac_ext_type_str(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.__init__(
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
    def test_T_type_str(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.__init__(
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
    def test_dt_type_str(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.__init__(
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
    def test_jestim_type_str(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.__init__(
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
    def test_jistim_type_str(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.__init__(
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
    def test_nBinsRecord_type_str(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.__init__(
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
    def test_N_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.__init__(
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
    def test_frac_exc_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.__init__(
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
    def test_frac_ext_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.__init__(
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
    def test_T_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.__init__(
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
    def test_dt_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.__init__(
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
    def test_nBinsRecord_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.__init__(
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

    def test_initialization(self):
        """
        Test initialization of plasticNeuralNetwork object.
        """
        N = 1000
        frac_exc = 0.8
        frac_ext = 0.2
        T = 1000
        dt = 0.1
        jestim = 0.0
        jistim = 0.0
        nBinsRecord = 10
        
        pnn = plasticNeuralNetwork(N, frac_exc, frac_ext, T, dt, jestim, jistim, nBinsRecord)
        
        self.assertEqual(pnn.N, N)
        self.assertEqual(pnn.Ne, int(round(frac_exc * N)))
        self.assertEqual(pnn.Ni, int(round((1 - frac_exc) * N)))
        self.assertEqual(pnn.Nx, int(round(frac_ext * N)))
        self.assertEqual(pnn.Nt, round(T / dt))
        assert_array_equal(pnn.total_time, np.arange(dt, T + dt, dt))
        assert_array_equal(pnn.Istim, np.zeros(len(np.arange(dt, T + dt, dt))))
        assert_array_equal(pnn.Jstim, np.sqrt(N) * np.hstack(
            (jestim * np.ones((1, pnn.Ne)), jistim * np.ones((1, pnn.Ni)))
        ))
        self.assertEqual(pnn.maxns, round(0.05 * N * T))
        assert_array_equal(pnn.timeRecord, np.arange(nBinsRecord * dt, T + nBinsRecord * dt, nBinsRecord * dt))
        self.assertEqual(pnn.Ntrec, len(np.arange(nBinsRecord * dt, T + nBinsRecord * dt, nBinsRecord * dt)))
    
    
#%%
class Test_connectivity(unittest.TestCase):
    def test_Jm_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=[[1,-1],[2,-2]],
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1],[.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Jxm_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=[1,2], 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_P_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=[[.1,.1],[.1,.1]], 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Px_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=[[.1],[.1]], 
                nJrecord0=100
            )   
    def test_nJrecord0_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=[100]
            )   
    # Value tests.
    def test_Jmee_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[-1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Jmie_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[-2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Jmei_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Jmii_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Jxmex_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[-1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Jmix_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[-2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Pee_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[-.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Pie_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[-.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Pei_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,-.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Pii_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,-.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Pex_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[-.1],[.1]]), 
                nJrecord0=100
            )   
    def test_Pix_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[-.1]]), 
                nJrecord0=100
            )   
    def test_nJrecord0_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.connectivity(
                self,
                Jm=np.array([[1,-1],[2,-2]]),
                Jxm=np.array([[1],[2]]), 
                P=np.array([[.1,.1],[.1,.1]]), 
                Px=np.array([[.1],[.1]]), 
                nJrecord0=-100
            )   
    def test_connectivity_outcomes(self):
        """
        Test connectivity of plasticNeuralNetwork object.
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
        
        pnn = plasticNeuralNetwork(N, frac_exc, frac_ext, T, dt, jestim, jistim, nBinsRecord)
        pnn.connectivity(Jm, Jxm, P, Px, nJrecord0)

        np.random.seed(314)
        assert_array_equal(pnn.J, np.vstack(
            (
                np.hstack(
                    (
                        np.array(
                            Jm[0, 0]
                            * np.random.binomial(1, P[0, 0], (pnn.Ne, pnn.Ne))
                        ),
                        np.array(
                            Jm[0, 1]
                            * np.random.binomial(1, P[0, 1], (pnn.Ne, pnn.Ni))
                        ),
                    )
                ),
                np.hstack(
                    (
                        np.array(
                            Jm[1, 0]
                            * np.random.binomial(1, P[1, 0], (pnn.Ni, pnn.Ne))
                        ),
                        np.array(
                            Jm[1, 1]
                            * np.random.binomial(1, P[1, 1], (pnn.Ni, pnn.Ni))
                        ),
                    )
                ),
            )
        )
        )
        assert_array_equal(pnn.Jx, np.vstack(
            (
                np.array(
                    Jxm[0, 0] * np.random.binomial(1, Px[0, 0], (pnn.Ne, pnn.Nx))
                ),
                np.array(
                    Jxm[1, 0] * np.random.binomial(1, Px[1, 0], (pnn.Ni, pnn.Nx))
                ),
            )
        )
        )
        self.assertEqual(len(pnn.Jrecord_ee[0,:]), nJrecord0)
        self.assertEqual(len(pnn.Jrecord_ei[0,:]), nJrecord0)
        self.assertEqual(len(pnn.Jrecord_ie[0,:]), nJrecord0)
        self.assertEqual(len(pnn.Jrecord_ii[0,:]), nJrecord0)
        self.assertEqual(pnn.numrecordJ_ee, nJrecord0)
        self.assertEqual(pnn.numrecordJ_ei, nJrecord0)
        self.assertEqual(pnn.numrecordJ_ie, nJrecord0)
        self.assertEqual(pnn.numrecordJ_ii, nJrecord0)
        

#%%

class Test_ffwd_spikes:
    def test_cx_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.ffwd_spikes(
                self,
                cx="0.1", 
                rx=10/1000, 
                taujitter=5, 
                T=10000
            ) 
    def test_rx_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.ffwd_spikes(
                self,
                cx=0.1, 
                rx=[10/1000], 
                taujitter=5, 
                T=10000
            ) 
    def test_taujitter_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.ffwd_spikes(
                self,
                cx=0.1, 
                rx=10/1000, 
                taujitter="5", 
                T=10000
            ) 
    def test_T_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.ffwd_spikes(
                self,
                cx=0.1, 
                rx=10/1000, 
                taujitter=5, 
                T="10000"
            ) 
    # Value tests
    def test_cx_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.ffwd_spikes(
                self,
                cx=-0.1, 
                rx=10/1000, 
                taujitter=5, 
                T=10000
            )   
    def test_rx_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.ffwd_spikes(
                self,
                cx=0.1, 
                rx=-10/1000, 
                taujitter=5, 
                T=10000
            )   
    def test_taujitter_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.ffwd_spikes(
                self,
                cx=0.1, 
                rx=10/1000, 
                taujitter=-5, 
                T=10000
            )   
    def test_T_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.ffwd_spikes(
                self,
                cx=0.1, 
                rx=10/1000, 
                taujitter=5, 
                T=-10000
            )   
    def test_ffwd_spikes_outcomes(self):
        """
        Test ffwd spike generation of plasticNeuralNetwork object.
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
        
        pnn = plasticNeuralNetwork(N, frac_exc, frac_ext, T, dt, jestim, jistim, nBinsRecord)
        pnn.ffwd_spikes(cx, rx, taujitter, T)

        # External ffwd spike trains should be Poisson. So the interspike interval (i.e, ISI) should be Exp-distributed.

        # inter-arrival times of events. WLOG Pick the first neuron in the layer.
        inter_arrival_times = pnn.sx[0, pnn.sx[1,:]==1]
        # fit the inter-arrival times to an exponential distribution
        fit_params = expon.fit(inter_arrival_times)
        # perform the KS test
        ks_stat, p_value = kstest(inter_arrival_times, 'expon', fit_params)

        # assert p-value is above significance level (e.g. 0.05)
        assert p_value > 0.05
    

#%%

class Test_simulate(unittest.TestCase):
    def test_Cm_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm="1",
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_gL_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_VT_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_Vre_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_Vth_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_EL_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_DeltaT_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_taue_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_taui_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_taux_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_tauSTDP_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_numrecord_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ee_hebb_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_Jmax_ee_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee="10",
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ee_koh_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh="1",
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_beta_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta="1",
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ie_homeo_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo="1",
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_alpha_ie_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie="1",
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ie_hebb_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb="1",
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_Jmax_ie_hebb_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb="1",
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ei_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei="-1",
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_alpha_ei_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei="1",
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ii_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii="-1",
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_alpha_ii_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii="1",
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_dt_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt="0.1",
                nBinsRecord=100
            ) 
    def test_nBinsRecord_type(self):
        with pytest.raises(TypeError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord="100"
            ) 
    def test_Cm_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=-1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_gL_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=-1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_VT_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_Vre_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_Vth_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_EL_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_DeltaT_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_taue_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_taui_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_taux_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_tauSTDP_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
                VT=-55,
                Vre=-75,
                Vth=-50,
                EL=-72,
                DeltaT=1,
                taue=8,
                taui=4,
                taux=10,
                tauSTDP=-200,
                numrecord=100,
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_numrecord_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ee_hebb_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                eta_ee_hebb=-1,
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_Jmax_ee_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=-10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ee_koh_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=-1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_beta_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=-1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ie_homeo_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=-1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_alpha_ie_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=-1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ie_hebb_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=-1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_Jmax_ie_hebb_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=-1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ei_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_alpha_ei_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=-1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_eta_ii_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_alpha_ii_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=-1,
                dt=0.1,
                nBinsRecord=100
            ) 
    def test_dt_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=-0.1,
                nBinsRecord=100
            ) 
    def test_nBinsRecord_value(self):
        with pytest.raises(ValueError):
            plasticNeuralNetwork.simulate(
                self,
                Cm=1,
                gL=1/15,
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
                Jmax_ee=10,
                eta_ee_koh=1,
                beta=1,
                eta_ie_homeo=1,
                alpha_ie=1,
                eta_ie_hebb=1,
                Jmax_ie_hebb=1,
                eta_ei=-1,
                alpha_ei=1,
                eta_ii=-1,
                alpha_ii=1,
                dt=0.1,
                nBinsRecord=-100
            ) 
    
    def test_simulate_outcomes(self):
        """
        Test network simulation of plasticNeuralNetwork object.
        """
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
        # Correlation of ffwd spike trains.
        cx = 0
        # Timescale of correlation in ms. Jitter spike trains in external layer by taujitter.
        taujitter = 5

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
        eta_ii = 0.015 / 10**3 / Jnorm_ii  # Learning rate
        rho_ii = 0.020  # Target rate 20Hz
        alpha_ii = 2 * rho_ii * tauSTDP

        # Indices of neurons to record currents, voltages
        numrecord = int(100)  # Number to record from each population
        # Number of time bins to average over when recording
        nBinsRecord = 10
        # Number of synapses to be sampled
        nJrecord0 = 1000

        # Set the random seed
        np.random.seed(31415)

        # Define the model.
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
        # Initialize the connectivity
        pnn.connectivity(Jm, Jxm, P, Px, nJrecord0)
        # Generate Poisson ffwd spike trains
        pnn.ffwd_spikes(cx, rx, taujitter, T)
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

        self.assertEqual(len(JRec_ee[:,0]), nJrecord0)
        self.assertEqual(len(JRec_ei[:,0]), nJrecord0)
        self.assertEqual(len(JRec_ie[:,0]), nJrecord0)
        self.assertEqual(len(JRec_ii[:,0]), nJrecord0)

        self.assertEqual(len(IeRec[:,0]), numrecord)
        self.assertEqual(len(IeRec[0,:]), pnn.Ntrec)
        
        self.assertEqual(len(IiRec[:,0]), numrecord)
        self.assertEqual(len(IiRec[0,:]), pnn.Ntrec)

        self.assertEqual(len(IxRec[:,0]), numrecord)
        self.assertEqual(len(IxRec[0,:]), pnn.Ntrec)

        self.assertEqual(len(JRec_ee[0,:]), pnn.Ntrec)
        self.assertEqual(len(JRec_ei[0,:]), pnn.Ntrec)
        self.assertEqual(len(JRec_ie[0,:]), pnn.Ntrec)
        self.assertEqual(len(JRec_ii[0,:]), pnn.Ntrec)

        assert np.all(s >= 0), "Spike times and neuron indices should be non-negative"
        assert np.all(JRec_ee >= 0), "EE weights should be positive"
        assert np.all(JRec_ei <= 0), "EI weights should be negative"
        assert np.all(JRec_ie >= 0), "IE weights should be positive"
        assert np.all(JRec_ii <= 0), "II weights should be negative"
        assert np.all(IeRec >= 0), "Excitatory currents should be positive"
        assert np.all(IiRec <= 0), "Inhibitory currents should be negative"
        assert np.all(IxRec >= 0), "External, exc currents should be positive"
        assert np.all(VRec < 0), "Memberane potential should always be negative"


#%%
