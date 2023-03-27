### Test functions in plastic_balanced_network/

#%%
import pytest
import numpy as np

from src.plastic_balanced_network.helpers import plasticNeuralNetwork

#%%

class Test__init__:
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
    
    
#%%
class Test_connectivity:
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
    

#%%

class Test_simulate:
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




#%%







