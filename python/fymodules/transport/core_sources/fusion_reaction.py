
import collections

import numpy as np
from fytok.numlib.misc import array_like
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.Equilibrium import Equilibrium
from scipy import constants
from spdm.data.Function import Function
from spdm.util.logger import logger


thermal_reactivities = np.array([
    # | $T_i \\ (keV)$ | $D(t,n)\alpha \\ (m^3/s)$ | $^3He(d,p)\alpha \\ (m^3)/s $ | $D(d,p)T \\ (m^3)/s $ | $D(d,p)^3He \\ (m^3)/s $ |
    [0.20, 1.254e-20, 1.414e-29, 4.640e-22, 4.482e-22],
    [0.30, 7.292e-19, 1.033e-26, 2.071e-20, 2.004e-20],
    [0.40, 9.344e-18, 6.537e-25, 2.237e-19, 2.168e-19],
    [0.50, 5.697e-17, 1.241e-23, 1.204e-18, 1.169e-18],
    [0.60, 2.253e-16, 1.166e-22, 4.321e-18, 4.200e-18],
    [0.70, 6.740e-16, 6.960e-22, 1.193e-17, 1.162e-17],
    [0.80, 1.662e-15, 3.032e-21, 2.751e-17, 2.681e-17],
    [1.00, 6.857e-15, 3.057e-20, 1.017e-16, 9.933e-17],
    [1.25, 2.546e-14, 2.590e-19, 3.387e-16, 3.319e-16],
    [1.30, 3.174e-14, 3.708e-19, 4.143e-16, 4.660e-16],
    [1.50, 6.923e-14, 1.317e-18, 8.431e-16, 8.284e-16],
    [1.75, 1.539e-13, 4.813e-18, 1.739e-15, 1.713e-15],
    [1.80, 1.773e-13, 6.053e-18, 1.976e-15, 1.948e-15],
    [2.00, 2.977e-13, 1.399e-17, 3.150e-15, 3.110e-15],
    [2.50, 8.425e-13, 7.477e-17, 7.969e-15, 7.905e-15],
    [3.00, 1.867e-12, 2.676e-16, 1.608e-14, 1.602e-14],
    [4.00, 5.974e-12, 1.710e-15, 4.428e-14, 4.447e-14],
    [5.00, 1.366e-11, 6.377e-15, 9.024e-14, 9.128e-14],
    [6.00, 2.554e-11, 1.739e-14, 1.545e-13, 1.573e-13],
    [8.00, 6.222e-11, 7.504e-14, 3.354e-13, 3.457e-13],
    [10.0, 1.136e-10, 2.126e-13, 5.781e-13, 6.023e-13],
    [12.0, 1.747e-10, 4.715e-13, 8.723e-13, 9.175e-13],
    [15.0, 2.740e-10, 1.175e-12, 1.390e-12, 1.481e-12],
    [20.0, 4.330e-10, 3.482e-12, 2.399e-12, 2.603e-12],
    [30.0, 6.681e-10, 1.363e-11, 4.728e-12, 5.271e-12],
    [40.0, 7.998e-10, 3.160e-11, 7.249e-12, 8.235e-12],
    [50.0, 8.649e-10, 5.554e-11, 9.838e-12, 1.133e-11],
])


class FusionReaction(CoreSources.Source):

    """ [summary]

        Args:
            CoreSources ([type]): [description]



        $\alpha$输运模型参考[@angioniGyrokineticCalculationsDiffusive2008; @angioniGyrokineticSimulationsImpurity2009]

        * energetic $\alpha$ particle density $n_{\alpha}$

        $$
        \frac{\partial n_{\alpha}}{\partial t}+\nabla\left(-D_{\alpha}\nabla n_{\alpha}+Vn_{\alpha}\right)=-\frac{n_{\alpha}}{\tau_{sd}^{*}}+n_{D}n_{T}\left\langle \sigma v\right\rangle _{DT}
        $$

        * $He$ ash density $n_{He}$

        $$
        \frac{\partial n_{He}}{\partial t}+\nabla\left(-D_{He}\nabla n_{He}+Vn_{He}\right)=\frac{n_{\alpha}}{\tau_{sd}^{*}}
        $$

        where
        $$
        \tau_{sd}^{*}=\ln\left(v_{\alpha}^{3}/v_{c}^{3}+1\right)\left(m_{e}m_{\alpha}v_{e}^{3}\right)/\left(64\sqrt{\pi}e^{4}n_{e}\ln\Lambda\right)
        $$
        is the actual thermalization slowing down time.

        energetic $\alpha$ particle flux
        $$
        \frac{R\Gamma_{\alpha}}{n_{\alpha}}=D_{\alpha}\left(\frac{R}{L_{n_{\alpha}}}C_{p_{\alpha}}\right)
        $$
        where
        $$
        D_{\alpha}=D_{\text{He}}\left[0.02+4.5\left(\frac{T_{e}}{E_{\alpha}}\right)+8\left(\frac{T_{e}}{E_{\alpha}}\right)^{2}+350\left(\frac{T_{e}}{E_{\alpha}}\right)^{3}\right]
        $$
        and
        $$
        C_{p_{\alpha}}=\frac{3}{2}\frac{R}{L_{T_{e}}}\left[\frac{1}{\log\left[\left(E_{\alpha}/E_{c}\right)^{3/2}+1\right]\left[1+\left(E_{c}/E_{\alpha}\right)^{3/2}\right]}-1\right]
        $$
        Here $E_{c}$ is the slowing down critical energy. We remind that $E_{c}/E_{\alpha}=33.05 T_e/E_{\alpha}$, where $E_{\alpha}=3500 keV$  is the thirth energy of $\alpha$ particles.
    """

    def __init__(self, d=None, /,  **kwargs):
        super().__init__(d,
                         identifier={
                             "name": f"fusion",
                             "index": 11,
                             "description": f"  $D + T -> \alpha$ burning and slowing down "
                         },   **kwargs)

    def refresh(self, *args,   equilibrium: Equilibrium,  core_profiles: CoreProfiles,     **kwargs) -> float:

        ionT: CoreProfiles.Profiles1D.Ion =
        core_profiles.profiles_1d.ion.get({"label": "T"})
        ionD: CoreProfiles.Profiles1D.Ion =
        core_profiles.profiles_1d.ion.get({"label": "D"})
        ionHe: CoreProfiles.Profiles1D.Ion =
        core_profiles.profiles_1d.ion.get({"label": "He"})
        ionAlpha: CoreProfiles.Profiles1D.Ion =
        core_profiles.profiles_1d.ion.get({"label": "alpha"})

        nD = ionD.density
        nT = ionT.density

        src_DT_burning = nD*nT*rate

        P_alpha = 0

        self.profiles_1d.ion.put({"label": "D", "particles": -src_DT_burning})
        self.profiles_1d.ion.put({"label": "T", "particles": -src_DT_burning})
        self.profiles_1d.ion.put(
            {"label": "alpha", "particle": src_DT_burning})
        self.profiles_1d.neutral.put(
            {"label": "p", "particle": src_DT_burning})

        self.profiles_1d.electrons["temperature"] = P_alpha

        return None


__SP_EXPORT__ = FusionReaction
