
import collections

import numpy as np
from fytok.constants.Atoms import nuclear_reaction
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreSources import CoreSources
from fytok.transport.Equilibrium import Equilibrium
from spdm.numlib.misc import array_like
from scipy import constants
from spdm.data import Function, Query, function_like


class FusionReaction(CoreSources.Source):

    """ [summary]

        Args:
            CoreSources ([type]): [description]


        $\\alpha$输运模型参考[@angioniGyrokineticCalculationsDiffusive2008; @angioniGyrokineticSimulationsImpurity2009]

        * energetic $\\alpha$ particle density $n_{\\alpha}$

        $$
        \\frac{\\partial n_{\\alpha}}{\\partial t}+\\nabla\\left(-D_{\\alpha}\\nabla n_{\\alpha}+Vn_{\\alpha}\\right)=-\\frac{n_{\\alpha}}{\\tau_{sd}^{*}}+n_{D}n_{T}\\left\\langle \\sigma v\\right\\rangle _{DT}
        $$

        * $He$ ash density $n_{He}$

        $$
        \\frac{\\partial n_{He}}{\\partial t}+\\nabla\\left(-D_{He}\\nabla n_{He}+Vn_{He}\\right)=\\frac{n_{\\alpha}}{\\tau_{sd}^{*}}
        $$

        where
        $$
        \\tau_{sd}^{*}=\\ln\\left(v_{\\alpha}^{3}/v_{c}^{3}+1\\right)\\left(m_{e}m_{\\alpha}v_{e}^{3}\\right)/\\left(64\\sqrt{\\pi}e^{4}n_{e}\\ln\\Lambda\\right)
        $$
        is the actual thermalization slowing down time.

        energetic $\\alpha$ particle flux
        $$
        \\frac{R\\Gamma_{\\alpha}}{n_{\\alpha}}=D_{\\alpha}\\left(\\frac{R}{L_{n_{\\alpha}}}C_{p_{\\alpha}}\\right)
        $$
        where
        $$
        D_{\\alpha}=D_{\\text{He}}\\left[0.02+4.5\\left(\\frac{T_{e}}{E_{\\alpha}}\\right)+8\\left(\\frac{T_{e}}{E_{\\alpha}}\\right)^{2}+350\\left(\\frac{T_{e}}{E_{\\alpha}}\\right)^{3}\\right]
        $$
        and
        $$
        C_{p_{\\alpha}}=\\frac{3}{2}\\frac{R}{L_{T_{e}}}\\left[\\frac{1}{\\log\\left[\\left(E_{\\alpha}/E_{c}\\right)^{3/2}+1\\right]\\left[1+\\left(E_{c}/E_{\\alpha}\\right)^{3/2}\\right]}-1\\right]
        $$
        Here $E_{c}$ is the slowing down critical energy. We remind that $E_{c}/E_{\\alpha}=33.05 T_e/E_{\\alpha}$, where $E_{\\alpha}=3500 keV$  is the thirth energy of $\\alpha$ particles.
    """

    def __init__(self, d: collections.abc.Mapping = None,  *args, **kwargs):
        super().__init__(collections.ChainMap(
            {"identifier": {"name": f"fusion",  "index": 13,
                            "description": f"  $D + T -> \\alpha$ burning and slowing down "},
             "code": {"name": "fusion_reaction"}}, d or {}),
            *args, **kwargs)

        self._reactivities = nuclear_reaction[r"D(t,n)\alpha"]["reactivities"]

    def refresh(self, *args,   equilibrium: Equilibrium,  core_profiles: CoreProfiles,     **kwargs) -> None:

        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        core_profiles_1d = core_profiles.profiles_1d

        rho_tor_norm = core_profiles_1d.grid.rho_tor_norm

        ionT: CoreProfiles.Profiles1D.Ion = core_profiles_1d.ion[Query({"label": "T"})]

        ionD: CoreProfiles.Profiles1D.Ion = core_profiles_1d.ion[Query({"label": "D"})]

        ionHe: CoreProfiles.Profiles1D.Ion = core_profiles_1d.ion[Query({"label": "He"})]

        nD = ionD.density(rho_tor_norm)
        nT = ionT.density(rho_tor_norm)
        TD = ionD.temperature(rho_tor_norm)
        TT = ionD.temperature(rho_tor_norm)
        Te = core_profiles_1d.electrons.temperature(rho_tor_norm)
        Ne = core_profiles_1d.electrons.density(rho_tor_norm)

        nAlpha = ionHe.density_fast(rho_tor_norm)

        Ti = (nD*TD + nT*TT)/(nD+nT)

        sDT = Function(rho_tor_norm, nD*nT*self._reactivities(Ti))

        if not ionHe.has_fast_particle:
            self.profiles_1d["ion"] = [
                {"label": "D", "particles": -sDT},
                {"label": "T", "particles": -sDT},
                {"label": "He", "particles": sDT},
            ]
        else:
            lnGamma = 17

            tau_slowing_down = 1.99 * ((Te/1000)**(3/2))/(Ne*1.0e-19*lnGamma)

            S_slowing_down = Function(rho_tor_norm, nAlpha / tau_slowing_down)

            self.profiles_1d["ion"] = [
                {"label": "D", "particles": -sDT},
                {"label": "T", "particles": -sDT},
                {"label": "He",
                 "particles_fast": sDT-S_slowing_down,
                 "particles": S_slowing_down
                 },
            ]


__SP_EXPORT__ = FusionReaction
