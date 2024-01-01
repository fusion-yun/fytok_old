import typing
import scipy.constants
from spdm.data.Expression import Variable, Expression, zero
from spdm.data.sp_property import sp_tree
from spdm.numlib.misc import step_function_approx
from spdm.utils.typing import array_type
from fytok.utils.atoms import nuclear_reaction, atoms
from fytok.utils.logger import logger

from fytok.modules.CoreSources import CoreSources
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.Utilities import *

PI = scipy.constants.pi


@sp_tree
class FusionReaction(CoreSources.Source):
    """[summary]

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

    identifier = "fusion"

    code = {"name": "fusion", "description": "Fusion reaction"}  # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _x = Variable(0, "x")
        x = np.linspace(0, 10, 256)
        dF = Function(x, 1.0 / (1 + x**1.5))
        F = Function(x, dF.I(x), label="F")
        self._sivukhin = F / (_x)

    def fetch(self, profiles_1d: CoreProfiles.TimeSlice.Profiles1D) -> CoreSources.Source.TimeSlice:
        current: CoreSources.Source.TimeSlice = super().fetch(profiles_1d)

        x = profiles_1d.rho_tor_norm

        source_1d = current.profiles_1d
        me = atoms["electrons"].mass
        Te = profiles_1d.electrons.temperature

        fusion_reactions: typing.List[str] = self.code.parameters.fusion_reactions or []

        ne = profiles_1d.electrons.density

        for tag in fusion_reactions:
            if tag != "D(t,n)alpha":
                raise NotImplementedError(f"NOT IMPLEMENTED YET！！ By now only support D(t,n)alpha!")
            reaction = nuclear_reaction[tag]

            r0, r1 = reaction.reactants
            p0, p1 = reaction.products

            pa = atoms[p1].label

            n0 = profiles_1d.ion[r0].density
            n1 = profiles_1d.ion[r1].density

            T0 = profiles_1d.ion[r0].temperature
            T1 = profiles_1d.ion[r1].temperature
            ni = n0 + n1
            Ti = (n0 * T0 + n1 * T1) / ni
            nEP = profiles_1d.ion[p1].density or zero

            lnGamma = 17

            nu_slowing_down = (ni * 1.0e-19 * lnGamma) / (1.99 * ((Ti / 1000) ** (3 / 2)))

            S = reaction.reactivities(Ti) * n0 * n1

            if r0 == r1:
                S *= 0.5

            source_1d.ion[r0].particles -= S
            source_1d.ion[r1].particles -= S
            source_1d.ion[p0].particles += S
            source_1d.ion[p1].particles += S - nEP * nu_slowing_down
            source_1d.ion[pa].particles += nEP * nu_slowing_down

            E0, E1 = reaction.energy

            Efus = E1 * nEP * nu_slowing_down

            mp: float = atoms[p1].mass
            # 离子加热分量
            #  [Stix, Plasma Phys. 14 (1972) 367 Eq.15
            C = 0.0
            m_tot = 0

            for ion in profiles_1d.ion:
                mi = ion.a * scipy.constants.atomic_mass
                zi = ion.z

                ni = ion.density

                m_tot += mi

                C += ni * zi**2 / (mi / mp)

            Ecrit = Te * (4 * np.sqrt(me / mp) / (3 * np.sqrt(PI) * C / ne)) ** (-2.0 / 3.0)

            frac = self._sivukhin(E1 / Ecrit)

            # 加热离子
            for ion in profiles_1d.ion:
                source_1d.ion[ion.label].energy += Efus * frac * ion.a * scipy.constants.atomic_mass / m_tot

            source_1d.electrons.energy += Efus * (1.0 - frac)

        return current


CoreSources.Source.register(["fusion"], FusionReaction)
