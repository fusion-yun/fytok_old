import typing
from fytok.utils.atoms import nuclear_reaction
from fytok.modules.CoreSources import CoreSources
from spdm.data.Expression import Variable, Expression
from spdm.data.sp_property import sp_tree


@CoreSources.Source.register(["fusion_reaction"])
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
    code = {"name": "fusion_reaction", "description": r"Burning $D + T \rightarrow \alpha$"}  # type: ignore

    def fetch(self, x: Variable, **vars: Expression) -> CoreSources.Source.TimeSlice:
        res: CoreSources.Source.TimeSlice = super().fetch(x, **vars)

        core_source_1d = res.profiles_1d

        reactivities = nuclear_reaction[r"D(t,n)He"].reactivities

        nD = vars.get("ion/D/density", 0.0)
        nT = vars.get("ion/T/density", 0.0)
        nEP = vars.get("ion/alpha/density", 0.0)

        TD = vars.get("ion/D/temperature", 0.0)
        TT = vars.get("ion/T/temperature", 0.0)
        Te = vars.get("electrons/temperature", 0.0)
        ne = vars.get("electrons/density", 0.0)

        Ti = (nD * TD + nT * TT) / (nD + nT)

        S = reactivities(Ti) * nT * nD

        lnGamma = 17

        tau_slowing_down = 1.99 * ((Te / 1000) ** (3 / 2)) / (ne * 1.0e-19 * lnGamma)

        core_source_1d["ion"] = [
            {"label": "D", "particles": -S},
            {"label": "T", "particles": -S},
            # {"label": "He", "particles": nEP / tau_slowing_down},
            {"label": "alpha", "particles": S - nEP / tau_slowing_down},
        ]

        # core_source_1d.electrons["energy"] = nEP / tau_slowing_down * Te

        return res
