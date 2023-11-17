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
    code = {"name": "fusion_reaction", "description": r"Burning $D + T \rightarrow \alpha$"}

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)

    def fetch(self, /, x: Variable, **vars: typing.Dict[str, Expression]) -> CoreSources.Source.TimeSlice:
        res = super().fetch(x=x, **vars)

        reactivities = nuclear_reaction[r"D(t,n)\alpha"]["reactivities"]

        nD: Expression | None = vars.get("ion/D/density_thermal")
        nT = vars.get("ion/T/density_thermal")
        TD = vars.get("ion/T/temperature")
        TT = vars.get("ion/T/temperature")
        Te = vars.get("electrons/temperature")
        ne = vars.get("electrons/density_thermal")
        nAlpha = vars.get("ion/alpha/density")

        Ti = (nD * TD + nT * TT) / (nD + nT)

        sDT = reactivities(Ti)

        core_source_1d = res.profiles_1d

        core_source_1d.ion["D"].particles_decomposed.implicit_part = -sDT * nT
        core_source_1d.ion["T"].particles_decomposed.implicit_part = -sDT * nD
        core_source_1d.ion["alpha"].particles_decomposed.explicit_part = sDT * nD * nT

        return res
