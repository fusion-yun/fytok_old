import typing
from fytok.utils.atoms import nuclear_reaction
from fytok.modules.CoreSources import CoreSources
from spdm.data.Expression import Variable, Expression, zero
from spdm.data.sp_property import sp_tree


@CoreSources.Source.register(["fusion_reaction"])
@sp_tree
class FusionReaction(CoreSources.Source):
    """[summary]

    Args:
        CoreSources ([type]): [description]


    $\\alpha$输运模型参考[@angioniGyrokineticCalculationsDiffusive2008; @angioniGyrokineticSimulationsImpurity2009]

    * energetic $\\alpha$ particle density_thermal $n_{\\alpha}$

    $$
    \\frac{\\partial n_{\\alpha}}{\\partial t}+\\nabla\\left(-D_{\\alpha}\\nabla n_{\\alpha}+Vn_{\\alpha}\\right)=-\\frac{n_{\\alpha}}{\\tau_{sd}^{*}}+n_{D}n_{T}\\left\\langle \\sigma v\\right\\rangle _{DT}
    $$

    * $He$ ash density_thermal $n_{He}$

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
    code = {"name": "fusion_reaction", "description": "Fusion reaction"}  # type: ignore

    def fetch(self, x: Variable, **variables: Expression) -> CoreSources.Source.TimeSlice:
        reactions = self.code.parameters.reactions or []

        source_ion = {}

        for tag in reactions:
            reaction = nuclear_reaction[tag]

            r0, r1 = reaction.reactants
            p0, p1 = reaction.products

            n0 = variables.get(f"ion/{r0}/density_thermal")
            n1 = variables.get(f"ion/{r1}/density_thermal")
            # nEP = vars.get("ion/alpha/density_thermal", 0.0)

            T0 = variables.get(f"ion/D/temperature")
            T1 = variables.get(f"ion/T/temperature")

            Ti = (n0 * T0 + n1 * T1) / (n0 + n1)

            S = reaction.reactivities(Ti) * n0 * n1

            source_ion.setdefault(r0, {"particles": zero})["particles"] -= S
            source_ion.setdefault(r1, {"particles": zero})["particles"] -= S
            source_ion.setdefault(p0, {"particles": zero})["particles"] += S
            source_ion.setdefault(p1, {"particles": zero})["particles"] += S

        current: CoreSources.Source.TimeSlice = super().fetch()

        current["profiles_1d/ion"] = [{"label": name, "particles": S["particles"]} for name, S in source_ion.items()]

        return current
