import typing

from spdm.data.Expression import Variable, Expression, zero
from fytok.utils.atoms import nuclear_reaction, atoms
from fytok.utils.logger import logger


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


def fusion_sources(
    Sij: typing.Dict[str, Expression],
    Qij: typing.Dict[str, Expression],
    variables: typing.Dict[str, Expression],
    fusion_reactions: typing.List[str],
):
    Te = variables.get("electrons/temperature")
    ne = variables.get("electrons/density")

    lnGamma = 17

    # tau_slowing_down = 1.99 * ((Te / 1000) ** (3 / 2)) / (ne * 1.0e-19 * lnGamma)
    nu_slowing_down = (ne * 1.0e-19 * lnGamma) / (1.99 * ((Te / 1000) ** (3 / 2)))

    for tag in fusion_reactions:
        reaction = nuclear_reaction[tag]

        r0, r1 = reaction.reactants
        p0, p1 = reaction.products

        ash = f"ion/{atoms[p1].label}"
        r0 = f"ion/{r0}"
        r1 = f"ion/{r1}"
        p0 = f"ion/{p0}"
        p1 = f"ion/{p1}"

        n0 = variables.get(f"{r0}/density")
        n1 = variables.get(f"{r1}/density")

        T0 = variables.get(f"{r0}/temperature")
        T1 = variables.get(f"{r1}/temperature")

        Ti = (n0 * T0 + n1 * T1) / (n0 + n1)

        S = reaction.reactivities(Ti) * n0 * n1

        nEP: Expression | None = variables.get(f"{p1}/density")

        Sij[r0] = Sij.get(r0, zero) - S
        Sij[r1] = Sij.get(r1, zero) - S
        Sij[p0] = Sij.get(p0, zero) + S
        Sij[p1] = Sij.get(p1, zero) + S - nEP * nu_slowing_down
        Sij[ash] = Sij.get(ash, zero) + nEP * nu_slowing_down

    return Sij, Qij
