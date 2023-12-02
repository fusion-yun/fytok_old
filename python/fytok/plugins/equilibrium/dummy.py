from fytok.modules.Equilibrium import Equilibrium


@Equilibrium.register(["dummy"])
class EquilibriumDummy(Equilibrium):
    code = {"name": "dummy"}


__SP_EXPORT__ = EquilibriumDummy
