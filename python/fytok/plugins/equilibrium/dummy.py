

from fytok.modules.Equilibrium import Equilibrium


@Equilibrium.register(["dummy"])
class EquilibriumDummy(Equilibrium):

    def update(self, *args, time=None,  **kwargs) -> float:
        residual = super().refresh(time=time)
        if len(args) > 0:
            self.update(args[0])
        if len(kwargs) > 0:
            self.update(kwargs)
        return residual


__SP_EXPORT__ = EquilibriumDummy
