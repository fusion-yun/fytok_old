from fytok.modules.Equilibrium import Equilibrium


@Equilibrium.register(["dummy"])
class EquilibriumDummy(Equilibrium):
    code = {"name": "dummy"}

    def refresh(self, *args, time=None, **kwargs) -> float:
        residual = super().refresh(time=time)
        if len(args) > 0:
            self.refresh(args[0])
        if len(kwargs) > 0:
            self.refresh(kwargs)
        return residual


__SP_EXPORT__ = EquilibriumDummy
