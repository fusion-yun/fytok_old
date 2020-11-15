import collections
import sys
from functools import cached_property
import matplotlib.pyplot as plt
import freegs
from freegs.critical import find_separatrix
import numpy as np
from spdm.util.Interpolate import derivate, integral, interpolate, Interpolate2D, Interpolate1D
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles
from spdm.util.AttributeTree import _next_, AttributeTree
from ...CoreProfiles import CoreProfiles
from ...Equilibrium import Equilibrium


class EquilibriumFreeGS(Equilibrium):
    def __init__(self,  config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        eq_wall = freegs.machine.Wall(self.tokamak.wall.limiter.outline.r,
                                      self.tokamak.wall.limiter.outline.z)

        eq_coils = []

        for coil in self.tokamak.pf_active.coil:
            t_coil = freegs.machine.Coil(
                coil.r+coil.width/2,
                coil.z+coil.height/2,
                turns=coil.turns)
            eq_coils.append((coil.name, t_coil))

        tokamak = freegs.machine.Machine(eq_coils, wall=eq_wall)

        dim1 = self._grid_box.dim1
        dim2 = self._grid_box.dim2

        if config.profiles_2d.psi is NotImplemented:
            psi = None
        else:
            psi =np.array( config.profiles_2d.psi())

        self._backend = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=min(dim1), Rmax=max(dim1),
            Zmin=min(dim2), Zmax=max(dim2),
            nx=len(dim1), ny=len(dim2),
            psi=psi,
            boundary=freegs.boundary.freeBoundaryHagenow)

    def _solve(self, profiles=None, constraints=None,  **kwargs):

        if profiles is None:
            profiles = {}
        elif not isinstance(profiles, collections.abc.Mapping):
            raise NotImplementedError()

        self.vacuum_toroidal_field.b0 = profiles.get("B0", self.vacuum_toroidal_field.b0)

        fvec = self.vacuum_toroidal_field.b0 * self.vacuum_toroidal_field.r0

        if "pprime" in profiles and "ffprime" in profiles:

            psi_norm = profiles.get("psi_norm", self.profiles_1d.psi_norm)
            pprime = Interpolate1D(profiles["pprime"], psi_norm)
            ffprime = Interpolate1D(profiles["ffprime"], psi_norm)

            profiles = freegs.jtor.ProfilesPprimeFfprime(pprime, ffprime, fvec)
            logger.debug("Create Profile:     Specified profile functions p'(psi), ff'(psi)")
        elif "betap" in profiles:
            #  Poloidal beta
            betap = profiles.get("betap", 1.0)
            # Plasma current [Amps]
            ip = profiles.get("ip", 1.0)
            profiles = freegs.jtor.ConstrainBetapIp(profiles["betap"], profiles["ip"], fvec)

            logger.debug(f"""Create Profile: Constrain poloidal Beta and plasma current
                     Plasma pressure on axis    ={pressure} [Pascals],  
                     Plasma current Ip          ={ip} [Amps], 
                     fvec                       ={fvec} [T.m]""")
        else:
            # Plasma pressure on axis [Pascals]
            pressure = profiles.get("pressure", 1.0e3)
            # Plasma current [Amps]
            ip = profiles.get("ip", 1.0e6)
            Raxis = self.vacuum_toroidal_field.r0
            profiles = freegs.jtor.ConstrainPaxisIp(pressure, ip, fvec, Raxis=Raxis)

            logger.debug(f"""Create Profile: Constrain pressure on axis and plasma current
                     Plasma pressure on axis    ={pressure} [Pascals],  
                     Plasma current Ip          ={ip} [Amps], 
                     fvec                       ={fvec} [T.m],
                     Raxis                      ={Raxis} [m]
                     """)

        constraints = freegs.control.constrain(** (constraints or {}))

        try:
            freegs.solve(self._backend, profiles, constraints)
        except ValueError as error:
            raise RuntimeError(f"Solve G-S equation failed [{self.__class__.__name__}]! {error}")

    @property
    def critical_points(self):
        R = self.profiles_2d.r
        Z = self.profiles_2d.z
        psi = self.profiles_2d.psi

        # find magnetic axis (o-point) and x-points
        opt, xpt = freegs.critical.find_critical(R, Z, psi)

        if not opt or not xpt:
            raise RuntimeError(f"Can not find O-point or X-points!")
        return opt, xpt

    class Profiles1D(Equilibrium.Profiles1D):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['_backend'] = eq._backend

        @cached_property
        def pprime(self):
            return self._backend.pprime(self.psi_norm)

        @cached_property
        def pressure(self):
            return self._backend.pressure(self.psi_norm)

        @cached_property
        def f(self):
            return self._backend.fpol(self.psi_norm)

        @cached_property
        def f_df_dpsi(self):
            return self._backend.ffprime(self.psi_norm)

        @cached_property
        def q(self):
            return self._backend.q(self.psi_norm)

    class Profiles2D(Equilibrium.Profiles2D):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['_backend'] = eq._backend

        @cached_property
        def r(self):
            return self._backend.R

        @cached_property
        def z(self):
            return self._backend.Z

        @cached_property
        def psi(self):
            return self._backend.psiRZ(self.r, self.z)

        @cached_property
        def theta(self):
            return NotImplemented

        @cached_property
        def phi(self):
            return NotImplemented

        @cached_property
        def b_field_r(self):
            return self._backend.Br(self.r, self.z)

        @cached_property
        def b_field_z(self):
            return self._backend.BZ(self.r, self.z)

        @cached_property
        def b_field_tor(self):
            return self._backend.Btor(self.r, self.z)

    class GlobalQuantities(Equilibrium.GlobalQuantities):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['_backend'] = eq._backend

        @property
        def beta_pol(self):
            # Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]
            return self._backend.poloidalBeta()

        @property
        def ip(self):
            # Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A].
            return self._backend.plasmaCurrent()

        @property
        def volume(self):
            # Total plasma volume {dynamic} [m^3]
            return self._backend.plasmaVolume()

        @property
        def magnetic_axis(self):
            """ Magnetic axis position and toroidal field	structure"""
            res = super().magnetic_axis
            res.b_field_tor = self._backend.Btor(res.r, res.z)
            return res

    class Boundary(Equilibrium.Boundary):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['_backend'] = eq._backend

        @cached_property
        def psi(self):
            """ Value of the poloidal flux at which the boundary is taken {dynamic} [Wb]"""
            return self._backend.psi_bndry

        @cached_property
        def outline(self):
            opt, xpt = self._eq.critical_points

            lcfs = np.array(freegs.critical.find_separatrix(
                self._backend,
                opoint=opt,
                xpoint=xpt,
                ntheta=self._ntheta))

            return AttributeTree({
                "r": lcfs[:, 0],
                "z": lcfs[:, 1]
            })
