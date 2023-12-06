from __future__ import annotations

import collections
import collections.abc
import functools
import typing
from enum import Enum

import numpy as np
import scipy.constants
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.Magnetics import Magnetics
from fytok.modules.PFActive import PFActive
from fytok.modules.Wall import Wall
from fytok.plugins.equilibrium.fy_eq import FyEqAnalyze
from spdm.data.TimeSeries import TimeSlice
from spdm.mesh.Mesh import Mesh
from spdm.data.Field import Field

from spdm.utils.constants import *
from spdm.utils.logger import logger
from spdm.utils.numeric import bitwise_and, squeeze
from spdm.utils.tags import _not_found_
from spdm.utils.typing import ArrayLike, ArrayType, NumericType, array_type, as_array, as_scalar, is_array, scalar_type

try:
    import freegs
    import freegs.boundary
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(f"Can not find module 'freegs'!") from error


@Equilibrium.register(["freegs"])
class EquilibriumFreeGS(FyEqAnalyze):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._machine = None

        self._eq_solver: freegs.Equilibrium = None

       

    def _setup_machine(self, *args, **kwargs) -> freegs.machine.Machine:
        """update the description of machine
        TODO:
            - check if wall and pf_active is changed
            - update current of coils

        return True if machine is updated
        """
        if self._machine is not None:
            return self._machine

        pf_active: PFActive = kwargs.get("pf_active", None) or getattr(self._parent, "pf_active", None)  # type:ignore

        eq_coils = []

        if pf_active is not None:
            for coil in pf_active.coil:
                rect = coil.element[0].geometry.rectangle

                turns = int(coil.element[0].turns_with_sign)
                t_coil = freegs.machine.Coil(rect.r + rect.width / 2, rect.z + rect.height / 2, turns=turns)
                eq_coils.append((coil.name, t_coil))

        eq_wall = None

        wall: Wall = kwargs.get("wall", None) or getattr(self._parent, "wall", None)  # type:ignore

        if wall is not None:
            eq_wall = freegs.machine.Wall(
                wall.description_2d[0].limiter.unit[0].outline.r, wall.description_2d[0].limiter.unit[0].outline.z
            )

        eq_sensors = []

        # or getattr(self._parent, "magnetics", None) # type:ignore
        magnetics: Magnetics = kwargs.get("magnetics", None)

        if magnetics is not None:
            for b_prob in magnetics.b_field_pol_probe:
                eq_sensors.append(freegs.machine.Sensor(b_prob.position.r, b_prob.position.z, b_prob.name))

            for flux in magnetics.flux_loop:
                for p in flux.position:
                    eq_sensors.append(freegs.machine.FluxLoopSensor(p.r, p.z, flux.name))

        self._machine = freegs.machine.Machine(coils=eq_coils, wall=eq_wall, sensors=eq_sensors)

        logger.info(f"Setup machine description: wall={wall.description_2d[0].type}")

        return self._machine

    def _setup_eq_solver(self,  psi_f: Field, **kwargs) -> freegs.Equilibrium:
        if self._eq_solver is not None:
            return self._eq_solver

        machine = self._setup_machine(**kwargs)

        boundary_type = kwargs.pop("boundary_type", None) or self.code.parameters.get("boundary", "free")

        grid=psi_f.mesh
       
        psi = psi_f.__array__()

        nx, ny = grid.shape

        (Rmin, Zmin) = grid.geometry.bbox.origin

        (Rmax, Zmax) = grid.geometry.bbox.origin + grid.geometry.bbox.dimensions

        if boundary_type == "fixed":
            boundary = freegs.boundary.fixedBoundary
        else:
            boundary = freegs.boundary.freeBoundaryHagenow

        logger.info(f"Using {boundary_type} boundary")

        self._eq_solver = freegs.Equilibrium(
            tokamak=machine, Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax, nx=nx, ny=ny, psi=psi, boundary=boundary
        )

        return self._eq_solver

    def execute1(self,current:Equilibrium.TimeSlice, *previous:Equilibrium.TimeSlice,  **kwargs)  :
        """update the last time slice, base on profiles_2d[-1].psi, and core_profiles_1d, wall, pf_active"""
        super().execute(current, *previous, **kwargs)
        core_profiles: CoreProfiles=self.inputs.get_source("core_profiles") # type:ignore

        core_profiles_1d=core_profiles.time_slice.current.profiles_1d

        eq_solver = self._setup_eq_solver(current.profiles_2d.psi, **kwargs)

        self.code.parameters["boudary"]

        R0 = core_profiles.time_slice.current.vacuum_toroidal_field.r0
        B0 = core_profiles.time_slice.current.vacuum_toroidal_field.b0

        fvac: float = R0 * B0  # type:ignore

        if "Ip" in kwargs and "beta_p" in kwargs:
            Ip = kwargs.pop("Ip")

            beta_p = kwargs.pop("beta_p")

            freegs_profiles = freegs.jtor.ConstrainBetapIp(eq_solver, beta_p, Ip, fvac)

            logger.info(
                f"""Create Profile: Constrain poloidal Beta and plasma current
                        Betap                      = {beta_p} [-],
                        Plasma current Ip          = {Ip} [Amp],
                        R0*B0                      = {fvac} [T.m]
                        """
            )

        elif "pressure_axis" in kwargs and "Ip" in kwargs:
            Ip = kwargs.pop("Ip")

            pressure_axis = kwargs.pop("pressure_axis")

            freegs_profiles = freegs.jtor.ConstrainPaxisIp(eq_solver, pressure_axis, Ip, fvac, Raxis=R0)

            logger.info(
                f"""Create Profile: Constrain pressure on axis and plasma current
                        Plasma pressure on axis    = {pressure_axis} [Pascals],
                        Plasma current Ip          = {Ip} [Amp ],
                        fvac                       = {fvac} [T.m],
                        Raxis                      = {R0} [m]
                        """
            )
        else:
            
            pprime = core_profiles_1d.pprime(core_profiles_1d.grid.psi_norm)

            ffprime = core_profiles_1d.ffprime(core_profiles_1d.grid.psi_norm)

            freegs_profiles = freegs.jtor.ProfilesPprimeFfprime(pprime, ffprime, fvac)

            logger.info("Create Profile: Specified profile functions p'(psi), ff'(psi)")
 
        freegs_constraints = None

        # if boundary_type == "fixed":
        #     freegs_constraints = None

        # psivals = kwargs.pop("lcfs", False)

        # psi_bndry = None

        # if psivals is True:
        #     boundary_outline_r = current_time_slice.boundary.outline.r
        #     boundary_outline_z = current_time_slice.boundary.outline.z
        #     boundary_psi = np.full_like(boundary_outline_r, current_time_slice.boundary.psi)

        #     psivals = np.vstack([boundary_outline_r, boundary_outline_z, boundary_psi]).T

        #     psi_bndry = current_time_slice.boundary.psi

        #     logger.info(f"Using fixed lcfs")

        # xpoints =
        # if xpoints is True:
        #     xpoints = [(x.r, x.z) for x in current_time_slice.boundary.x_point]
        #     logger.info(f"Using xpoints: {xpoints}")
        # elif isinstance(xpoints, typing.Sequence):
        #     logger.info(f"Using xpoints: {xpoints}")

        # isoflux = kwargs.pop("isoflux", False)

        try:
            freegs_constraints = freegs.control.constrain(
                psivals=kwargs.pop("psivals", []),
                isoflux=kwargs.pop("isoflux", []),
                xpoints=kwargs.pop("xpoints", []),
            )
        except Exception as error:
            freegs_constraints = None

        rtol = kwargs.pop("tolerance", 0.1)

        try:
            logger.info("Solve G-S equation START")
            freegs.solve(
                eq_solver,
                freegs_profiles,
                freegs_constraints,
                show=True,
                #  psi_bndry=psi_bndry,
                rtol=rtol,
            )
        except Exception as error:
            raise RuntimeError(f"Solve G-S equation failed [{self.__class__.__name__}]!") from error
        else:
            logger.info(f"Solve G-S equation Done")

      
    def postprocess(self, current: TimeSlice):
        super().postprocess(current)
        
        # psi_norm = self.code.parameters.get("psi_norm", None)
        # if psi_norm is None or psi_norm is _not_found_:
        #     psi_norm = np.linspace(0, 1.0, 128)

        # if self._eq_solver.psi_bndry is not None:
        #     psi = psi_norm * (self._eq_solver.psi_bndry - self._eq_solver.psi_axis) + self._eq_solver.psi_axis
        # else:
        #     psi = None
        # current["global_quantities"]= {
        #         "ip": self._eq_solver.plasmaCurrent()
        #     } 
        # current[ "profiles_1d"]= {
        #         "psi": psi,
        #         # "q": equilibrium.q(psi_norm),
        #         # "pressure": equilibrium.pressure(psi_norm),
        #         "dpressure_dpsi": self._eq_solver.pprime(psi_norm),
        #         # "f": equilibrium.fpol(psi_norm),
        #         "f_df_dpsi": self._eq_solver.ffprime(psi_norm),
        #     } 
   
        # trim=self.code.parameters.get("trim",0)

        # if trim > 0:
        #     current["profiles_2d"]= {
        #             "type": "total",
        #             "grid_type": {"name": "rectangular", "index": 1},
        #             "grid": {
        #                 "dim1": self._eq_solver.R_1D[trim:-trim],
        #                 "dim2": self._eq_solver.Z_1D[trim:-trim],
        #             },
        #             "psi": self._eq_solver.psi()[trim:-trim, trim:-trim],
        #         }
             
        # else:
        current["profiles_2d"]={ 
                    "type": "total",
                    "grid_type": {"name": "rectangular", "index": 1},
                    "grid": {
                        "dim1":0,# self._eq_solver.R_1D,
                        "dim2":0,# self._eq_solver.Z_1D,
                    },
                    # "psi": self._eq_solver.psi(),
                    # "j_tor": equilibrium.Jtor,
                }
           

 