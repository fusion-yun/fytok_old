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


class EquilibriumFreeGS(FyEqAnalyze):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._machine: freegs.machine.Machine = None

        self._eq_solver: freegs.Equilibrium = None

    def _setup_machine(self, **kwargs) -> freegs.machine.Machine:
        """update the description of machine
        TODO:
            - check if wall and pf_active is changed
            - update current of coils

        return True if machine is updated
        """
        logger.debug(f"Setup machine")

        pf_active: PFActive = self.inputs.get_source("pf_active")

        eq_coils = []

        for coil in pf_active.coil:
            rect = coil.element[0].geometry.rectangle

            turns = int(coil.element[0].turns_with_sign)
            t_coil = freegs.machine.Coil(rect.r + rect.width / 2, rect.z + rect.height / 2, turns=turns)
            eq_coils.append((coil.name, t_coil))

        wall: Wall = self.inputs.get_source("wall")  # type:ignore

        eq_wall = freegs.machine.Wall(
            wall.description_2d[0].limiter.unit[0].outline.r,
            wall.description_2d[0].limiter.unit[0].outline.z,
        )

        magnetics: Magnetics = self.inputs.get_source("magnetics")  # type:ignore

        eq_sensors = []

        for b_prob in magnetics.b_field_pol_probe:
            eq_sensors.append(freegs.machine.Sensor(b_prob.position.r, b_prob.position.z, b_prob.name))

        for flux in magnetics.flux_loop:
            eq_sensors.append(freegs.machine.FluxLoopSensor(flux.position[0].r, flux.position[0].z, flux.name))

        self._machine = freegs.machine.Machine(coils=eq_coils, wall=eq_wall, sensors=eq_sensors)

        logger.info(f"Setup machine description: wall={wall.description_2d[0].type}")

        return self._machine

    def preprocess(self, *args, **kwargs):
        super().preprocess(*args, **kwargs)

        if self._eq_solver is None:
            machine = self._setup_machine()

            logger.debug(f"Setup freegs solver")

            boundary_type = self.code.parameters.get("boundary", "free")

            nx, ny = self.code.parameters.get("dims", (128, 128))

            box = self.code.parameters.get("box", None)

            if box is not None:
                Rmin, Zmin, Rmax, Zmax = box
            else:
                wall: Wall = self.inputs.get_source("wall")  # type:ignore
                description_2d = wall.description_2d[0]
                r: array_type = description_2d.limiter.unit[0].outline.r
                z: array_type = description_2d.limiter.unit[0].outline.z

                Rmin = r.min()
                Rmax = r.max()
                Zmin = z.min()
                Zmax = z.max()

            if boundary_type == "fixed":
                boundary = freegs.boundary.fixedBoundary
            else:
                boundary = freegs.boundary.freeBoundaryHagenow

            self._eq_solver = freegs.Equilibrium(
                tokamak=machine,
                Rmin=Rmin,
                Rmax=Rmax,
                Zmin=Zmin,
                Zmax=Zmax,
                nx=nx,
                ny=ny,
                boundary=boundary,
            )

            logger.info(f"Using {boundary_type} boundary")

        logger.debug(f"Setup profiles for equilibrium solver")

        core_profiles: CoreProfiles = self.inputs.get_source("core_profiles", _not_found_)  # type:ignore

        if core_profiles is not _not_found_:
            core_profiles_1d = core_profiles.time_slice.current.profiles_1d

            R0 = core_profiles.time_slice.current.vacuum_toroidal_field.r0
            B0 = core_profiles.time_slice.current.vacuum_toroidal_field.b0
            fvac: float = R0 * B0  # type:ignore
            psi_norm = core_profiles_1d.grid.psi_norm
        else:
            R0 = kwargs.pop("R0", 1.7)
            B0 = kwargs.pop("B0", 3.0)
            fvac = R0 * B0
            psi_norm = self.code.parameters.get("psi_norm", np.linspace(0, 1.0, 128))

        self._freegs_profiles = None

        if "Ip" in kwargs and "beta_p" in kwargs:
            Ip = self.code.parameters.Ip

            beta_p = self.code.parameters.beta_p

            self._freegs_profiles = freegs.jtor.ConstrainBetapIp(self._eq_solver, beta_p, Ip, fvac)

            logger.info(
                f"""Create Profile: Constrain poloidal Beta and plasma current
                        Betap                      = {beta_p} [-],
                        Plasma current Ip          = {Ip} [Amp],
                        R0*B0                      = {fvac} [T.m]
                        """
            )
        elif "pressure_axis" in kwargs and "Ip" in kwargs:
            Ip = self.code.parameters.Ip

            pressure_axis = self.code.parameters.pressure_axis

            self._freegs_profiles = freegs.jtor.ConstrainPaxisIp(self._eq_solver, pressure_axis, Ip, fvac, Raxis=R0)

            logger.info(
                f"""Create Profile: Constrain pressure on axis and plasma current
                        Plasma pressure on axis    = {pressure_axis} [Pascals],
                        Plasma current Ip          = {Ip} [Amp ],
                        fvac                       = {fvac} [T.m],
                        Raxis                      = {R0} [m]
                        """
            )
        elif core_profiles is not _not_found_:
            core_profiles_1d = core_profiles.time_slice.current.profiles_1d

            pprime = core_profiles_1d.pprime(psi_norm)
            ffprime = core_profiles_1d.ffprime(psi_norm)

            self._freegs_profiles = freegs.jtor.ProfilesPprimeFfprime(pprime, ffprime, fvac)

            logger.info("Create Profile: Specified profile functions p'(psi), ff'(psi)")
        else:
            raise RuntimeError(f"Can not create freegs profile!")

        self._freegs_constraints = None

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
            self._freegs_constraints = freegs.control.constrain(
                psivals=kwargs.pop("psivals", []),
                isoflux=kwargs.pop("isoflux", []),
                xpoints=kwargs.pop("xpoints", []),
            )
        except Exception as error:
            raise RuntimeError(f"Can not create freegs constraints!") from error

    def execute(self, current: Equilibrium.TimeSlice, *previous: Equilibrium.TimeSlice):
        """update the last time slice, base on profiles_2d[-1].psi, and core_profiles_1d, wall, pf_active"""
        super().execute(current, *previous)

        rtol = self.code.parameters.get("tolerance", 0.1)

        try:
            logger.info("Solve G-S equation START")
            freegs.solve(
                self._eq_solver,
                self._freegs_profiles,
                constrain=self._freegs_constraints,
                show=False,
                #  psi_bndry=psi_bndry,
                rtol=rtol,
            )
        except Exception as error:
            raise RuntimeError(f"Solve G-S equation failed [{self.__class__.__name__}]!") from error
        else:
            logger.info(f"Solve G-S equation Done")

    def postprocess(self, current: FyEqAnalyze.TimeSlice):
        psi_norm = self.code.parameters.get("psi_norm", np.linspace(0, 1.0, 128))

        psi = psi_norm * (self._eq_solver.psi_bndry - self._eq_solver.psi_axis) + self._eq_solver.psi_axis

        R0 = current.vacuum_toroidal_field.r0 or self.code.parameters.r0  # self._eq_solver.Rgeometric()

        B0 = self._eq_solver.fvac() / R0

        current["global_quantities"] = {"ip": self._eq_solver.plasmaCurrent()}

        current["vacuum_toroidal_field"] = {"b0": B0}

        current["profiles_1d"] = {
            "grid": {
                "psi_axis": self._eq_solver.psi_axis,
                "psi_boundary": self._eq_solver.psi_bndry,
                "psi_norm": psi_norm,
                "rho_tor_boundary": np.sqrt(
                    self._eq_solver.tor_flux(self._eq_solver.psi_bndry) / (scipy.constants.pi * B0)
                ),
                "rho_tor_norm": self._eq_solver.rhotor(psi),
            },
            "psi": psi,
            "q": self._eq_solver.q(psi_norm),
            "pressure": self._eq_solver.pressure(psi_norm),
            "dpressure_dpsi": self._eq_solver.pprime(psi_norm),
            "f": self._eq_solver.fpol(psi_norm),
            "f_df_dpsi": self._eq_solver.ffprime(psi_norm),
        }

        psi2d = self._eq_solver.psi()

        trim = self.code.parameters.get("trim", 0)

        if trim > 0:
            grid2d = {"dim1": self._eq_solver.R_1D[trim:-trim], "dim2": self._eq_solver.Z_1D[trim:-trim]}
            psi2d = psi2d[trim:-trim, trim:-trim]
        else:
            grid2d = {"dim1": self._eq_solver.R_1D, "dim2": self._eq_solver.Z_1D}

        current["profiles_2d"] = {
            "type": "total",
            "grid_type": {"name": "rectangular", "index": 1},
            "grid": grid2d,
            "psi": psi2d,
        }

        super().postprocess(current)


Equilibrium.register(["freegs"], EquilibriumFreeGS)
