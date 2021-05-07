import collections
from functools import cached_property, lru_cache

import numpy as np
import scipy.constants
from spdm.data.Function import Function
from spdm.data.Node import List
from spdm.data.Profiles import Profiles
from spdm.util.logger import logger

from ..utilities.IDS import IDS
from .MagneticCoordSystem import RadialGrid


class CoreSources(IDS):
    """CoreSources
    """
    _IDS = "core_sources"

    def __init__(self, *args, grid: RadialGrid = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid

    @property
    def time(self):
        return np.asarray([profile.time for profile in self.profiles_1d])

    class Profiles1D(Profiles):
        def __init__(self, *args, axis=None, **kwargs):
            super().__init__(*args, axis=axis, **kwargs)

        @property
        def time(self) -> float:
            return self._time

        @property
        def grid(self) -> RadialGrid:
            return self._grid

        class Particle(Profiles):
            def __init__(self, *args,  **kwargs):
                super().__init__(*args, **kwargs)

            @cached_property
            def particles(self):
                return Function(self._parent.grid.rho_tor_norm, self["particles"], parent=self._parent)

            @cached_property
            def energy(self):
                d = self["energy"]
                if not isinstance(d, Function):
                    return Function(self._parent.grid.rho_tor_norm, self["energy"], parent=self._parent)
                else:
                    return d

            @cached_property
            def momentum(self):
                return Profiles({
                    "radial": Function(self._parent.grid.rho_tor_norm, self["momentum.radial"], parent=self._parent),
                    "diamagnetic": Function(self._parent.grid.rho_tor_norm, self["momentum.diamagnetic"], parent=self._parent),
                    "parallel": Function(self._parent.grid.rho_tor_norm, self["momentum.parallel"], parent=self._parent),
                    "poloidal": Function(self._parent.grid.rho_tor_norm, self["momentum.poloidal"], parent=self._parent),
                    "toroidal": Function(self._parent.grid.rho_tor_norm, self["momentum.toroidal"], parent=self._parent)
                })

        class Electrons(Profiles):
            def __init__(self,  *args,   **kwargs):
                super().__init__(* args,  **kwargs)

            @cached_property
            def particle(self):
                return (
                    + self.S_neutrals   # ionization source from neutrals (wall recycling, gas puffing, etc),
                    + self.S_nbi        # NBI,
                    + self.S_ext        # optional additional source ‘EXT’,
                    + self.S_ripple     # particle losses induced by toroidal magnetic field ripple.
                )

            @cached_property
            def energy(self):
                return (
                    - self._parent.Qei      # electron–ion collisional energy transfer
                    + self._parent.Qneo     # neoclassical contribution
                    # + self.Qoh              # ohmic
                    # + self.Qe_lh            # LH,
                    # + self.Qe_nbi           # NBI,
                    # + self.Qe_icrh          # ICRH,
                    # + self.Qe_ecrh          # ECRH
                    # + self.Qe_n0            # charge exchange
                    # + self.Qe_ext           # optional additional source ‘EXT’
                    # - self.Qrad             # line radiation
                    # - self.Qbrem            # bremsstrahlung
                    # - self.Qcyclo           # synchroton radiation
                    # + self.Qe_fus           # fusion reactions
                    # + self.Qe_rip           # energy losses induced by toroidal magnetic field ripple
                )

        class Ion(Profiles):
            def __init__(self,  *args,   **kwargs):
                super().__init__(* args,  **kwargs)

            @cached_property
            def energy(self):
                return (
                    + self._parent.Qei      # electron–ion collisional energy transfer
                    - self._parent.Qneo     # neoclassical contribution (opposite sign w.r.t. electron heat equation)
                    + self.Qi_lh            # LH,
                    + self.Qi_nbi           # NBI,
                    + self.Qi_icrh          # ICRH,
                    + self.Qi_ecrh          # ECRH
                    + self.Qi_n0            # charge exchange
                    + self.Qi_ext           # optional additional source EXT’
                    + self.Qi_fus           # fusion reactions
                    + self.Qi_rip           # energy losses induced by toroidal magnetic field ripple
                )

        class Neutral(Profiles):
            def __init__(self,  *args,   **kwargs):
                super().__init__(* args,  **kwargs)

        @cached_property
        def electrons(self):
            return CoreSources.Profiles1D.Electrons(self["electrons"])

        @cached_property
        def ion(self):
            return Profiles([CoreSources.Profiles1D.Ion(d, parent=self) for d in self["ion"]], parent=self)

        @cached_property
        def neutral(self):
            return Profiles([CoreSources.Profiles1D.Neutral(d, parent=self) for d in self["neutral"]], parent=self)

        @cached_property
        def total_ion_energy(self):
            res = Function(self.grid.rho_tor_norm,  0.0)
            for ion in self.ion:
                res += ion.energy
            return res

        @cached_property
        def total_ion_power_inside(self):
            return NotImplemented

        @cached_property
        def torque_tor_inside(self):
            return NotImplemented

        @cached_property
        def j_parallel(self):
            return Function(self.grid.rho_tor_norm, self["j_parallel"])

        @cached_property
        def current_parallel_inside(self):
            return Function(self.grid.rho_tor_norm, self["current_parallel_inside"])

        @cached_property
        def conductivity_parallel(self):
            return Function(self.grid.rho_tor_norm, self["conductivity_parallel"])

        @cached_property
        def Qei(self):
            Te = self._core_profile.profiles_1d.electrons.temperature
            ne = self._core_profile.profiles_1d.electrons.density

            gamma_ei = 15.2 - np.log(ne)/np.log(1.0e20) + np.log(Te)/np.log(1.0e3)
            epsilon = scipy.constants.epsilon_0
            e = scipy.constants.elementary_charge
            me = scipy.constants.electron_mass
            mp = scipy.constants.proton_mass
            PI = scipy.constants.pi
            tau_e = 12*(PI**(3/2))*(epsilon**2)/(e**4)*np.sqrt(me/2)*((e*Te)**(3/2))/ne/gamma_ei

            def qei(ion):
                return ion.density*(ion.z_ion**2)/sum(ele.atoms_n*ele.a for ele in ion.element)*(Te-ion.temperature)

            return sum(qei(ion) for ion in self._core_profile.ions)*(3/2) * e/(mp/me/2)/tau_e

        @cached_property
        def Qneo(self):
            return NotImplemented

        @cached_property
        def Qoh(self):
            return NotImplemented

        @cached_property
        def j_ni(self):
            r"""
                the current density driven by the non-inductive sources
            """
            return (
                self.j_boot         # bootstrap current
                + self.j_nbi        # neutral beam injection (NBI)
                + self.j_lh         # lower hybrid (LH) waves
                + self.j_ec         # electron cyclotron (EC) waves
                + self.j_ic         # ion cyclotron (IC) waves
                + self.j_ext        # current source ‘EXT
            )

    @cached_property
    def profiles_1d(self):
        return List(self["profiles_1d"], default_factory=CoreSources.Profiles1D, parent=self)
