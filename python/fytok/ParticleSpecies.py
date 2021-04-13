import collections
from functools import cached_property

import numpy as np
import scipy
import scipy.constants
from spdm.data.AttributeTree import AttributeTree
from spdm.data.List import List
from spdm.data.Node import Node, _next_
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.numerical.Function import Function
from spdm.util.logger import logger

from fytok.RadialGrid import RadialGrid

from .Profiles import Profiles


class ParticleSpecies(PhysicalGraph):
    def __init__(self, *args, radial_grid: RadialGrid = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._radial_grid = radial_grid

    @property
    def radial_grid(self):
        return self._radial_grid

    @cached_property
    def type(self):
        r"""
            Species type. 
                index=1 for electron; 
                index=2 for ion species in a single/average state (refer to ion structure); 
                index=3 for ion species in a particular state (refer to ion/state structure); 
                index=4 for neutral species in a single/average state (refer to neutral structure); 
                index=5 for neutral species in a particular state (refer to neutral/state structure); 
                index=6 for neutron; 
                index=7 for photon.
            Available options (refer to the children of this identifier structure) :
            Name	            |Index	        |Description
            unspecified	        0	            unspecified
            electron	        1	            Electron
            ion	                2	            Ion species in a single/average state; refer to ion-structure
            ion_state	        3	            Ion species in a particular state; refer to ion/state-structure
            neutral	            4	            Neutral species in a single/average state; refer to neutral-structure
            neutral_state	    5	            Neutral species in a particular state; refer to neutral/state-structure
            neutron	            6	            Neutron
            photon	            7	            Photon
        """
        return self["type"]

    @property
    def label(self):
        return self["label"]

    @property
    def z(self):
        return self["z"]

    @property
    def mass(self):
        return np.sum([e.a*e.atoms_n for e in self.element])

    class Transport(Profiles):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        class ParticleCoeff(PhysicalGraph):
            def __init__(self,   *args, **kwargs):
                super().__init__(*args,   **kwargs)

            @cached_property
            def d(self):
                return Function(self._parent.grid_d.rho_tor_norm, self["d"])

            @cached_property
            def v(self):
                return Function(self._parent.grid_d.rho_tor_norm, self["v"])

            @cached_property
            def flux(self):
                return Function(self._parent.grid_flux.rho_tor_norm, self["flux"])

        class EngeryCoeff(PhysicalGraph):
            def __init__(self, *args, **kwargs):
                super().__init__(*args,   **kwargs)
                self.d = Function(self._parent.grid_d.rho_tor_norm, self["d"])
                self.v = Function(self._parent.grid_v.rho_tor_norm, self["v"])

            @cached_property
            def flux(self):
                self.flux = Function(self._parent.grid_flux.rho_tor_norm, self["flux"])

        @cached_property
        def particles(self):
            return ParticleSpecies.Transport.ParticleCoeff(self["particles"], profile=self._profile, parent=self._parent)

        @cached_property
        def energy(self):
            return ParticleSpecies.Transport.EngeryCoeff(self["energy"], profile=self._profile,  parent=self._parent)

    class Source(Profiles):
        def __init__(self, *args,  **kwargs):
            super().__init__(*args, **kwargs)

        @cached_property
        def particles(self):
            return Function(self._parent.grid.rho_tor_norm, self["particles"], parent=self._parent)

        @cached_property
        def energy(self):
            return Function(self._parent.grid.rho_tor_norm, self["energy"], parent=self._parent)

        @cached_property
        def momentum(self):
            return PhysicalGraph({
                "radial": Function(self._parent.grid.rho_tor_norm, self["momentum.radial"], parent=self._parent),
                "diamagnetic": Function(self._parent.grid.rho_tor_norm, self["momentum.diamagnetic"], parent=self._parent),
                "parallel": Function(self._parent.grid.rho_tor_norm, self["momentum.parallel"], parent=self._parent),
                "poloidal": Function(self._parent.grid.rho_tor_norm, self["momentum.poloidal"], parent=self._parent),
                "toroidal": Function(self._parent.grid.rho_tor_norm, self["momentum.toroidal"], parent=self._parent)
            })

    class BoundaryCondition(Profiles):
        pass

    class Element(PhysicalGraph):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        @cached_property
        def a(self):
            return self["a"]

        @cached_property
        def atoms_n(self):
            return self["atoms_n"]

        @cached_property
        def z_n(self):
            return self["z_n"]

    @cached_property
    def element(self) -> List:
        return List(self["element"], default_factory=lambda d: ParticleSpecies.Element(d, parent=self))

    @cached_property
    def tarnsport(self):
        return ParticleSpecies.Transport(self["transport"])

    @cached_property
    def source(self):
        return ParticleSpecies.Source(self["source"])

    @cached_property
    def boundary_condition(self):
        return ParticleSpecies.BoundaryCondition(self["boundary_condition"])
