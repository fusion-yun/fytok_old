
import numpy as np
from scipy import constants
from spdm.utils.tags import _not_found_
from spdm.data.Dict import Dict
from spdm.data.Entry import asentry
from spdm.data.Expression import Expression 
from spdm.data.HTree import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property, SpTree

from .Atoms import atoms
from .MagneticCoordSystem import RadialGrid


class Species(Dict[Node]):

    Element = SpeciesElement

    label: str = sp_property()
    """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """

    @property
    def grid(self) -> RadialGrid:
        return self._parent.grid

     

    multiple_states_flag: int = sp_property(default=0)

    @sp_property
    def element(self, value) -> List[Element]:
        if value is None:
            value = asentry(atoms).get(f"{self.label}/element")
        return value

    @sp_property
    def a(self, value) -> float:
        """Mass of ion {dynamic} [Atomic Mass Unit]"""
        if value is None:
            value = sum([a.a * a.atoms_n for a in self.element])
        return value

    @sp_property
    def z(self, value) -> float:
        if value is None:
            value = asentry(atoms).get(f"{self.label}/z")
        return value
