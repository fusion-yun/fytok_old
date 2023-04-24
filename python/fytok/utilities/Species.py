
import numpy as np
from scipy import constants
from spdm.common.tags import _not_found_
from spdm.data.Dict import Dict
from spdm.data.Entry import as_entry
from spdm.data.Function import Function
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property

from ..constants.Atoms import atoms
from ..transport.MagneticCoordSystem import RadialGrid


class SpeciesElement(Dict[Node]):

    a: float = sp_property()
    """Mass of atom {dynamic} [Atomic Mass Unit] """

    atoms_n: int = sp_property()
    """Number of atoms of this element in the molecule {dynamic}"""

    z_n: int = sp_property()


class Species(Dict[Node]):

    Element = SpeciesElement

    label: str = sp_property()
    """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """

    @property
    def grid(self) -> RadialGrid:
        return self._parent.grid

    def _as_child(self, *args, **kwargs):
        value = super()._as_child(*args, **kwargs)
        if isinstance(value, Function):
            value.setdefault_x(self.grid.rho_tor_norm)
        return value

    multiple_states_flag: int = sp_property(default=0)

    @sp_property
    def element(self, value) -> List[Element]:
        if value is None:
            value = as_entry(atoms).get(f"{self.label}/element")
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
            value = as_entry(atoms).get(f"{self.label}/z")
        return value


class SpeciesElectron(Species):

    label: str = sp_property(default_value="electron")

    @sp_property
    def a(self) -> float:
        """Mass of electron {dynamic} [Atomic Mass Unit]"""
        return constants.m_e/constants.m_u

    z: float = sp_property(default_value=-1)


class SpeciesIonState(Dict):
    z_min: float = sp_property()
    """Minimum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""

    z_max: float = sp_property()
    """Maximum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""

    label: str = sp_property()
    """String identifying charge state (e.g. C+, C+2 , C+3, C+4, C+5, C+6, ...) {dynamic}	STR_0D	"""

    vibrational_level: float = sp_property()
    """Vibrational level (can be bundled) {dynamic} [Elementary Charge Unit]	FLT_0D	"""

    vibrational_mode: str = sp_property()
    """Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature. {dynamic}	STR_0D	"""

    electron_configuration: str = sp_property()
    """Configuration of atomic orbitals of this state, e.g. 1s2-2s1 {dynamic}	STR_0D	"""


class SpeciesIon(Species):

    is_impurity: bool = sp_property(default_value=False)

    has_fast_particle: bool = sp_property(default_value=False)

    z_ion: float = sp_property()
    """Ion charge (of the dominant ionisation state; lumped ions are allowed),
    volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """

    neutral_index: int = sp_property(default_value=0)
    """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """

    z_ion_1d: Function = sp_property()
    """Average charge of the ion species (sum of states charge weighted by state density and
    divided by ion density) {dynamic} [-]  """

    z_ion_square_1d: Function = sp_property()
    """Average square charge of the ion species (sum of states square charge weighted by
      state density and divided by ion density) {dynamic} [-]  """

    multiple_states_flag: int = sp_property(default_value=0)
    """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state  {dynamic}    """

    state: List[SpeciesIonState] = sp_property()
    """Quantities related to the different states of the species (ionisation, energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
