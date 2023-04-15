import collections
import collections.abc
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import constants
from spdm.common.tags import _not_found_, _undefined_
from spdm.data.Dict import Dict
from spdm.data.Entry import as_entry
from spdm.data.Field import Field
from spdm.data.Function import Function, function_like
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from spdm.util.logger import logger
from spdm.util.misc import convert_to_named_tuple, try_get

from ..common.GGD import GGD
from ..common.IDS import IDS
from ..common.Misc import RZTuple, VacuumToroidalField
from ..device.PFActive import PFActive
from ..device.Wall import Wall
from .MagneticCoordSystem import MagneticCoordSystem, RadialGrid

TOLERANCE = 1.0e-6
EPS = np.finfo(float).eps

TWOPI = 2.0*constants.pi


class EquilibriumConstraintsPurePosition(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)


class EquilibriumConstraints0D(Dict):
    def __init__(self,  d, *args, **kwargs):
        if isinstance(d, float):
            d = {"measured": d}
        super().__init__(*args, **collections.ChainMap(d, kwargs))

    """Measured value {dynamic} [as_parent]	FLT_0D	"""
    measured: float = 0
    """Path to the source data for this measurement in the IMAS data dictionary {dynamic}	STR_0D	"""
    source: str = ""
    """Exact time slice used from the time array of the measurement source data. If the time slice does not exist in the time array of the source data, it means linear interpolation has been used {dynamic} [s]	FLT_0D	"""
    time_measurement: float = 0
    """Integer flag : 1 means exact data, taken as an exact input without being fitted; 0 means the equilibrium code does a least square fit {dynamic}	INT_0D	"""
    exact: float = 0
    """Weight given to the measurement {dynamic} [-]	FLT_0D	"""
    weight: float = 0
    """Value calculated from the reconstructed equilibrium {dynamic} [as_parent]	FLT_0D	"""
    reconstructed: float = 0
    """Squared error normalized by the standard deviation considered in the minimization process : chi_squared = weight^2 *(reconstructed - measured)^2 / sigma^2, where sigma is the standard deviation of the measurement error {dynamic} [as_parent]	FLT_0D    """
    chi_squared: float = 0


class EquilibriumConstraints(Dict):
    r"""
        In case of equilibrium reconstruction under constraints, measurements used to constrain the equilibrium,
        reconstructed values and accuracy of the fit. The names of the child nodes correspond to the following
        definition: the solver aims at minimizing a cost function defined as :
            J=1/2*sum_i [ weight_i^2 (reconstructed_i - measured_i)^2 / sigma_i^2 ]. in which sigma_i is the
            standard deviation of the measurement error (to be found in the IDS of the measurement)
    """
    Constraints0D = EquilibriumConstraints0D
    PurePosition = EquilibriumConstraintsPurePosition

    def refresh(self, *args, **kwargs):
        return

    @sp_property
    def b_field_tor_vacuum_r(self) -> Constraints0D:
        """	Vacuum field times major radius in the toroidal field magnet. Positive sign means anti-clockwise when viewed from above [T.m]	structure	"""
        return self["b_field_tor_vacuum_r"]

    @sp_property
    def bpol_probe(self) -> List[Constraints0D]:
        """Set of poloidal field probes [T]	struct_array [max_size=unbounded]	1- IDS:magnetics/bpol_probe"""
        return self["bpol_probe"]

    @sp_property
    def diamagnetic_flux(self) -> List[Constraints0D]:
        """Diamagnetic flux [Wb]	structure	"""
        return self["diamagnetic_flux"]

    @sp_property
    def faraday_angle(self) -> List[Constraints0D]:
        """Set of faraday angles [rad]	struct_array [max_size=unbounded]	1- IDS:polarimeter/channel"""
        return self["faraday_angle"]

    @sp_property
    def mse_polarisation_angle(self) -> List[Constraints0D]:
        """Set of MSE polarisation angles [rad]	struct_array [max_size=unbounded]	1- IDS:mse/channel"""
        return self["mse_polarisation_ang"]

    @sp_property
    def flux_loop(self) -> List[Constraints0D]:
        """Set of flux loops [Wb]	struct_array [max_size=unbounded]	1- IDS:magnetics/flux_loop"""
        return self["flux_loop"]

    @sp_property
    def ip(self) -> Constraints0D:
        """Plasma current. Positive sign means anti-clockwise when viewed from above [A]	structure	"""
        return self["ip"]

    @dataclass
    class Magnetisation:
        magnetisation_r: EquilibriumConstraints0D
        magnetisation_z: EquilibriumConstraints0D

    @sp_property
    def iron_core_segment(self) -> List[Magnetisation]:
        """Magnetisation M of a set of iron core segments [T]	struct_array [max_size=unbounded]	1- IDS:iron_core/segment"""
        return self["iron_core_segment"]

    @sp_property
    def n_e(self) -> List[Constraints0D]:
        """Set of local density measurements [m^-3]	struct_array [max_size=unbounded]	1- 1...N"""
        return self["n_e"]

    @sp_property
    def n_e_line(self) -> List[Constraints0D]:
        """Set of line integrated density measurements [m^-2]	struct_array [max_size=unbounded]	1- IDS:interferometer/channel"""
        return self["n_e_line"]

    @sp_property
    def pf_current(self) -> List[Constraints0D]:
        """Current in a set of poloidal field coils [A]	struct_array [max_size=unbounded]	1- IDS:pf_active/coil"""
        return self["pf_current"]

    @sp_property
    def pf_passive_current(self) -> List[Constraints0D]:
        """Current in a set of axisymmetric passive conductors [A]	struct_array [max_size=unbounded]	1- IDS:pf_passive/loop"""
        return self["pf_passive_current"]

    @sp_property
    def pressure(self) -> List[Constraints0D]:
        """Set of total pressure estimates [Pa]	struct_array [max_size=unbounded]	1- 1...N"""
        return self["pressure"]

    @sp_property
    def q(self) -> List[Constraints0D]:
        """Set of safety factor estimates at various positions [-]	struct_array [max_size=unbounded]	1- 1...N"""
        return self["q"]

    @sp_property
    def x_point(self) -> PurePosition:
        "Array of X-points, for each of them the RZ position is given	struct_array [max_size=unbounded]	1- 1...N"""
        return self["x_point"]

    @sp_property
    def strike_point(self) -> PurePosition:
        """Array of strike points, for each of them the RZ position is given"""
        return self["strike_point"]


class EquilibriumGlobalQuantities(Dict):

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    beta_pol: float = sp_property()
    """Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""

    beta_tor: float = sp_property()
    """Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by(B0 ^ 2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0 ^ 2  [-]"""

    beta_normal: float = sp_property()
    """Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0[T] / ip[MA][-]"""

    ip: float = sp_property()
    """Plasma current(toroidal component). Positive sign means anti-clockwise when viewed from above.  [A]."""

    # @sp_property
    # def beta_pol(self):
    #     """Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
    #     return NotImplemented

    # @sp_property
    # def beta_tor(self):
    #     """Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by(B0 ^ 2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0 ^ 2  [-]"""
    #     return NotImplemented
    #
    # @sp_property
    # def beta_normal(self):
    #     """Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0[T] / ip[MA][-]"""
    #     return NotImplemented
    #
    # @sp_property
    # def ip(self):
    #     """Plasma current(toroidal component). Positive sign means anti-clockwise when viewed from above.  [A]."""
    #     return NotImplemented #self._parent.profiles_1d.plasma_current[-1]

    @sp_property
    def li_3(self):
        """Internal inductance[-]"""
        return NotImplemented

    @sp_property
    def volume(self):
        """Total plasma volume[m ^ 3]"""
        return NotImplemented

    @sp_property
    def area(self):
        """Area of the LCFS poloidal cross section[m ^ 2]"""
        return NotImplemented

    @sp_property
    def surface(self):
        """Surface area of the toroidal flux surface[m ^ 2]"""
        return NotImplemented

    @sp_property
    def length_pol(self):
        """Poloidal length of the magnetic surface[m]"""
        return NotImplemented

    @cached_property
    def magnetic_axis(self):
        """Magnetic axis position and toroidal field	structure"""
        return convert_to_named_tuple(self._coord.magnetic_axis)

    @sp_property
    def x_points(self):
        _, x = self._coord.critical_points
        return x

    @sp_property
    def psi_axis(self) -> float:
        """Poloidal flux at the magnetic axis[Wb]."""
        o, _ = self._coord.critical_points
        return o[0].psi

    @sp_property
    def psi_boundary(self) -> float:
        """Poloidal flux at the selected plasma boundary[Wb]."""
        _, x = self._coord.critical_points
        if len(x) > 0:
            return x[0].psi
        else:
            raise ValueError(f"No x-point")

    @sp_property
    def q_axis(self):
        """q at the magnetic axis[-]."""
        return NotImplemented

    @sp_property
    def q_95(self):
        """q at the 95 % poloidal flux surface
        (IMAS uses COCOS=11: only positive when toroidal current
        and magnetic field are in same direction)[-]."""
        return NotImplemented

    @sp_property
    def q_min(self):
        """Minimum q value and position structure"""
        return NotImplemented

    @sp_property
    def energy_mhd(self):
        """Plasma energy content: 3/2 * int(p, dV) with p being the total pressure(thermal + fast particles)[J].  Time-dependent  Scalar[J]"""
        return NotImplemented


class EquilibriumProfiles1D(Dict):
    """Equilibrium profiles(1D radial grid) as a function of the poloidal flux	"""

    @cached_property
    def _predefined_psi_norm(self):
        psi = self._entry.get("psi", None)
        return (psi-psi[0])/(psi[-1]-psi[0])

    @sp_property
    def pressure(self) -> Function:
        return function_like(self._predefined_psi_norm, self._entry.get("pressure", None))

    @sp_property
    def dpressure_dpsi(self) -> Function:
        return function_like(self._predefined_psi_norm,  self._entry.get("dpressure_dpsi", None))

    @property
    def pprime(self) -> Function:
        return self.dpressure_dpsi

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @sp_property
    def ffprime(self) -> Function:
        """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
        return function_like(self._coord.psi_norm, self._coord.ffprime)

    @sp_property
    def f_df_dpsi(self) -> Function:
        """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
        return function_like(self._coord.psi_norm, self._coord.ffprime)

    @sp_property
    def fpol(self) -> Function:
        """Diamagnetic function(F=R B_Phi)[T.m]."""
        return function_like(self._coord.psi_norm, self._coord.fpol)

    @sp_property
    def f(self) -> Function:
        """Diamagnetic function(F=R B_Phi)[T.m]."""
        return self.fpol

    @sp_property
    def plasma_current(self) -> Function:
        r"""Toroidal current driven inside the flux surface.
          .. math:: I_{pl}\equiv\int_{S_{\zeta}}\mathbf{j}\cdot dS_{\zeta}=\frac{\text{gm2}}{4\pi^{2}\mu_{0}}\frac{\partial V}{\partial\psi}\left(\frac{\partial\psi}{\partial\rho}\right)^{2}
         {dynamic}[A]"""
        return self.gm2 * self.dvolume_drho_tor / self.dpsi_drho_tor/constants.mu_0

    @sp_property
    def j_tor(self) -> Function:
        r"""Flux surface averaged toroidal current density = average(j_tor/R) / average(1/R) {dynamic}[A.m ^ -2]. """
        return self.plasma_current.derivative() / (self._coord.psi_boundary - self._coord.psi_axis)/self.dvolume_dpsi * self._coord.r0

    @sp_property
    def j_parallel(self) -> Function:
        r"""Flux surface averaged parallel current density = average(j.B) / B0, where B0 = Equilibrium/Global/Toroidal_Field/B0 {dynamic}[A/m ^ 2]. """
        fvac = self._coord._fvac
        d = np.asarray(Function(np.asarray(self.volume), np.asarray(fvac*self.plasma_current/self.fpol)).derivative())
        return self._coord.r0*(self.fpol / fvac)**2 * d

    @sp_property
    def psi_norm(self) -> Function:
        """Normalized poloidal flux[Wb]. """
        return function_like(self._coord.psi_norm, self._coord.psi_norm)

    @sp_property
    def psi(self) -> Function:
        """Poloidal flux[Wb]. """
        return function_like(self._coord.psi_norm, self._coord.psi)

    @sp_property
    def dphi_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dphi_dpsi)

    @sp_property
    def q(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.q)

    @sp_property
    def phi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.phi)

    @sp_property
    def rho_tor(self) -> Function:
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0[m]"""
        return function_like(self._coord.psi_norm, self._coord.rho_tor)

    @sp_property
    def rho_tor_norm(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.rho_tor_norm)

    @sp_property
    def drho_tor_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.drho_tor_dpsi)

    @sp_property
    def rho_volume_norm(self) -> Function:
        """Normalised square root of enclosed volume(radial coordinate). The normalizing value is the enclosed volume at the equilibrium boundary
            (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)[-]"""
        return function_like(self._coord.psi_norm, self._coord.rho_volume_norm)

    @sp_property
    def area(self) -> Function:
        """Cross-sectional area of the flux surface[m ^ 2]"""
        return function_like(self._coord.psi_norm, self._coord.area)

    @sp_property
    def darea_dpsi(self) -> Function:
        """Radial derivative of the cross-sectional area of the flux surface with respect to psi[m ^ 2.Wb ^ -1]. """
        return function_like(self._coord.psi_norm, self._coord.darea_dpsi)

    @sp_property
    def darea_drho_tor(self) -> Function	:
        """Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor[m]"""
        return function_like(self._coord.psi_norm, self._coord.darea_drho_tor)

    @sp_property
    def surface(self):
        """Surface area of the toroidal flux surface[m ^ 2]"""
        return function_like(self._coord.psi_norm, self._coord.surface)

    @sp_property
    def volume(self) -> Function:
        """Volume enclosed in the flux surface[m ^ 3]"""
        return function_like(self._coord.psi_norm, self._coord.volume)

    @sp_property
    def dvolume_dpsi(self) -> Function:
        r"""
            Radial derivative of the volume enclosed in the flux surface with respect to Psi[m ^ 3.Wb ^ -1].
        """
        return function_like(self._coord.psi_norm, self._coord.dvolume_dpsi)

    @sp_property
    def dpsi_drho_tor(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dpsi_drho_tor)

    @sp_property
    def dpsi_drho_tor_norm(self) -> Function:
        return self.dpsi_drho_tor*self.rho_tor[-1]

    @sp_property
    def dvolume_drho_tor(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dvolume_drho_tor)

    @cached_property
    def shape_property(self) -> MagneticCoordSystem.ShapeProperty:
        return self._coord.shape_property()

    @sp_property
    def geometric_axis(self) -> RZTuple:
        return self.shape_property.geometric_axis

    @sp_property
    def minor_radius(self) -> Function:
        """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
        return function_like(self._coord.psi_norm, self.shape_property.minor_radius)

    @sp_property
    def r_inboard(self) -> Function:
        """Radial coordinate(major radius) on the inboard side of the magnetic axis[m]"""
        return function_like(self._coord.psi_norm, self.shape_property.r_inboard)

    @sp_property
    def r_outboard(self) -> Function:
        """Radial coordinate(major radius) on the outboard side of the magnetic axis[m]"""
        return function_like(self._coord.psi_norm, self.shape_property.r_outboard)

    # @sp_property
    # def elongation(self) -> Function:
    #     """Elongation. {dynamic}[-]"""
    #     return self.shape_property.elongation
    @sp_property
    def elongation(self) -> Function:
        """Radial coordinate(major radius) on the outboard side of the magnetic axis[m]"""
        return function_like(self._coord.psi_norm, self.shape_property.elongation)

    @sp_property
    def triangularity(self) -> Function	:
        """Upper triangularity w.r.t. magnetic axis. {dynamic}[-]"""
        return function_like(self._coord.psi_norm, self.shape_property.triangularity)

    @sp_property
    def triangularity_upper(self) -> Function	:
        """Upper triangularity w.r.t. magnetic axis. {dynamic}[-]"""
        return function_like(self._coord.psi_norm, self.shape_property.triangularity_upper)

    @sp_property
    def triangularity_lower(self) -> Function:
        """Lower triangularity w.r.t. magnetic axis. {dynamic}[-]"""
        return function_like(self._coord.psi_norm, self.shape_property.triangularity_lower)

    @sp_property
    def gm1(self) -> Function:
        r"""
            Flux surface averaged 1/R ^ 2  [m ^ -2]
            .. math: : \left\langle\frac{1}{R^{2}}\right\rangle
        """
        return function_like(self._coord.psi_norm, self._coord.gm1)

    @sp_property
    def gm2(self) -> Function:
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2/R^2  [m^-2]
            .. math:: \left\langle\left |\frac{\nabla\rho}{R}\right|^{2}\right\rangle
        """
        return function_like(self._coord.psi_norm, self._coord.gm2)

    @sp_property
    def gm3(self) -> Function:
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2  [-]
            .. math:: {\left\langle \left |\nabla\rho\right|^{2}\right\rangle}
        """
        return function_like(self._coord.psi_norm, self._coord.gm3)

    @sp_property
    def gm4(self) -> Function:
        r"""
            Flux surface averaged 1/B ^ 2  [T ^ -2]
            .. math: : \left\langle \frac{1}{B^{2}}\right\rangle
        """
        return function_like(self._coord.psi_norm, self._coord.gm4)

    @sp_property
    def gm5(self) -> Function:
        r"""
            Flux surface averaged B ^ 2  [T ^ 2]
            .. math: : \left\langle B^{2}\right\rangle
        """
        return function_like(self._coord.psi_norm, self._coord.gm5)

    @sp_property
    def gm6(self) -> Function:
        r"""
            Flux surface averaged  .. math: : \left | \nabla \rho_{tor}\right|^2/B^2  [T^-2]
            .. math:: \left\langle \frac{\left |\nabla\rho\right|^{2}}{B^{2}}\right\rangle
        """
        return function_like(self._coord.psi_norm, self._coord.gm6)

        # return function_like(self._grid.psi_norm, self._coord.surface_average(self.norm_grad_rho_tor**2/self._coord.B2))

    @sp_property
    def gm7(self) -> Function:
        r"""
            Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right |  [-]
            .. math: : \left\langle \left |\nabla\rho\right |\right\rangle
        """
        return function_like(self._coord.psi_norm, self._coord.gm7)

    @sp_property
    def gm8(self) -> Function:
        r"""
            Flux surface averaged R[m]
            .. math: : \left\langle R\right\rangle
        """
        return function_like(self._coord.psi_norm, self._coord.gm8)

    @sp_property
    def gm9(self) -> Function:
        r"""
            Flux surface averaged 1/R[m ^ -1]
            .. math: : \left\langle \frac{1}{R}\right\rangle
        """
        return function_like(self._coord.psi_norm, self._coord.gm9)

    @sp_property
    def magnetic_shear(self) -> Function:
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
        return function_like(self._coord.psi_norm, self._coord.magnetic_shear)

    @sp_property
    def trapped_fraction(self) -> Function:
        """Trapped particle fraction[-]
            Tokamak 3ed, 14.10
        """
        d = self.get("trapped_fraction", _not_found_)
        if d is _not_found_:
            epsilon = self.rho_tor/self._coord.r0
            d = np.asarray(1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon)))
        return function_like(self._coord.psi_norm, d)

    @sp_property
    def b_field_max(self) -> Function:
        """Maximum(modulus(B)) on the flux surface(always positive, irrespective of the sign convention for the B-field direction)[T]"""
        return NotImplemented

    @sp_property
    def beta_pol(self) -> Function:
        """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
        return NotImplemented


class EquilibriumProfiles2D(Dict):
    """
        Equilibrium 2D profiles in the poloidal plane.
    """
    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @cached_property
    def grid(self):
        return convert_to_named_tuple(self._entry.get("grid", {}))

    @cached_property
    def grid_type(self):
        return convert_to_named_tuple(self._entry.get("grid_type", {}))

    @cached_property
    def psi(self) -> Field:
        return self._coord._psirz  # (self._coord.r,self._coord.z)

    @sp_property
    def r(self) -> np.ndarray:
        """Values of the major radius on the grid  [m] """
        return self._coord.r

    @sp_property
    def z(self) -> np.ndarray:
        """Values of the Height on the grid  [m] """
        return self._coord.z

    # @sp_property
    # def psi(self):
    #     """Values of the poloidal flux at the grid in the poloidal plane  [Wb]. """
    #     return self.apply_psifunc(lambda p: p, unit="Wb")

    @sp_property
    def theta(self):
        """	Values of the poloidal angle on the grid  [rad] """
        return NotImplementedError()

    @sp_property
    def phi(self):
        """	Toroidal flux  [Wb]"""
        return self.apply_psifunc("phi")

    @sp_property
    def j_tor(self):
        """	Toroidal plasma current density  [A.m^-2]"""
        return self.apply_psifunc("j_tor")

    @sp_property
    def j_parallel(self):
        """	Parallel (to magnetic field) plasma current density  [A.m^-2]"""
        return self.apply_psifunc("j_parallel")

    @sp_property
    def b_field_r(self):
        """R component of the poloidal magnetic field  [T]"""
        return Field(self._coord.Br, self._coord.r, self._coord.z, mesh_type="curvilinear")

    @sp_property
    def b_field_z(self):
        """Z component of the poloidal magnetic field  [T]"""
        return Field(self._coord.Bz, self._coord.r, self._coord.z, mesh_type="curvilinear")

    @sp_property
    def b_field_tor(self):
        """Toroidal component of the magnetic field  [T]"""
        return Field(self._coord.Btor, self._coord.r, self._coord.z, mesh_type="curvilinear")


class EquilibriumBoundary(Dict[Node]):
    """
        Description of the plasma boundary used by fixed-boundary codes and typically chosen at psi_norm = 99.x%
        of the separatrix
    """
    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @sp_property
    def type(self):
        """0 (limiter) or 1 (diverted)  """
        return 1

    @sp_property
    def outline(self) -> RZTuple:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surface(self.psi, o_point=True))
        return RZTuple(surf.xyz[0], surf.xyz[1])

    @sp_property
    def x_point(self):
        _, xpt = self._parent.critical_points
        return xpt

    @sp_property
    def psi_axis(self) -> float:
        return self._coord.psi_axis

    @sp_property
    def psi_boundary(self) -> float:
        return self._coord.psi_boundary

    @sp_property
    def psi(self) -> float:
        """Value of the poloidal flux at which the boundary is taken  [Wb]"""
        return self.psi_norm*(self._coord.psi_boundary-self._coord.psi_axis)+self._coord.psi_axis

    psi_norm: float = sp_property(default_value=0.999)
    """Value of the normalized poloidal flux at which the boundary is taken (typically 99.x %),
            the flux being normalized to its value at the separatrix """

    @property
    def shape_property(self) -> MagneticCoordSystem.ShapeProperty:
        return self._coord.shape_property(self.psi_norm)

    @sp_property
    def geometric_axis(self) -> RZTuple:
        """RZ position of the geometric axis (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the boundary)"""
        return self.shape_property.geometric_axis

    @sp_property
    def minor_radius(self) -> float:
        """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
        return self.shape_property.minor_radius

    @sp_property
    def elongation(self) -> float:
        """Elongation of the plasma boundary. [-]	"""
        return self.shape_property.elongation

    @sp_property
    def elongation_upper(self) -> float:
        """Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]	"""
        return self.shape_property.elongation_upper

    @sp_property
    def elongation_lower(self) -> float:
        """Elongation(lower half w.r.t. geometric axis) of the plasma boundary. [-]	"""
        return self.shape_property.elongation_lower

    @sp_property
    def triangularity(self) -> float:
        """Triangularity of the plasma boundary. [-]	"""
        return self.shape_property.triangularity

    @sp_property
    def triangularity_upper(self) -> float:
        """Upper triangularity of the plasma boundary. [-]	"""
        return self.shape_property.triangularity_upper

    @sp_property
    def triangularity_lower(self) -> float:
        """Lower triangularity of the plasma boundary. [-]"""
        return self.shape_property.triangularity_lower

    @sp_property
    def strike_point(self)	:
        """Array of strike points, for each of them the RZ position is given	struct_array [max_size=unbounded]	1- 1...N"""
        return NotImplemented

    @sp_property
    def active_limiter_point(self):
        """	RZ position of the active limiter point (point of the plasma boundary in contact with the limiter)"""
        return NotImplemented


class EquilibriumBoundarySeparatrix(Dict[Node]):

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @sp_property
    def type(self):
        """0 (limiter) or 1 (diverted)  """
        return 1

    @sp_property
    def outline(self) -> RZTuple:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surface_by_psi_norm(1.0, o_point=None))
        return RZTuple(*surf.xyz)

    @sp_property
    def psi_axis(self) -> float:
        return self._coord.psi_axis

    @sp_property
    def psi_boundary(self) -> float:
        return self._coord.psi_boundary

    @sp_property
    def psi(self) -> float:
        """Value of the poloidal flux at which the boundary is taken  [Wb]"""
        return self._coord.psi_norm*(self._coord.psi_boundary-self._coord.psi_axis)+self._coord.psi_axis

    psi_norm: float = sp_property(default_value=1.0)
    """Value of the normalized poloidal flux at which the boundary is taken (typically 99.x %),
            the flux being normalized to its value at the separatrix """


class Equilibrium(IDS):
    r"""
        Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.

        Reference:
            - O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS", Computer Physics Communications 184, 2 (2013), pp. 293--302.

        COCOS  11
        ```{text}
            Top view
                     ***************
                    *               *
                   *   ***********   *
                  *   *           *   *
                 *   *             *   *
                 *   *             *   *
             Ip  v   *             *   ^  \phi
                 *   *    Z o--->R *   *
                 *   *             *   *
                 *   *             *   *
                 *   *     Bpol    *   *
                  *   *     o     *   *
                   *   ***********   *
                    *               *
                     ***************
                       Bpol x
                    Poloidal view
                ^Z
                |
                |       ************
                |      *            *
                |     *         ^    *
                |     *   \rho /     *
                |     *       /      *
                +-----*------X-------*---->R
                |     *  Ip, \phi   *
                |     *              *
                |      *            *
                |       *****<******
                |       Bpol,\theta
                |
                    Cylindrical coordinate      : $(R,\phi,Z)$
            Poloidal plane coordinate   : $(\rho,\theta,\phi)$
        ```
    """
    _IDS = "equilibrium"
    _fy_module_prefix = "fymodules.transport.equilibrium."

    Constraints = EquilibriumConstraints

    Profiles1D = EquilibriumProfiles1D
    Profiles2D = EquilibriumProfiles2D
    GlobalQuantities = EquilibriumGlobalQuantities
    Constraints = EquilibriumConstraints
    Boundary = EquilibriumBoundary
    BoundarySeparatrix = EquilibriumBoundarySeparatrix

    def __init__(self,  *args, ** kwargs):
        super().__init__(*args, ** kwargs)

    def refresh(self,  *args,
                wall: Wall = _undefined_,
                pf_active: PFActive = _undefined_,
                core_profiles=_undefined_,
                **kwargs):
        super().refresh(*args, **kwargs)

        # self.profiles_1d.pressure = core_profiles.profiles_1d.pressure
        # self.profiles_1d.pressure = core_profiles.profiles_1d.fpol

        # # call Eq solver
        # psi_2d = self._eq_solver(
        #     {
        #         "vacuum_toroidal_field": self.vacuum_toroidal_field,
        #         "global_quantities": {"ip": self.global_quantities.ip,
        #                               "bet": self.global_quantities.betn
        #                               }
        #     }
        # )
        # self.profiles_2d.psi = psi_2d
        # return {
        #     "psi": psi_2d,
        #     "fpol": fpol,
        #     "pprime": pprime,
        # }
        return

    time: float = sp_property()

    vacuum_toroidal_field: VacuumToroidalField = sp_property()

    grid_ggd: GGD = sp_property()

    constraints: Constraints = sp_property()

    profiles_1d: Profiles1D = sp_property()

    profiles_2d: Profiles2D = sp_property()

    global_quantities: GlobalQuantities = sp_property()

    boundary: Boundary = sp_property()

    boundary_separatrix: BoundarySeparatrix = sp_property()

    # coordinate_system: MagneticCoordSystem = sp_property(create_coordinate_system)
    @sp_property
    def coordinate_system(self, desc) -> MagneticCoordSystem:
        psirz = self.profiles_2d._entry.get("psi", None)

        if not isinstance(psirz, Field):
            psirz = Field(psirz,
                          self.profiles_2d.grid.dim1,
                          self.profiles_2d.grid.dim2,
                          mesh="rectilinear")

        psi_1d = self.profiles_1d._entry.get("psi")
        fpol_1d = self.profiles_1d._entry.get("f", _not_found_)
        if not isinstance(psi_1d, np.ndarray) or len(psi_1d) != len(fpol_1d):
            psi_1d = np.linspace(0, 1.0, len(fpol_1d))

        if isinstance(psi_1d, np.ndarray):
            psi_1d = (psi_1d-psi_1d[0])/(psi_1d[-1]-psi_1d[0])

        # pprime_1d = self.profiles_1d._entry.get("dpressure_dpsi", None)

        return MagneticCoordSystem(
            psirz=psirz,
            B0=self.vacuum_toroidal_field.b0,
            R0=self.vacuum_toroidal_field.r0,
            Ip=self.global_quantities._entry.get("ip"),
            fpol=function_like(psi_1d, fpol_1d),
            # pprime=self.profiles_1d._entry.get("dpressure_dpsi", None),
            # fpol=function_like(psi_norm, self.profiles_1d._entry.get("f", None)),
            # pprime=function_like(psi_norm, self.profiles_1d._entry.get("dpressure_dpsi", None)),
            **desc
        )

    @property
    def radial_grid(self) -> RadialGrid:
        return self.coordinate_system.radial_grid

    # @sp_property
    # def grid_ggd(self) -> GGD:
    #     return self.get("grid_ggd")
    # @sp_property
    # def vacuum_toroidal_field(self) -> VacuumToroidalField:
    #     return {"r0": self.get("vacuum_toroidal_field.r0"), "b0": self.get("vacuum_toroidal_field.b0")}

    # @sp_property
    # def constraints(self) -> Constraints:
    #     return self.get("constraints", {})

    # @sp_property
    # def profiles_1d(self) -> Profiles1D:
    #     return Equilibrium.Profiles1D(self.get("profiles_1d", {}),   parent=self)

    # @sp_property
    # def profiles_2d(self) -> Profiles2D:
    #     return Equilibrium.Profiles2D(self.get("profiles_2d", {}),  parent=self)

    # @sp_property
    # def global_quantities(self) -> GlobalQuantities:
    #     return Equilibrium.GlobalQuantities(self.get("global_quantities", {}), parent=self)

    # @sp_property
    # def boundary(self) -> Boundary:
    #     return Equilibrium.Boundary(self.get("boundary", {}),   parent=self)
    # @sp_property
    # def boundary_separatrix(self) -> BoundarySeparatrix:
    #     return Equilibrium.BoundarySeparatrix(self.get("boundary_separatrix", {}),  parent=self)

    def plot(self, axis=None, /,
             scalar_field=[],
             vector_field=[],
             boundary=True,
             separatrix=True,
             contours=True,
             oxpoints=True,
             **kwargs):
        """
            plot o-point,x-point,lcfs,separatrix and contour of psi
        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as error:
            logger.error(f"Can not load matplotlib! [{error}]")
            return
        if axis is None:
            axis = plt.gca()

        # R = self.profiles_2d.r
        # Z = self.profiles_2d.z
        # psi = self.profiles_2d.psi(R, Z)

        # axis.contour(R[1:-1, 1:-1], Z[1:-1, 1:-1], psi[1:-1, 1:-1], levels=levels, linewidths=0.2)
        if oxpoints is not False:
            o_point, x_point = self.coordinate_system.critical_points

            axis.plot(o_point[0].r,
                      o_point[0].z,
                      'g+',
                      linewidth=0.5,
                      #   markersize=2,
                      label="Magnetic axis")

            if len(x_point) > 0:
                for idx, p in enumerate(x_point):
                    axis.plot(p.r, p.z, 'rx')
                    axis.text(p.r, p.z, idx,
                              horizontalalignment='center',
                              verticalalignment='center')

                axis.plot([], [], 'rx', label="X-Point")

        if boundary is not False:
            boundary_points = np.vstack([self.boundary.outline.r,
                                         self.boundary.outline.z]).T

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='solid',
                                       linewidth=0.5, fill=False, closed=True))

            axis.plot([], [], 'g-', label="Boundary")

        if separatrix is not False:
            # r0 = self._entry.get("boundary_separatrix.outline.r", None)
            # z0 = self._entry.get("boundary_separatrix.outline.z", None)
            # if r0 is not None and z0 is not None:
            #     axis.add_patch(plt.Polygon(np.vstack([r0, z0]).T, color='b', linestyle=':',
            #                                linewidth=1.0, fill=False, closed=True))

            boundary_points = np.vstack([self.boundary_separatrix.outline.r,
                                         self.boundary_separatrix.outline.z]).T

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                       linewidth=0.5, fill=False, closed=False))
            axis.plot([], [], 'r--', label="Separatrix")

        if contours is not False:
            if contours is True:
                contours = 16
            self.coordinate_system.plot_contour(axis, contours)
            # if isinstance(contour, int):
            #     c_list = range(0, self.coordinate_system.mesh.shape[0], int(
            #         self.coordinate_system.mesh.shape[0]/contour+0.5))
            # elif isinstance(contour, collections.abc.Sequcence):
            #     c_list = contour
            # for idx in c_list:
            #     ax0 = self.coordinate_system.mesh.axis(idx, axis=0)

            #     if ax0.xy.shape[1] == 1:
            #         axis.add_patch(plt.Circle(ax0.xy[:, 0], radius=0.05, fill=False,color="b", linewidth=0.2))
            #     else:
            #         axis.add_patch(plt.Polygon(ax0.xy, fill=False, closed=True, color="b", linewidth=0.2))

        for s, opts in scalar_field:
            if s == "psirz":
                self.coordinate_system._psirz.plot(axis, **opts)
            else:
                if "." not in s:
                    sf = f"profiles_2d.{s}"
                # self.coordinate_system.norm_grad_psi
                sf = try_get(self, s, None)
                if isinstance(sf, Field):
                    sf.plot(axis, **opts)
                elif isinstance(sf, np.ndarray):
                    axis.contour(self.profiles_2d.r, self.profiles_2d.z, sf, **opts)
                else:
                    logger.error(f"Can not find field {sf} {type(sf)}!")

        for u, v, opts in vector_field:
            uf = self.profiles_2d[u]
            vf = self.profiles_2d[v]
            axis.streamplot(self.profiles_2d.grid.dim1,
                            self.profiles_2d.grid.dim2,
                            vf, uf, **opts)

        return axis
