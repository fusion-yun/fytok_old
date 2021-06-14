import collections
from dataclasses import dataclass
from typing import Sequence, Union

import matplotlib.pyplot as plt
from fytok.device.PFActive import PFActive
from spdm.numlib import np
import scipy
from spdm.numlib import constants
import scipy.integrate
from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.data.Node import Dict, List
from spdm.data.Profiles import Profiles
from spdm.data.Node import sp_property
from spdm.util.logger import logger
from spdm.util.utilities import _not_found_, try_get

from ..device.Wall import Wall
from ..device.PFActive import PFActive
from ..device.Magnetics import Magnetics
from ..common.GGD import GGD
from ..common.IDS import IDS
from ..common.Misc import RZTuple, VacuumToroidalField
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

    def __init__(self, *args,   **kwargs):
        super().__init__(*args,    **kwargs)

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
    def __init__(self,   *args,  coord: MagneticCoordSystem = None,  **kwargs):
        super().__init__(*args, **kwargs)
        self._coord = coord or self._parent.coordinate_system

    @sp_property
    def beta_pol(self):
        """Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
        return NotImplemented

    @sp_property
    def beta_tor(self):
        """Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by(B0 ^ 2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0 ^ 2  [-]"""
        return NotImplemented

    @sp_property
    def beta_normal(self):
        """Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0[T] / ip[MA][-]"""
        return NotImplemented

    @sp_property
    def ip(self):
        """Plasma current(toroidal component). Positive sign means anti-clockwise when viewed from above.  [A]."""
        return self._parent.profiles_1d.plasma_current[-1]

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

    @sp_property
    def magnetic_axis(self):
        """Magnetic axis position and toroidal field	structure"""
        return self._coord.magnetic_axis

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


class EquilibriumProfiles1D(Profiles):
    """Equilibrium profiles(1D radial grid) as a function of the poloidal flux	"""

    def __init__(self,  *args,  coord: MagneticCoordSystem = None, parent=None,    **kwargs):
        if coord is None:
            coord = parent.coordinate_system

        super().__init__(*args, axis=coord.psi_norm, parent=parent, **kwargs)
        self._coord = coord
        self._grid = self._coord.radial_grid()

    @sp_property
    def ffprime(self) -> Function:
        """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
        # return self._parent._ffprime  # Function(self.psi_norm, self._coord.ffprime(self.psi_norm))
        return self._coord.ffprime

    @sp_property
    def f_df_dpsi(self) -> Function:
        """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
        return self.ffprime

    @sp_property
    def fpol(self) -> Function:
        """Diamagnetic function(F=R B_Phi)[T.m]."""
        return self._coord.fpol

    @sp_property
    def f(self) -> Function:
        """Diamagnetic function(F=R B_Phi)[T.m]."""
        return self.fpol

    @sp_property
    def pprime(self) -> Function:
        return self._coord._pprime

    @sp_property
    def pressure(self) -> Function:
        return self._coord._pressure

    @sp_property
    def dpressure_dpsi(self) -> Function:
        return self.pprime

    @sp_property
    def plasma_current(self) -> Function:
        """Toroidal current driven inside the flux surface.
          .. math:: I_{pl}\equiv\int_{S_{\zeta}}\mathbf{j}\cdot dS_{\zeta}=\frac{\text{gm2}}{4\pi^{2}\mu_{0}}\frac{\partial V}{\partial\psi}\left(\frac{\partial\psi}{\partial\rho}\right)^{2}
         {dynamic}[A]"""
        return self._coord.plasma_current

    @sp_property
    def j_tor(self) -> Function:
        r"""Flux surface averaged toroidal current density = average(j_tor/R) / average(1/R) {dynamic}[A.m ^ -2]. """
        return self.plasma_current.derivative() / (self._coord.psi_boundary - self._coord.psi_axis)/self.dvolume_dpsi * self._r0

    @sp_property
    def j_parallel(self) -> Function:
        r"""Flux surface averaged parallel current density = average(j.B) / B0, where B0 = Equilibrium/Global/Toroidal_Field/B0 {dynamic}[A/m ^ 2]. """
        return self._coord.j_parallel

    @sp_property
    def psi_norm(self) -> Function:
        """Normalized poloidal flux[Wb]. """
        return self._coord.psi_norm

    @sp_property
    def psi(self) -> Function:
        """Poloidal flux[Wb]. """
        return self._coord.psi

    @sp_property
    def dphi_dpsi(self) -> Function:
        return self._coord.dphi_dpsi

    @sp_property
    def q(self) -> Function:
        return self._coord.q

    @sp_property
    def phi(self) -> Function:
        return self._coord.phi

    @sp_property
    def rho_tor(self) -> Function:
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0[m]"""
        return self._coord.rho_tor

    @sp_property
    def rho_tor_norm(self) -> Function:
        return self._coord.rho_tor_norm

    @sp_property
    def drho_tor_dpsi(self) -> Function:
        return self._coord.drho_tor_dpsi

    @sp_property
    def rho_volume_norm(self) -> Function:
        """Normalised square root of enclosed volume(radial coordinate). The normalizing value is the enclosed volume at the equilibrium boundary
            (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)[-]"""
        return self._coord.rho_volume_norm

    @sp_property
    def area(self) -> Function:
        """Cross-sectional area of the flux surface[m ^ 2]"""
        return self._coord.area

    @sp_property
    def darea_dpsi(self) -> Function:
        """Radial derivative of the cross-sectional area of the flux surface with respect to psi[m ^ 2.Wb ^ -1]. """
        return self._coord.darea_dpsi

    @sp_property
    def darea_drho_tor(self) -> Function	:
        """Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor[m]"""
        return self._coord.darea_drho_tor

    @sp_property
    def surface(self):
        """Surface area of the toroidal flux surface[m ^ 2]"""
        return self._coord.surface

    @sp_property
    def volume(self) -> Function:
        """Volume enclosed in the flux surface[m ^ 3]"""
        return self._coord.volume

    @sp_property
    def dvolume_dpsi(self) -> Function:
        r"""
            Radial derivative of the volume enclosed in the flux surface with respect to Psi[m ^ 3.Wb ^ -1].
        """
        return self._coord.dvolume_dpsi

    @sp_property
    def dpsi_drho_tor(self) -> Function:
        return self._coord.dpsi_drho_tor

    @sp_property
    def dpsi_drho_tor_norm(self) -> Function:
        return self.dpsi_drho_tor*self.rho_tor[-1]

    @sp_property
    def dvolume_drho_tor(self) -> Function:
        return self._coord.dvolume_drho_tor

    @sp_property
    def shape_property(self) -> MagneticCoordSystem.ShapePropety:
        return self._coord.shape_property()

    @sp_property
    def geometric_axis(self) -> RZTuple:
        return RZTuple(
            Function(self._grid.psi_norm, self.shape_property.geometric_axis.r),
            Function(self._grid.psi_norm, self.shape_property.geometric_axis.z)
        )

    @sp_property
    def minor_radius(self) -> Function:
        """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
        return self.shape_property.minor_radius

    @sp_property
    def r_inboard(self) -> Function:
        """Radial coordinate(major radius) on the inboard side of the magnetic axis[m]"""
        return self.shape_property.r_inboard

    @sp_property
    def r_outboard(self) -> Function:
        """Radial coordinate(major radius) on the outboard side of the magnetic axis[m]"""
        return self.shape_property.r_outboard

    @sp_property
    def elongation(self) -> Function:
        """Elongation. {dynamic}[-]"""
        return self.shape_property.elongation

    @sp_property
    def triangularity(self) -> Function	:
        """Upper triangularity w.r.t. magnetic axis. {dynamic}[-]"""
        return self.shape_property.triangularity

    @sp_property
    def triangularity_upper(self) -> Function	:
        """Upper triangularity w.r.t. magnetic axis. {dynamic}[-]"""
        return self.shape_property.triangularity_upper

    @sp_property
    def triangularity_lower(self) -> Function:
        """Lower triangularity w.r.t. magnetic axis. {dynamic}[-]"""
        return self.shape_property.triangularity_lower

    @sp_property
    def gm1(self) -> Function:
        r"""
            Flux surface averaged 1/R ^ 2  [m ^ -2]
            .. math: : \left\langle\frac{1}{R^{2}}\right\rangle
        """
        return self._coord.gm1

    @sp_property
    def gm2(self) -> Function:
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2/R^2  [m^-2]
            .. math:: \left\langle\left |\frac{\nabla\rho}{R}\right|^{2}\right\rangle
        """
        return self._coord.gm2

    @sp_property
    def gm3(self) -> Function:
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2  [-]
            .. math:: {\left\langle \left |\nabla\rho\right|^{2}\right\rangle}
        """
        return self._coord.gm3

    @sp_property
    def gm4(self) -> Function:
        r"""
            Flux surface averaged 1/B ^ 2  [T ^ -2]
            .. math: : \left\langle \frac{1}{B^{2}}\right\rangle
        """
        return self._coord.gm4

    @sp_property
    def gm5(self) -> Function:
        r"""
            Flux surface averaged B ^ 2  [T ^ 2]
            .. math: : \left\langle B^{2}\right\rangle
        """
        return self._coord.gm5

    @sp_property
    def gm6(self) -> Function:
        r"""
            Flux surface averaged  .. math: : \left | \nabla \rho_{tor}\right|^2/B^2  [T^-2]
            .. math:: \left\langle \frac{\left |\nabla\rho\right|^{2}}{B^{2}}\right\rangle
        """
        return self._coord.gm6

        # return Function(self._grid.psi_norm, self._coord.surface_average(self.norm_grad_rho_tor**2/self._coord.B2))

    @sp_property
    def gm7(self) -> Function:
        r"""
            Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right |  [-]
            .. math: : \left\langle \left |\nabla\rho\right |\right\rangle
        """
        return self._coord.gm7

    @sp_property
    def gm8(self) -> Function:
        r"""
            Flux surface averaged R[m]
            .. math: : \left\langle R\right\rangle
        """
        return self._coord.gm8

    @sp_property
    def gm9(self) -> Function:
        r"""
            Flux surface averaged 1/R[m ^ -1]
            .. math: : \left\langle \frac{1}{R}\right\rangle
        """
        return self._coord.gm9

    @sp_property
    def magnetic_shear(self) -> Function:
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
        return self._coord.magnetic_shear

    @sp_property
    def trapped_fraction(self) -> Function:
        """Trapped particle fraction[-]
            Tokamak 3ed, 14.10
        """
        d = self.get("trapped_fraction", _not_found_)
        if d is _not_found_:
            epsilon = self.rho_tor/self._coord.vacuum_toroidal_field.r0
            d = np.asarray(1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon)))
        return d

    @sp_property
    def b_field_max(self):
        """Maximum(modulus(B)) on the flux surface(always positive, irrespective of the sign convention for the B-field direction)[T]"""
        return NotImplemented

    @sp_property
    def beta_pol(self):
        """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
        return NotImplemented


class EquilibriumProfiles2D(Dict):
    """
        Equilibrium 2D profiles in the poloidal plane.
    """

    def __init__(self,   *args, coord: MagneticCoordSystem = None,   ** kwargs):
        super().__init__(*args, **kwargs)
        self._coord = coord or getattr(self._parent, "coordinate_system", _not_found_)

    @sp_property
    def grid_type(self) -> RadialGrid:
        return self._coord.grid_type

    @sp_property
    def grid(self):
        return self._coord.grid

    @sp_property
    def r(self) -> np.ndarray:
        """Values of the major radius on the grid  [m] """
        return self._coord.r

    @sp_property
    def z(self) -> np.ndarray:
        """Values of the Height on the grid  [m] """
        return self._coord.z

    @sp_property
    def psi(self):
        """Values of the poloidal flux at the grid in the poloidal plane  [Wb]. """
        return self.apply_psifunc(lambda p: p, unit="Wb")

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


class EquilibriumBoundary(Dict):
    """
        Description of the plasma boundary used by fixed-boundary codes and typically chosen at psi_norm = 99.x%
        of the separatrix
    """

    def __init__(self,  *args, coord: MagneticCoordSystem = None, parent=None,   ** kwargs):
        if coord is None:
            coord = parent.coordinate_system
        super().__init__(*args, axis=coord.psi_norm, parent=parent, **kwargs)
        self._coord = coord

    @sp_property
    def type(self):
        """0 (limiter) or 1 (diverted)  """
        return 1

    @sp_property
    def outline(self) -> RZTuple:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surface(self.psi, o_point=True))
        return RZTuple(*surf.xyz)

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

    @sp_property
    def psi_norm(self) -> float:
        """Value of the normalised poloidal flux at which the boundary is taken (typically 99.x %),
            the flux being normalised to its value at the separatrix """
        return self.get("psi_norm", self._coord.psi_norm[-1])

    @sp_property
    def shape_property(self) -> MagneticCoordSystem.ShapePropety:
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


class EquilibriumBoundarySeparatrix(Profiles):

    def __init__(self,  *args, coord: MagneticCoordSystem = None, parent=None,   ** kwargs):
        if coord is None:
            coord = parent.coordinate_system
        super().__init__(*args, axis=coord.psi_norm, parent=parent, **kwargs)
        self._coord = coord

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
        return self.psi_norm*(self._coord.psi_boundary-self._coord.psi_axis)+self._coord.psi_axis

    @sp_property
    def psi_norm(self) -> float:
        """Value of the normalised poloidal flux at which the boundary is taken (typically 99.x %),
            the flux being normalised to its value at the separatrix """
        return self.get("psi_norm", 1.0)


class EquilibriumTimeSlice(Dict):
    """
       Time slice of   Equilibrium
    """
    Profiles1D = EquilibriumProfiles1D
    Profiles2D = EquilibriumProfiles2D
    GlobalQuantities = EquilibriumGlobalQuantities
    Constraints = EquilibriumConstraints
    Boundary = EquilibriumBoundary
    BoundarySeparatrix = EquilibriumBoundarySeparatrix

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @sp_property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        return self.get("vacuum_toroidal_field", None) or self._parent.vacuum_toroidal_field

    def radial_grid(self, *args, **kwargs) -> RadialGrid:
        return self.coordinate_system.radial_grid(*args, **kwargs)

    @sp_property
    def coordinate_system(self) -> MagneticCoordSystem:
        psirz = self.get("profiles_2d.psi", None)
        if not isinstance(psirz, Field):
            psirz = Field(psirz,
                          self.get("profiles_2d.grid.dim1"),
                          self.get("profiles_2d.grid.dim2"),
                          mesh="rectilinear")

        psi_norm = self.get("coordinate_system.psi_norm", None)

        if not isinstance(psi_norm, np.ndarray):
            psi_norm_bdry = self.get("coordinate_system.psi_norm.boundary", 0.995)
            psi_norm_axis = self.get("coordinate_system.psi_norm.axis", 0.0001)
            npoints = self.get("coordinate_system.psi_norm.npoints", 128)
            psi_norm = np.linspace(psi_norm_axis, psi_norm_bdry, npoints)
            logger.debug((psi_norm[0], psi_norm[-1]))

        p_psi_norm = self.get("profiles_1d.psi_norm", None)
        fpol = self.get("profiles_1d.f", None)
        pressure = self.get("profiles_1d.pressure", None)
        if isinstance(fpol, Function):
            pass
        elif isinstance(fpol, np.ndarray) and fpol.shape == p_psi_norm.shape:
            fpol = Function(p_psi_norm, np.abs(fpol))
        else:
            raise TypeError(f"{type(fpol)}")

        if isinstance(pressure, Function):
            pass
        elif isinstance(pressure, np.ndarray) and pressure.shape == p_psi_norm.shape:
            pressure = Function(p_psi_norm, pressure)
        else:
            raise TypeError(f"{type(pressure)}")

        return MagneticCoordSystem(
            psirz=psirz,
            fpol=fpol,
            pressure=pressure,
            vacuum_toroidal_field=self.vacuum_toroidal_field,
            psi_norm=psi_norm,
            ntheta=self.get("coordinate_system.ntheta", None)
        )

    @sp_property
    def constraints(self) -> Constraints:
        return self.get("constraints", {})

    @sp_property
    def profiles_1d(self) -> Profiles1D:
        return self.get("profiles_1d", {})

    @sp_property
    def profiles_2d(self) -> Profiles2D:
        return self.get("profiles_2d", {})

    @sp_property
    def global_quantities(self) -> GlobalQuantities:
        return self.get("global_quantities", {})

    @sp_property
    def boundary(self) -> Boundary:
        return self.get("boundary", {})

    @sp_property
    def boundary_separatrix(self) -> BoundarySeparatrix:
        return self.get("boundary_separatrix", {})

    def plot(self, axis=None, *args,
             scalar_field=[],
             vector_field=[],
             boundary=False,
             separatrix=True,
             contour=False,
             levels=32, oxpoints=True,
             **kwargs):
        """learn from freegs
        """
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

            axis.add_patch(plt.Polygon(boundary_points, color='g', linestyle='solid',
                                       linewidth=0.2, fill=False, closed=True))

            axis.plot([], [], 'g-', label="Boundary")

        if separatrix is not False:
            r0 = self._entry.find("boundary_separatrix.outline.r", None)
            z0 = self._entry.find("boundary_separatrix.outline.z", None)
            if r0 is not None and z0 is not None:
                axis.add_patch(plt.Polygon(np.vstack([r0, z0]).T, color='b', linestyle=':',
                                           linewidth=1.0, fill=False, closed=True))

            boundary_points = np.vstack([self.boundary_separatrix.outline.r,
                                         self.boundary_separatrix.outline.z]).T

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                       linewidth=0.2, fill=False, closed=False))
            axis.plot([], [], 'r--', label="Separatrix")

        if contour is not False:
            if contour is True:
                contour = 16
            self.coordinate_system.plot_contour(axis, contour)
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


class Equilibrium(IDS):
    r"""
        Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.

        Reference:
            - O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS", Computer Physics Communications 184, 2 (2013), pp. 293--302.

        COCOS  11

        #    Top view
        #             ***************
        #            *               *
        #           *   ***********   *
        #          *   *           *   *
        #         *   *             *   *
        #         *   *             *   *
        #     Ip  v   *             *   ^  \phi
        #         *   *    Z o--->R *   *
        #         *   *             *   *
        #         *   *             *   *
        #         *   *     Bpol    *   *
        #          *   *     o     *   *
        #           *   ***********   *
        #            *               *
        #             ***************
        #               Bpol x
        #            Poloidal view
        #        ^Z
        #        |
        #        |       ************
        #        |      *            *
        #        |     *         ^    *
        #        |     *   \rho /     *
        #        |     *       /      *
        #        +-----*------X-------*---->R
        #        |     *  Ip, \phi   *
        #        |     *              *
        #        |      *            *
        #        |       *****<******
        #        |       Bpol,\theta
        #        |
        #            Cylindrical coordinate      : (R,\phi,Z)
        #    Poloidal plane coordinate   : (\rho,\theta,\phi)
    """
    _IDS = "equilibrium"
    _actor_module_prefix = "transport.equilibrium."
    Constraints = EquilibriumConstraints

    @dataclass
    class State(IDS.State):
        vacuum_toroidal_field: VacuumToroidalField
        time_slice: EquilibriumTimeSlice
        grid_gdd: GGD

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        r0 = self.get("vacuum_toroidal_field.r0", 0.0)
        b0 = abs(self.get("vacuum_toroidal_field.b0", 0.0))
        return {"r0": r0, "b0": b0}

    @sp_property
    def grid_ggd(self) -> GGD:
        return self.get("grid_ggd", {})

    @sp_property
    def time_slice(self) -> EquilibriumTimeSlice:
        return self.get("time_slice", {})

    def update(self,  *args, constraints: Constraints, core_profiles=None, wall: Wall = None,
               pf_active: PFActive = None, magnetics: Magnetics = None, **kwargs):
        super().update(*args, **kwargs)

    ####################################################################################
    # Plot profiles

    def plot(self, axis=None, *args,   time_slice=True, ggd=False, **kwargs):
        if time_slice is not False:
            axis = self.time_slice.plot(axis, *args, **kwargs)
        if ggd:
            axis = self.grid_ggd.plot(axis, *args, **kwargs)
        return axis

    def fetch_profile(self, d):
        if isinstance(d, str):
            data = d
            opts = {"label": d}
        elif isinstance(d, collections.abc.Mapping):
            data = d.get("name", None)
            opts = d.get("opts", {})
        elif isinstance(d, tuple):
            data, opts = d
        elif isinstance(d, Dict):
            data = d.data
            opts = d.opts
        else:
            raise TypeError(f"Illegal profile type! {d}")

        if isinstance(opts, str):
            opts = {"label": opts}

        if isinstance(data, str):
            nlist = data.split(".")
            if len(nlist) == 1:
                data = self.profiles_1d[nlist[0]]
            elif nlist[0] == 'cache':
                data = self.profiles_1d[nlist[1:]]
            else:
                data = self.profiles_1d[nlist]
        elif isinstance(data, list):
            data = np.array(data)
        elif isinstance(d, np.ndarray):
            pass
        else:
            raise TypeError(f"Illegal data type! {type(data)}")

        return data, opts

    def plot_profiles(self, fig_axis, axis, profiles):
        if not isinstance(profiles, list):
            profiles = [profiles]

        for idx, data in enumerate(profiles):
            ylabel = None
            opts = {}
            if isinstance(data, tuple):
                data, ylabel = data
            if isinstance(data, str):
                ylabel = data

            if not isinstance(data, list):
                data = [data]

            for d in data:
                value, opts = self.fetch_profile(d)

                if value is not NotImplemented and value is not None and len(value) > 0:
                    fig_axis[idx].plot(axis.data, value, **opts)
                else:
                    logger.error(f"Can not find profile '{d}'")

            fig_axis[idx].legend(fontsize=6)

            if ylabel:
                fig_axis[idx].set_ylabel(ylabel, fontsize=6).set_rotation(0)
            fig_axis[idx].labelsize = "media"
            fig_axis[idx].tick_params(labelsize=6)
        return fig_axis[-1]

    def plot_full(self, *args,
                  axis=("psi_norm",   r'$(\psi-\psi_{axis})/(\psi_{boundary}-\psi_{axis}) [-]$'),
                  profiles=None,
                  profiles_2d=[],
                  vec_field=[],
                  surface_mesh=False,
                  **kwargs):

        axis, axis_opts = self.fetch_profile(axis)

        assert (axis.data is not NotImplemented)
        nprofiles = len(profiles) if profiles is not None else 0
        if profiles is None or nprofiles <= 1:
            fig, ax_right = plt.subplots(ncols=1, nrows=1, sharex=True)
        else:
            fig, axs = plt.subplots(ncols=2, nrows=nprofiles, sharex=True)
            # left
            ax_left = self.plot_profiles(axs[:, 0], axis, profiles)

            ax_left.set_xlabel(axis_opts.get("label", "[-]"), fontsize=6)

            # right
            gs = axs[0, 1].get_gridspec()
            for ax in axs[:, 1]:
                ax.remove()  # remove the underlying axes
            ax_right = fig.add_subplot(gs[:, 1])

        if surface_mesh:
            self.coordinate_system.plot(ax_right)

        self.plot(ax_right, profiles=profiles_2d, vec_field=vec_field, **kwargs.get("equilibrium", {}))

        self._tokamak.plot_machine(ax_right, **kwargs.get("machine", {}))

        ax_right.legend()
        fig.tight_layout()

        fig.subplots_adjust(hspace=0)
        fig.align_ylabels()

        return fig

    # # Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2]  [-]
    # self.global_quantities.beta_pol = NotImplemented
    # # Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2  [-]
    # self.global_quantities.beta_tor = NotImplemented
    # # Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA]  [-]
    # self.global_quantities.beta_normal = NotImplemented
    # # Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above.  [A].
    # self.global_quantities.ip = NotImplemented
    # # Internal inductance  [-]
    # self.global_quantities.li_3 = NotImplemented
    # # Total plasma volume  [m^3]
    # self.global_quantities.volume = NotImplemented
    # # Area of the LCFS poloidal cross section  [m^2]
    # self.global_quantities.area = NotImplemented
    # # Surface area of the toroidal flux surface  [m^2]
    # self.global_quantities.surface = NotImplemented
    # # Poloidal length of the magnetic surface  [m]
    # self.global_quantities.length_pol = NotImplemented
    # # Poloidal flux at the magnetic axis  [Wb].
    # self.global_quantities.psi_axis = NotImplemented
    # # Poloidal flux at the selected plasma boundary  [Wb].
    # self.global_quantities.psi_boundary = NotImplemented
    # # Magnetic axis position and toroidal field	structure
    # self.global_quantities.magnetic_axis = NotImplemented
    # # q at the magnetic axis  [-].
    # self.global_quantities.q_axis = NotImplemented
    # # q at the 95% poloidal flux surface (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)  [-].
    # self.global_quantities.q_95 = NotImplemented
    # # Minimum q value and position	structure
    # self.global_quantities.q_min = NotImplemented
    # # Plasma energy content = 3/2 * int(p,dV) with p being the total pressure (thermal + fast particles) [J]. Time-dependent; Scalar  [J]
    # self.global_quantities.energy_mhd = NotImplemented
