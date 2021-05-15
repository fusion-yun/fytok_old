import collections
from functools import cached_property
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.constants
import scipy.integrate
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Field import Field
from spdm.data.Function import Expression, Function
from spdm.data.Node import Dict, List
from spdm.data.Profiles import Profiles
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.flow.Actor import Actor
from spdm.util.logger import logger
from spdm.util.utilities import convert_to_named_tuple, try_get

from ..common.GGD import GGD
from ..common.IDS import IDS
from ..common.Misc import VacuumToroidalField
from .MagneticCoordSystem import MagneticCoordSystem, RadialGrid

TOLERANCE = 1.0e-6
EPS = np.finfo(float).eps

TWOPI = 2.0*scipy.constants.pi


class EquilibriumConstraints(AttributeTree):
    r"""
        In case of equilibrium reconstruction under constraints, measurements used to constrain the equilibrium,
        reconstructed values and accuracy of the fit. The names of the child nodes correspond to the following
        definition: the solver aims at minimizing a cost function defined as :
            J=1/2*sum_i [ weight_i^2 (reconstructed_i - measured_i)^2 / sigma_i^2 ]. in which sigma_i is the
            standard deviation of the measurement error (to be found in the IDS of the measurement)
    """

    def __init__(self, *args,   **kwargs):
        super().__init__(*args,    **kwargs)


class EquilibriumGlobalQuantities(Profiles):
    def __init__(self,   *args,  coord: MagneticCoordSystem = None,  **kwargs):
        super().__init__(*args, **kwargs)
        self._coord = coord or self._parent.coordinate_system

    @property
    def beta_pol(self):
        """Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
        return NotImplemented

    @property
    def beta_tor(self):
        """Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by(B0 ^ 2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0 ^ 2  [-]"""
        return NotImplemented

    @property
    def beta_normal(self):
        """Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0[T] / ip[MA][-]"""
        return NotImplemented

    @property
    def ip(self):
        """Plasma current(toroidal component). Positive sign means anti-clockwise when viewed from above.  [A]."""
        return self._parent.profiles_1d.plasma_current[-1]

    @property
    def li_3(self):
        """Internal inductance[-]"""
        return NotImplemented

    @property
    def volume(self):
        """Total plasma volume[m ^ 3]"""
        return NotImplemented

    @property
    def area(self):
        """Area of the LCFS poloidal cross section[m ^ 2]"""
        return NotImplemented

    @property
    def surface(self):
        """Surface area of the toroidal flux surface[m ^ 2]"""
        return NotImplemented

    @property
    def length_pol(self):
        """Poloidal length of the magnetic surface[m]"""
        return NotImplemented

    @property
    def magnetic_axis(self):
        """Magnetic axis position and toroidal field	structure"""
        return self._coord.magnetic_axis

    @cached_property
    def x_points(self):
        _, x = self._coord.critical_points
        return x

    @cached_property
    def psi_axis(self):
        """Poloidal flux at the magnetic axis[Wb]."""
        o, _ = self._coord.critical_points
        return o[0].psi

    @cached_property
    def psi_boundary(self):
        """Poloidal flux at the selected plasma boundary[Wb]."""
        _, x = self._coord.critical_points
        if len(x) > 0:
            return x[0].psi
        else:
            raise ValueError(f"No x-point")

    @property
    def q_axis(self):
        """q at the magnetic axis[-]."""
        return NotImplemented

    @property
    def q_95(self):
        """q at the 95 % poloidal flux surface
        (IMAS uses COCOS=11: only positive when toroidal current
        and magnetic field are in same direction)[-]."""
        return NotImplemented

    @property
    def q_min(self):
        """Minimum q value and position structure"""
        return NotImplemented

    @property
    def energy_mhd(self):
        """Plasma energy content: 3/2 * int(p, dV) with p being the total pressure(thermal + fast particles)[J].  Time-dependent  Scalar[J]"""
        return NotImplemented


class EquilibriumProfiles1D(Profiles):
    """Equilibrium profiles(1D radial grid) as a function of the poloidal flux	"""

    def __init__(self,  *args,  coord: MagneticCoordSystem = None,    **kwargs):
        if coord is None:
            coord = self._parent.coordinate_system

        super().__init__(*args, axis=coord.psi_norm, **kwargs)

        self._coord = coord
        self._b0 = np.abs(self._parent.vacuum_toroidal_field.b0)
        self._r0 = self._parent.vacuum_toroidal_field.r0

    @property
    def psi_norm(self) -> Function:
        """Normalized poloidal flux[Wb]. """
        return self._coord.psi_norm

    @cached_property
    def psi(self) -> Function:
        """Poloidal flux[Wb]. """
        return Function(self._axis, self.psi_norm * (self._coord.psi_boundary - self._coord.psi_axis) + self._coord.psi_axis)

    @cached_property
    def vprime(self) -> Function:
        r"""
            .. math: : V^{\prime} =  2 \pi  \int{R / |\nabla \psi | * dl }
            .. math: : V^{\prime}(psi) = 2 \pi  \int{dl * R / |\nabla \psi|}
        """
        return Function(self._axis, self._coord.dvolume_dpsi*(self._coord.psi_boundary-self._coord.psi_axis))

    @cached_property
    def dvolume_dpsi(self) -> Function:
        r"""
            Radial derivative of the volume enclosed in the flux surface with respect to Psi[m ^ 3.Wb ^ -1].
        """
        return Function(self._axis, self._coord.dvolume_dpsi)

    @cached_property
    def volume(self) -> Function:
        """Volume enclosed in the flux surface[m ^ 3]"""
        return self.vprime.antiderivative

    @cached_property
    def ffprime(self) -> Function:
        """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
        # return self._parent._ffprime  # Function(self.psi_norm, self._coord.ffprime(self.psi_norm))
        return Function(self._axis,   self._coord._ffprime(self._axis))

    @property
    def f_df_dpsi(self) -> Function:
        """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
        return self.ffprime

    @cached_property
    def fpol(self) -> Function:
        """Diamagnetic function(F=R B_Phi)[T.m]."""
        return Function(self._axis, self._coord.fpol(self._axis))

    @property
    def f(self) -> Function:
        """Diamagnetic function(F=R B_Phi)[T.m]."""
        return self.fpol

    @cached_property
    def pprime(self) -> Function:
        return Function(self._axis, self._coord._pprime(self._axis))

    @property
    def dpressure_dpsi(self) -> Function:
        return self.pprime

    @cached_property
    def plasma_current(self) -> Function:
        """Toroidal current driven inside the flux surface.
          .. math:: I_{pl}\equiv\int_{S_{\zeta}}\mathbf{j}\cdot dS_{\zeta}=\frac{\text{gm2}}{4\pi^{2}\mu_{0}}\frac{\partial V}{\partial\psi}\left(\frac{\partial\psi}{\partial\rho}\right)^{2}
         {dynamic}[A]"""
        return self._coord.surface_average(self._coord.grad_psi2 / (self._coord.r**2))*self._coord.dvolume_dpsi/scipy.constants.mu_0

    @cached_property
    def j_tor(self) -> Function:
        r"""Flux surface averaged toroidal current density = average(j_tor/R) / average(1/R) {dynamic}[A.m ^ -2]. """
        return self.plasma_current.derivative / (self._coord.psi_boundary - self._coord.psi_axis)/self.dvolume_dpsi * self._r0

    @cached_property
    def j_parallel(self) -> Function:
        r"""Flux surface averaged parallel current density = average(j.B) / B0, where B0 = Equilibrium/Global/Toroidal_Field/B0 {dynamic}[A/m ^ 2]. """
        return (self.fpol**2)/self.dvolume_dpsi * ((self.plasma_current/self.fpol).derivative / (self._coord.psi_boundary - self._coord.psi_axis))/self._b0

    @cached_property
    def dphi_dpsi(self) -> Function:
        return self.gm1 * self.fpol * self.dvolume_dpsi / (TWOPI**2)

    @property
    def q(self) -> Expression:
        r"""
            Safety factor
            (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)[-].
            .. math:: q(\psi) =\frac{d\Phi}{d\psi} =\frac{FV^{\prime}\left\langle R^{-2}\right\rangle }{4\pi^{2}}
        """
        return self.fpol * self.dvolume_dpsi*self._coord.surface_average(1.0/(self._coord.r**2))/(TWOPI**2)

    @cached_property
    def phi(self) -> Function:
        r"""
            Note:
            .. math::
                \Phi_{tor}\left(\psi\right) =\int_{0} ^ {\psi}qd\psi
        """
        return Function(self._axis, self._coord.phi(self._axis))

    @cached_property
    def rho_tor(self) -> Function:
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0[m]"""
        return Function(self._axis, self._coord.rho_tor(self._axis))

    @cached_property
    def rho_tor_norm(self) -> Function:
        return Function(self._axis, self._coord.rho_tor_norm(self._axis))

    @cached_property
    def drho_tor_dpsi(self) -> Function:
        r"""
            .. math::
                \frac{d\rho_{tor}}{d\psi} =\frac{d}{d\psi}\sqrt{\frac{\Phi_{tor}}{\pi B_{0}}} \
                                        =\frac{1}{2\sqrt{\pi B_{0}\Phi_{tor}}}\frac{d\Phi_{tor}}{d\psi} \
                                        =\frac{q}{2\pi B_{0}\rho_{tor}}
        """

        return Function(self._axis[1:],  1.0/self.dpsi_drho_tor[1:])

    @cached_property
    def dpsi_drho_tor(self) -> Function:
        """
            Derivative of Psi with respect to Rho_Tor[Wb/m].
        """
        return (TWOPI*self._b0)*self.rho_tor/self.dphi_dpsi

    @cached_property
    def dpsi_drho_tor_norm(self) -> Function:
        """
            Derivative of Psi with respect to Rho_Tor[Wb/m].
        """
        return self.dpsi_drho_tor*self.rho_tor[-1]

    @cached_property
    def dvolume_drho_tor(self) -> Function:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
        return (4*scipy.constants.pi**2*self._b0)*self.rho_tor/(self.fpol*self.gm1)

    @cached_property
    def shape_property(self) -> Function:
        return self._coord.shape_property()

    @cached_property
    def geometric_axis(self) -> Function:
        return convert_to_named_tuple({
            "r": Function(self._axis, self.shape_property.geometric_axis.r),
            "z": Function(self._axis, self.shape_property.geometric_axis.z),
        })

    @property
    def minor_radius(self) -> Function:
        """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
        return self.shape_property.minor_radius

    @cached_property
    def r_inboard(self) -> Function:
        """Radial coordinate(major radius) on the inboard side of the magnetic axis[m]"""
        return Function(self._axis, self.shape_property.r_inboard)

    @cached_property
    def r_outboard(self) -> Function:
        """Radial coordinate(major radius) on the outboard side of the magnetic axis[m]"""
        return Function(self._axis, self.shape_property.r_outboard)

    @cached_property
    def elongation(self) -> Function:
        """Elongation. {dynamic}[-]"""
        return Function(self._axis, self.shape_property.elongation)

    @cached_property
    def triangularity(self)	:
        """Upper triangularity w.r.t. magnetic axis. {dynamic}[-]"""
        return Function(self._axis, self.shape_property.triangularity)

    @cached_property
    def triangularity_upper(self)	:
        """Upper triangularity w.r.t. magnetic axis. {dynamic}[-]"""
        return Function(self._axis, self.shape_property.triangularity_upper)

    @cached_property
    def triangularity_lower(self)	:
        """Lower triangularity w.r.t. magnetic axis. {dynamic}[-]"""
        return Function(self._axis, self.shape_property.triangularity_lower)

    @cached_property
    def gm1(self) -> Function:
        r"""
            Flux surface averaged 1/R ^ 2  [m ^ -2]
            .. math: : \left\langle\frac{1}{R^{2}}\right\rangle
        """
        return Function(self._axis, self._coord.surface_average(1.0/(self._coord.r**2)))

    @cached_property
    def gm2(self) -> Function:
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2/R^2  [m^-2]
            .. math:: \left\langle\left |\frac{\nabla\rho}{R}\right|^{2}\right\rangle
        """
        return Function(self._axis[1:], self._coord.surface_average(self._coord.grad_psi2/(self._coord.r)**2)[1:] / (self.dpsi_drho_tor[1:]**2))
        # return Function(self._axis, Function(self._axis[1:], d[1:])(self._axis))

    @cached_property
    def gm3(self) -> Function:
        r"""
            Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right|^2  [-]
            .. math:: {\left\langle \left |\nabla\rho\right|^{2}\right\rangle}
        """
        return Function(self._axis[1:], self._coord.surface_average(self._coord.grad_psi2)[1:] / (self.dpsi_drho_tor[1:]**2))

    @cached_property
    def gm4(self) -> Function:
        r"""
            Flux surface averaged 1/B ^ 2  [T ^ -2]
            .. math: : \left\langle \frac{1}{B^{2}}\right\rangle
        """
        return Function(self._axis, self._coord.surface_average(1.0/self._coord.B2))

    @cached_property
    def gm5(self) -> Function:
        r"""
            Flux surface averaged B ^ 2  [T ^ 2]
            .. math: : \left\langle B^{2}\right\rangle
        """
        return Function(self._axis, self._coord.surface_average(self._coord.B2))

    @cached_property
    def gm6(self) -> Function:
        r"""
            Flux surface averaged  .. math: : \left | \nabla \rho_{tor}\right|^2/B^2  [T^-2]
            .. math:: \left\langle \frac{\left |\nabla\rho\right|^{2}}{B^{2}}\right\rangle
        """
        return Function(self._axis[1:], self._coord.surface_average(self._coord.grad_psi2/self._coord.B2)[1:]/(self.dpsi_drho_tor[1:]**2))

        # return Function(self._axis, self._coord.surface_average(self.norm_grad_rho_tor**2/self._coord.B2))

    @cached_property
    def gm7(self) -> Function:
        r"""
            Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right |  [-]
            .. math: : \left\langle \left |\nabla\rho\right |\right\rangle
        """
        return Function(self._axis[1:], self._coord.surface_average(self._coord.norm_grad_psi)[1:]/self.dpsi_drho_tor[1:])
        # d = self._coord.surface_average(self.norm_grad_rho_tor)
        # return Function(self._axis, Function(self._axis[1:], d[1:])(self._axis))

    @cached_property
    def gm8(self) -> Function:
        r"""
            Flux surface averaged R[m]
            .. math: : \left\langle R\right\rangle
        """
        return Function(self._axis, self._coord.surface_average(self._coord.r))

    @cached_property
    def gm9(self) -> Function:
        r"""
            Flux surface averaged 1/R[m ^ -1]
            .. math: : \left\langle \frac{1}{R}\right\rangle
        """
        return Function(self._axis, self._coord.surface_average(1.0/self._coord.r))

    @cached_property
    def magnetic_shear(self) -> Function:
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
        return self.rho_tor/self.q * self.q.derivative

    @cached_property
    def rho_volume_norm(self)	:
        """Normalised square root of enclosed volume(radial coordinate). The normalizing value is the enclosed volume at the equilibrium boundary
            (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)[-]"""
        return NotImplemented

    @cached_property
    def area(self) -> Function:
        """Cross-sectional area of the flux surface[m ^ 2]"""
        return NotImplemented

    @cached_property
    def darea_dpsi(self) -> Function:
        """Radial derivative of the cross-sectional area of the flux surface with respect to psi[m ^ 2.Wb ^ -1]. """
        return NotImplemented

    @cached_property
    def darea_drho_tor(self)	:
        """Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor[m]"""
        return NotImplemented

    @cached_property
    def surface(self) -> Function:
        """Surface area of the toroidal flux surface[m ^ 2]"""
        return NotImplemented

    @cached_property
    def trapped_fraction(self) -> Function:
        """Trapped particle fraction[-]"""
        return Function(self._axis, self["trapped_fraction"])

    @cached_property
    def b_field_max(self) -> Function:
        """Maximum(modulus(B)) on the flux surface(always positive, irrespective of the sign convention for the B-field direction)[T]"""
        return NotImplemented

    @cached_property
    def beta_pol(self) -> Function:
        """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
        return NotImplemented


class EquilibriumProfiles2D(Profiles):
    """
        Equilibrium 2D profiles in the poloidal plane.
    """

    def __init__(self,   *args, coord: MagneticCoordSystem = None,   ** kwargs):
        super().__init__(*args, **kwargs)
        self._coord = coord

    @property
    def grid_type(self):
        return self._coord.grid_type

    @cached_property
    def grid(self):
        return self._coord.grid

    @property
    def r(self):
        """Values of the major radius on the grid  [m] """
        return self._coord.r

    @property
    def z(self):
        """Values of the Height on the grid  [m] """
        return self._coord.z

    @cached_property
    def psi(self):
        """Values of the poloidal flux at the grid in the poloidal plane  [Wb]. """
        return self.apply_psifunc(lambda p: p, unit="Wb")

    @cached_property
    def theta(self):
        """	Values of the poloidal angle on the grid  [rad] """
        return NotImplementedError()

    @cached_property
    def phi(self):
        """	Toroidal flux  [Wb]"""
        return self.apply_psifunc("phi")

    @cached_property
    def j_tor(self):
        """	Toroidal plasma current density  [A.m^-2]"""
        return self.apply_psifunc("j_tor")

    @cached_property
    def j_parallel(self):
        """	Parallel (to magnetic field) plasma current density  [A.m^-2]"""
        return self.apply_psifunc("j_parallel")

    @cached_property
    def b_field_r(self):
        """R component of the poloidal magnetic field  [T]"""
        return Field(self._coord.Br, self._coord.r, self._coord.z, mesh_type="curvilinear")

    @cached_property
    def b_field_z(self):
        """Z component of the poloidal magnetic field  [T]"""
        return Field(self._coord.Bz, self._coord.r, self._coord.z, mesh_type="curvilinear")

    @cached_property
    def b_field_tor(self):
        """Toroidal component of the magnetic field  [T]"""
        return Field(self._coord.Btor, self._coord.r, self._coord.z, mesh_type="curvilinear")


class EquilibriumBoundary(Profiles):
    def __init__(self,  *args, coord: MagneticCoordSystem = None,   ** kwargs):
        super().__init__(*args, **kwargs)
        self._coord = coord

    @cached_property
    def type(self):
        """0 (limiter) or 1 (diverted)  """
        return 1

    @cached_property
    def outline(self):
        """RZ outline of the plasma boundary  """
        RZ = np.asarray([[r, z] for r, z in self._coord.flux_surface_map(1.0)])
        return convert_to_named_tuple(r=RZ[:, 0], z=RZ[:, 1])

    @cached_property
    def x_point(self):
        _, xpt = self._parent.critical_points
        return xpt

    @property
    def psi_axis(self):
        return self._coord.psi_axis

    @property
    def psi_boundary(self):
        return self._coord.psi_boundary

    @cached_property
    def psi(self):
        """Value of the poloidal flux at which the boundary is taken  [Wb]"""
        return self._parent.psi_boundary

    @cached_property
    def psi_norm(self):
        """Value of the normalised poloidal flux at which the boundary is taken (typically 99.x %),
            the flux being normalised to its value at the separatrix """
        return self.psi*0.99

    @cached_property
    def shape_property(self):
        return self._coord.shape_property(1.0)

    @property
    def geometric_axis(self):
        """RZ position of the geometric axis (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the boundary)"""
        return self.shape_property.geometric_axis

    @property
    def minor_radius(self):
        """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
        return self.shape_property.minor_radius

    @property
    def elongation(self):
        """Elongation of the plasma boundary. [-]	"""
        return self.shape_property.elongation

    @property
    def elongation_upper(self):
        """Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]	"""
        return self.shape_property.elongation_upper

    @property
    def elongation_lower(self):
        """Elongation(lower half w.r.t. geometric axis) of the plasma boundary. [-]	"""
        return self.shape_property.elongation_lower

    @property
    def triangularity(self):
        """Triangularity of the plasma boundary. [-]	"""
        return self.shape_property.triangularity

    @property
    def triangularity_upper(self):
        """Upper triangularity of the plasma boundary. [-]	"""
        return self.shape_property.triangularity_upper

    @property
    def triangularity_lower(self):
        """Lower triangularity of the plasma boundary. [-]"""
        return self.shape_property.triangularity_lower

    @cached_property
    def strike_point(self)	:
        """Array of strike points, for each of them the RZ position is given	struct_array [max_size=unbounded]	1- 1...N"""
        return NotImplemented

    @cached_property
    def active_limiter_point(self):
        """	RZ position of the active limiter point (point of the plasma boundary in contact with the limiter)"""
        return NotImplemented


class EquilibriumBoundarySeparatrix(Profiles):
    def __init__(self,   *args,     ** kwargs):
        super().__init__(*args, **kwargs)


class EquilibriumTimeSlice(TimeSlice):
    """
       Time slice of   Equilibrium
    """
    Profiles1D = EquilibriumProfiles1D
    Profiles2D = EquilibriumProfiles2D
    GlobalQuantities = EquilibriumGlobalQuantities
    Constraints = EquilibriumConstraints
    Boundary = EquilibriumBoundary
    BoundarySeparatrix = EquilibriumBoundarySeparatrix

    def __init__(self, *args, vacuum_toroidal_field: VacuumToroidalField = None, **kwargs):
        super().__init__(*args, **kwargs)

        self._vacuum_toroidal_field = vacuum_toroidal_field or \
            VacuumToroidalField(**self["vacuum_toroidal_field"]._as_dict())
        if self._vacuum_toroidal_field.b0 < 0:
            self._vacuum_toroidal_field = VacuumToroidalField(
                self._vacuum_toroidal_field.r0, np.abs(self._vacuum_toroidal_field.b0))

    @property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        return self._vacuum_toroidal_field

    def radial_grid(self, *args, **kwargs) -> RadialGrid:
        return self.coordinate_system.radial_grid(*args, **kwargs)

    @cached_property
    def coordinate_system(self) -> MagneticCoordSystem:
        psirz = Field(self["profiles_2d.psi"],
                      self["profiles_2d.grid.dim1"],
                      self["profiles_2d.grid.dim2"],
                      mesh_type="rectilinear")

        psi_norm = self["profiles_1d.psi_norm"]
        ffprime = self["profiles_1d.f_df_dpsi"]
        pprime = self["profiles_1d.dpressure_dpsi"]
        return MagneticCoordSystem(self["coordinate_system"],
                                   vacuum_toroidal_field=self._vacuum_toroidal_field,
                                   psirz=psirz,
                                   ffprime=Function(psi_norm, ffprime),
                                   pprime=Function(psi_norm, pprime),
                                   parent=self)

    @cached_property
    def constraints(self):
        return EquilibriumTimeSlice.Constraints(self["constraints"], coord=self.coordinate_system, parent=self)

    @cached_property
    def profiles_1d(self) -> Profiles1D:
        return EquilibriumTimeSlice.Profiles1D(self["profiles_1d"], coord=self.coordinate_system, parent=self)

    @cached_property
    def profiles_2d(self) -> Profiles2D:
        return EquilibriumTimeSlice.Profiles2D(self["profiles_2d"], coord=self.coordinate_system,  parent=self)

    @cached_property
    def global_quantities(self) -> GlobalQuantities:
        return EquilibriumTimeSlice.GlobalQuantities(self["global_quantities"],   coord=self.coordinate_system,  parent=self)

    @cached_property
    def boundary(self) -> Boundary:
        return EquilibriumBoundary(self["boundary"], coord=self.coordinate_system, parent=self)

    @cached_property
    def boundary_separatrix(self) -> BoundarySeparatrix:
        return EquilibriumBoundarySeparatrix(self["boundary_separatrix"], coord=self.coordinate_system,   parent=self)

    def plot(self, axis=None, *args,
             scalar_field=[],
             vector_field=[],
             mesh=True,
             boundary=True,
             contour_=False,
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
                      'g.',
                      linewidth=0.5,
                      markersize=2,
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

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                       linewidth=0.5, fill=False, closed=True))
            axis.plot([], [], 'r--', label="Separatrix")

        if mesh is not False:
            for idx in range(0, self.coordinate_system.mesh.shape[0], 4):
                ax0 = self.coordinate_system.mesh.axis(idx, axis=0)
                axis.add_patch(plt.Polygon(ax0.xy, fill=False, closed=True, color="b", linewidth=0.2))

            for idx in range(0, self.coordinate_system.mesh.shape[1], 4):
                ax1 = self.coordinate_system.mesh.axis(idx, axis=1)
                axis.plot(ax1.xy[:, 0], ax1.xy[:, 1],  "r", linewidth=0.2)

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


class Equilibrium(IDS, Actor[EquilibriumTimeSlice]):
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
    _stats_ = "time_slice", "grid._ggd"
    TimeSlice = EquilibriumTimeSlice

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def time_slice(self) -> TimeSeries[EquilibriumTimeSlice]:
        return TimeSeries[EquilibriumTimeSlice](self["time_slice"], parent=self)

    @cached_property
    def grid_ggd(self) -> TimeSeries[GGD]:
        return TimeSeries[GGD](self["grid_ggd"], parent=self)

    @property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        r0 = np.asarray([t_slice.vacuum_toroidal_field.r0 for t_slice in self.time_slice])
        b0 = np.asarray([t_slice.vacuum_toroidal_field.b0 for t_slice in self.time_slice])
        return VacuumToroidalField(r0[0], b0)

    @property
    def current_state(self) -> EquilibriumTimeSlice:
        return self.time_slice[-1]

    @property
    def previous_state(self) -> EquilibriumTimeSlice:
        return self.time_slice[-2]
        
    ####################################################################################
    # Plot profiles

    def plot(self, axis=None, *args, time: float = None, time_slice=True, ggd=False, **kwargs):
        if time_slice is not False:
            axis = self.time_slice(time).plot(axis, *args, **kwargs)
        if ggd:
            axis = self.grid_ggd(time).plot(axis, *args, **kwargs)
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
