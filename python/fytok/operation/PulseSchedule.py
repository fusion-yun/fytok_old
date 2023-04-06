import numpy as np
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.Signal import Signal
from spdm.data.sp_property import sp_property

from ..common.IDS import IDS
from ..common.Misc import RZTuple
from spdm.common.tags import _next_


class PulseScheduleReference(Dict):
    reference_name: str = sp_property()
    """Reference name (e.g. in the native pulse schedule system of the device) {constant}"""

    reference: Signal = sp_property()
    """Reference waveform [mixed]"""

    reference_type: int = sp_property()
    """Reference type: 0:relative; 1: absolute; refers to the reference/data node {constant}"""

    envelope_type: int = sp_property()
    """Envelope type: 0:relative; 1: absolute; refers to the envelope bounds which are given by the
        reference/data_error_upper and reference/data_error_lower nodes {constant}"""


class PulseScheduleEvent(Dict):
    pass


class PulseScheduleFluxControl(Dict):

    @sp_property
    def i_plasma(self) -> PulseScheduleReference:
        """Plasma current [A] """
        return PulseScheduleReference(name='i_plasma', time=self._time, data=self._cache.i_plasma)

    @sp_property
    def loop_voltage(self) -> PulseScheduleReference:
        """Loop voltage [V] """
        return PulseScheduleReference(name='loop_voltage', time=self._time, data=self._cache.loop_voltage)

    @sp_property
    def li_3(self) -> PulseScheduleReference:
        """Internal inductance [-] """
        return PulseScheduleReference(name='li_3', time=self._time, data=self._cache.li_3)

    @sp_property
    def beta_normal(self) -> PulseScheduleReference:
        """Normalized toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] [-] """
        return PulseScheduleReference(name='beta_normal', time=self._time, data=self._cache.beta_normal)

    @sp_property
    def mode(self) -> Signal:
        """Control mode (operation mode and/or settings used by the controller) """
        return Signal(time=self._time, data=np.arange(self._time.shape[0]))


class PulseSchedulePositionControl(Dict):

    @sp_property
    def mode(self) -> Signal:
        """Control mode (operation mode and/or settings used by the controller) """
        return Signal(time=self._time, data=np.arange(self._time.shape[0]))

    @sp_property
    def magnetic_axis(self) -> RZTuple:
        """Magnetic axis position"""
        return RZTuple(PulseScheduleReference(name='magnetic_axis.r', time=self._time, data=self._cache.magnetic_axis.r),
                       PulseScheduleReference(name='magnetic_axis.z', time=self._time, data=self._cache.magnetic_axis.z))

    @sp_property
    def geometric_axis(self) -> RZTuple:
        """RZ position of the geometric axis (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the boundary)"""
        return RZTuple(PulseScheduleReference(name='geometric_axis.r', time=self._time, data=self._cache.geometric_axis.r),
                       PulseScheduleReference(name='geometric_axis.z', time=self._time, data=self._cache.geometric_axis.z))

    @sp_property
    def minor_radius(self) -> PulseScheduleReference:
        """Minor radius of the plasma boundary (defined as (Rmax-Rmin) / 2 of the boundary) [m]"""
        return PulseScheduleReference(name='minor_radius', time=self._time, data=self._cache.minor_radius)

    @sp_property
    def elongation(self) -> PulseScheduleReference:
        """Elongation of the plasma boundary [-]     """
        return PulseScheduleReference(name='elongation', time=self._time, data=self._cache.elongation)

    @sp_property
    def elongation_upper(self) -> PulseScheduleReference:
        """Elongation (upper half w.r.t. geometric axis) of the plasma boundary [-]     """
        return PulseScheduleReference(name='elongation_upper', time=self._time, data=self._cache.elongation_upper)

    @sp_property
    def elongation_lower(self) -> PulseScheduleReference:
        """Elongation (lower half w.r.t. geometric axis) of the plasma boundary [-]     """
        return PulseScheduleReference(name='elongation_lower', time=self._time, data=self._cache.elongation_lower)

    @sp_property
    def triangularity(self) -> PulseScheduleReference:
        """Triangularity of the plasma boundary [-]     """
        return PulseScheduleReference(name='triangularity', time=self._time, data=self._cache.triangularity)

    @sp_property
    def triangularity_upper(self) -> PulseScheduleReference:
        """Upper triangularity of the plasma boundary [-]     """
        return PulseScheduleReference(name='triangularity_upper', time=self._time, data=self._cache.triangularity_upper)

    @sp_property
    def triangularity_lower(self) -> PulseScheduleReference:
        """Lower triangularity of the plasma boundary [-]     """
        return PulseScheduleReference(name='triangularity_lower', time=self._time, data=self._cache.triangularity_lower)

    @sp_property
    def x_point(self):
        """Array of X-points, for each of them the RZ position is given     struct_array [max_size=2]     1- 1...N"""
        res = Dict(default_factory_array=lambda _time: Dict(r=PulseScheduleReference(name='x_point.r', time=_time),
                                                            z=PulseScheduleReference(name='x_point.z', time=_time)))
        for xp in self._cache.x_point:
            pit = res[_next_]
            pit.r.data = xp.r
            pit.z.data = xp.z
        return res

    @sp_property
    def strike_point(self) -> Dict:
        """Array of strike points, for each of them the RZ position is given     struct_array [max_size=4]     1- 1...N"""
        res = Dict(default_factory_array=lambda _time: RZTuple(PulseScheduleReference(name='strike_point.r', time=_time),
                                                               PulseScheduleReference(name='strike_point.z', time=_time)))
        for xp in self._cache.strike_point:
            pit = res[_next_]
            pit.r.data = xp.r
            pit.z.data = xp.z
        return res

    @sp_property
    def active_limiter_point(self) -> RZTuple:
        """RZ position of the active limiter point (point of the plasma boundary in contact with the limiter)     """
        return RZTuple(PulseScheduleReference(name='active_limiter_point.r', time=_time, data=self._cache.active_limiter_point.r),
                       PulseScheduleReference(name='active_limiter_point.z', time=_time, data=self._cache.active_limiter_point.z))

    @sp_property
    def boundary_outline(self) -> RZTuple:
        """Set of (R,Z) points defining the outline of the plasma boundary     struct_array [max_size=301]     1- 1...N"""
        res = Dict(default_factory_array=lambda _time: Dict(r=PulseScheduleReference(name='boundary_outline.r', time=_time),
                                                            z=PulseScheduleReference(name='boundary_outline.z', time=_time)))
        for xp in self._cache.boundary_outline:
            pit = res[_next_]
            pit.r.data = xp.r
            pit.z.data = xp.z
        return res

    @sp_property
    def gap(self) -> RZTuple:
        """Set of gaps, defined by a reference point and a direction."""
        res = Dict(default_factory_array=lambda _time: Dict(
            name="",
            identifier="",
            angle=0.0,
            r=0.0,
            z=0.0,
            value=PulseScheduleReference(name='gap.value', time=_time)
        ))
        for xp in self._cache.gap:
            pit = res[_next_]
            pit.name = str(xp.name)
            pit.identifier = str(xp.identifier)
            pit.r = float(xp.r)
            pit.z = float(xp.z)
            pit.value.data = xp.value

        return res


class PulseScheduleTF(Dict):

    b_field_tor_vacuum_r: PulseScheduleReference = sp_property()

    mode: Signal = sp_property()
    """Control mode (operation mode and/or settings used by the controller) """


class PulseSchedule(IDS):
    r"""Description of Pulse Schedule, described by subsystems waveform references and an enveloppe around them.
        The controllers, pulse schedule and SDN are defined in separate IDSs. All names and identifiers of subsystems
        appearing in the pulse_schedule must be identical to those used in the IDSs describing the related subsystems.
    """
    _IDS = "pulse_schedule"
    Event = PulseScheduleEvent
    FluxControl = PulseScheduleFluxControl
    PositionControl = PulseSchedulePositionControl
    TF = PulseScheduleTF

    @sp_property
    def ic(self):
        """Ion cyclotron heating and current drive system"""
        return self["ic"]

    @sp_property
    def ec(self):
        """Electron cyclotron heating and current drive system"""
        return self["ec"]

    @sp_property
    def lh(self):
        """Lower Hybrid heating and current drive system"""
        return self["lh"]

    @sp_property
    def nbi(self):
        """Neutral beam heating and current drive system"""
        return self["nbi"]

    @sp_property
    def density_control(self):
        """Gas injection system and density control references"""
        return self["density_control"]

    @sp_property
    def event(self) -> List[Event]:
        """List of events, either predefined triggers or events recorded during the pulse."""
        return self["Event"]

    @sp_property
    def flux_control(self) -> FluxControl:
        """Magnetic flux control references"""
        return self["flux_control"]

    @sp_property
    def position_control(self) -> PositionControl:
        """Plasma position and shape control references"""
        return self["position_control"]

    @sp_property
    def tf(self) -> TF:
        """Toroidal field references"""
        return self["tf"]
