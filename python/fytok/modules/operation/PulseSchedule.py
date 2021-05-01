from functools import cached_property
import numpy as np
from spdm.data.Node import Dict


class PulseSchedule(Dict):
    r"""Description of Pulse Schedule, described by subsystems waveform references and an enveloppe around them.

    The controllers, pulse schedule and SDN are defined in separate IDSs. All names and identifiers of subsystems
    appearing in the pulse_schedule must be identical to those used in the IDSs describing the related subsystems.

    """
    _IDS = "pulse_schedule"

    def __init__(self,  *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def ids_properties(self):
        """Interface Data Structure properties. This element identifies the node above as an IDS"""
        return NotImplemented

    @cached_property
    def ic(self):
        """Ion cyclotron heating and current drive system"""
        return NotImplemented

    @cached_property
    def ec(self):
        """Electron cyclotron heating and current drive system"""
        return NotImplemented

    @cached_property
    def lh(self):
        """Lower Hybrid heating and current drive system"""
        return NotImplemented

    @cached_property
    def nbi(self):
        """Neutral beam heating and current drive system"""
        return NotImplemented

    @cached_property
    def density_control(self):
        """Gas injection system and density control references"""
        return NotImplemented

    class Reference(Dict):
        def __init__(self, *args, name=None, time=None, data=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__["_signal"] = Signal(time=time, data=data)

        @property
        def reference_name(self):
            """Reference name (e.g. in the native pulse schedule system of the device) {constant}"""
            return self.__name

        @property
        def reference(self):
            """Reference waveform [mixed]"""
            return self._signal

        @property
        def reference_type(self):
            """Reference type: 0:relative; 1: absolute; refers to the reference/data node {constant}"""
            return NotImplemented

        @property
        def envelope_type(self):
            """Envelope type: 0:relative; 1: absolute; refers to the envelope bounds which are given by the
            reference/data_error_upper and reference/data_error_lower nodes {constant}"""
            return NotImplemented

    class Event(Dict):
        pass

    @cached_property
    def event(self):
        """List of events, either predefined triggers or events recorded during the pulse."""
        return Dict(default_factory_array=lambda _holder=self: PulseSchedule.Event(None, parent=_holder))

    class FluxControl(Dict):
        def __init__(self,   *args, time=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__["_time"] = time

        @cached_property
        def i_plasma(self):
            """Plasma current [A] """
            return PulseSchedule.Reference(name='i_plasma', time=self._time, data=self._cache.i_plasma)

        @cached_property
        def loop_voltage(self):
            """Loop voltage [V] """
            return PulseSchedule.Reference(name='loop_voltage', time=self._time, data=self._cache.loop_voltage)

        @cached_property
        def li_3(self):
            """Internal inductance [-] """
            return PulseSchedule.Reference(name='li_3', time=self._time, data=self._cache.li_3)

        @cached_property
        def beta_normal(self):
            """Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] [-] """
            return PulseSchedule.Reference(name='li_3', time=self._time, data=self._cache.beta_normal)

        @cached_property
        def mode(self):
            """Control mode (operation mode and/or settings used by the controller) """
            return Signal(time=self._time, data=np.arange(self._time.shape[0]))

    @cached_property
    def flux_control(self):
        """Magnetic flux control references"""
        return PulseSchedule.FluxControl(self._cache.flux_control, time=self.time)

    class PositionControl(Dict):
        def __init__(self,  *args, time=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__["_time"] = time

        @cached_property
        def mode(self):
            """Control mode (operation mode and/or settings used by the controller) """
            return Signal(time=self._time, data=np.arange(self._time.shape[0]))

        @cached_property
        def magnetic_axis(self):
            """Magnetic axis position"""
            return Dict(r=PulseSchedule.Reference(name='magnetic_axis.r', time=self._time, data=self._cache.magnetic_axis.r),
                                 z=PulseSchedule.Reference(name='magnetic_axis.z', time=self._time, data=self._cache.magnetic_axis.z))

        @cached_property
        def geometric_axis(self):
            """RZ position of the geometric axis (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the boundary)"""
            return Dict(r=PulseSchedule.Reference(name='geometric_axis.r', time=self._time, data=self._cache.geometric_axis.r),
                                 z=PulseSchedule.Reference(name='geometric_axis.z', time=self._time, data=self._cache.geometric_axis.z))

        @cached_property
        def minor_radius(self):
            """Minor radius of the plasma boundary (defined as (Rmax-Rmin) / 2 of the boundary) [m]"""
            return PulseSchedule.Reference(name='minor_radius', time=self._time, data=self._cache.minor_radius)

        @cached_property
        def elongation(self):
            """Elongation of the plasma boundary [-]     """
            return PulseSchedule.Reference(name='elongation', time=self._time, data=self._cache.elongation)

        @cached_property
        def elongation_upper(self):
            """Elongation (upper half w.r.t. geometric axis) of the plasma boundary [-]     """
            return PulseSchedule.Reference(name='elongation_upper', time=self._time, data=self._cache.elongation_upper)

        @cached_property
        def elongation_lower(self):
            """Elongation (lower half w.r.t. geometric axis) of the plasma boundary [-]     """
            return PulseSchedule.Reference(name='elongation_lower', time=self._time, data=self._cache.elongation_lower)

        @cached_property
        def triangularity(self):
            """Triangularity of the plasma boundary [-]     """
            return PulseSchedule.Reference(name='triangularity', time=self._time, data=self._cache.triangularity)

        @cached_property
        def triangularity_upper(self):
            """Upper triangularity of the plasma boundary [-]     """
            return PulseSchedule.Reference(name='triangularity_upper', time=self._time, data=self._cache.triangularity_upper)

        @cached_property
        def triangularity_lower(self):
            """Lower triangularity of the plasma boundary [-]     """
            return PulseSchedule.Reference(name='triangularity_lower', time=self._time, data=self._cache.triangularity_lower)

        @cached_property
        def x_point(self):
            """Array of X-points, for each of them the RZ position is given     struct_array [max_size=2]     1- 1...N"""
            res = Dict(default_factory_array=lambda _time: Dict(r=PulseSchedule.Reference(name='x_point.r', time=_time),
                                                                                  z=PulseSchedule.Reference(name='x_point.z', time=_time)))
            for xp in self._cache.x_point:
                pit = res[_next_]
                pit.r.data = xp.r
                pit.z.data = xp.z
            return res

        @cached_property
        def strike_point(self):
            """Array of strike points, for each of them the RZ position is given     struct_array [max_size=4]     1- 1...N"""
            res = Dict(default_factory_array=lambda _time: Dict(r=PulseSchedule.Reference(name='strike_point.r', time=_time),
                                                                                  z=PulseSchedule.Reference(name='strike_point.z', time=_time)))
            for xp in self._cache.strike_point:
                pit = res[_next_]
                pit.r.data = xp.r
                pit.z.data = xp.z
            return res

        @cached_property
        def active_limiter_point(self):
            """RZ position of the active limiter point (point of the plasma boundary in contact with the limiter)     """
            return Dict(r=PulseSchedule.Reference(name='active_limiter_point.r', time=_time, data=self._cache.active_limiter_point.r),
                                 z=PulseSchedule.Reference(name='active_limiter_point.z', time=_time, data=self._cache.active_limiter_point.z))

        @cached_property
        def boundary_outline(self):
            """Set of (R,Z) points defining the outline of the plasma boundary     struct_array [max_size=301]     1- 1...N"""
            res = Dict(default_factory_array=lambda _time: Dict(r=PulseSchedule.Reference(name='boundary_outline.r', time=_time),
                                                                                  z=PulseSchedule.Reference(name='boundary_outline.z', time=_time)))
            for xp in self._cache.boundary_outline:
                pit = res[_next_]
                pit.r.data = xp.r
                pit.z.data = xp.z
            return res

        @cached_property
        def gap(self):
            """Set of gaps, defined by a reference point and a direction."""
            res = Dict(default_factory_array=lambda _time: Dict(
                name="",
                identifier="",
                angle=0.0,
                r=0.0,
                z=0.0,
                value=PulseSchedule.Reference(name='gap.value', time=_time)
            ))
            for xp in self._cache.gap:
                pit = res[_next_]
                pit.name = str(xp.name)
                pit.identifier = str(xp.identifier)
                pit.r = float(xp.r)
                pit.z = float(xp.z)
                pit.value.data = xp.value

            return res

    @cached_property
    def position_control(self):
        """Plasma position and shape control references"""
        return PulseSchedule.PositionControl(self._cache.position_control, time=self.time)

    class TF(Dict):
        def __init__(self, cache=None, *args, time=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__["_time"] = time
            self.__dict__["_cache"] = cache

        @cached_property
        def b_field_tor_vacuum_r(self):
            return PulseSchedule.Reference(name='b_field_tor_vacuum_r', time=self._time, data=self._cache.b_field_tor_vacuum_r)

        @cached_property
        def mode(self):
            """Control mode (operation mode and/or settings used by the controller) """
            return Signal(time=self._time, data=np.arange(self._time.shape[0]))

    @cached_property
    def tf(self):
        """Toroidal field references"""
        return PulseSchedule.TF(self._cache.position_control, time=self.time)

    @property
    def time(self):
        """Generic time {dynamic} [s]"""
        return self._time
