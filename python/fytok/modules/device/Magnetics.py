import numpy as np
from spdm.data.Node import Dict, List, _not_found_
from spdm.data.Node import sp_property

from ..common.IDS import IDS
from ..common.Signal import Signal
from ..common.Misc import Identifier


class MagneticsFluxLoop(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, ** kwargs)

    @sp_property
    def name(self) -> str:
        """Name of the probe {static}  """
        return self["name"]

    @sp_property
    def identifier(self) -> str:
        """ID of the probe {static}  """
        return self["identifier"]

    @sp_property
    def type(self) -> Identifier:
        """Probe type. Available options (refer to the children of this identifier structure) :

            ==========================  ==========  ================================
            Name                        Index       Description
            ==========================  ==========  ================================
            toroidal                    1           Toroidal flux loop
            saddle                      2           Saddle loop
            diamagnetic_internal        3           Diamagnetic internal loop
            diamagnetic_external        4           Diamagnetic external loop
            diamagnetic_compensation    5           Diamagnetic compensation loop
            diamagnetic_differential    6           Diamagnetic differential loop
            ==========================  ==========  ================================
        """
        return Identifier(**self.get("type", _not_found_)._as_dict())

    @sp_property
    def position(self):
        """List of (R,Z,phi) points defining the position of the loop (see data structure documentation FLUXLOOPposition.pdf) {static}   """
        return NotImplemented

    @sp_property
    def indices_differential(self):
        """Indices (from the flux_loop array of structure) of the two flux loops used to build the flux difference flux(second index) - flux(first index).
         Use only if ../type/index = 6, leave empty otherwise {static}  """
        return NotImplemented

    @sp_property
    def area(self):
        """Effective area (ratio between flux and average magnetic field over the loop) {static} [m^2]    """
        return NotImplemented

    @sp_property
    def gm9(self):
        """Integral of 1/R over the loop area (ratio between flux and magnetic rigidity R0.B0). Use only if ../type/index = 3 to 6,
          leave empty otherwise. {static} [m]         """
        return NotImplemented

    @sp_property
    def flux(self):
        """Measured magnetic flux over loop in which Z component of normal to loop is directed downwards (negative grad Z direction) [Wb].
         """
        return NotImplemented

    @sp_property
    def voltage(self):
        """Measured voltage between the loop terminals [V]"""
        return NotImplemented


class MagneticsMagneticProbe(Dict):
    def __init__(self,   *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def name(self):
        """Name of the probe {static}  """
        return str(self["name"])

    @sp_property
    def identifier(self):
        """ID of the probe {static}  """
        return str(self["identifier"])

    @sp_property
    def type(self):
        """Probe type. Available options (refer to the children of this identifier structure) :

           =============== =========== ==============================
           Name            Index       Description
           =============== =========== ==============================
           position        1           Position measurement probe
           mirnov          2           Mirnov probe
           hall            3           Hall probe
           flux_gate       4           Flux gate probe
           faraday_fiber   5           Faraday fiber
           differential    6           Differential probe
           =============== =========== ==============================
        """
        return Identifier(self["type"])

    @sp_property
    def position(self):
        """R, Z, Phi position of the coil centre    structure    """
        return NotImplemented
        # return Dict(
        #     r=float(self["position.r"]),
        #     z=float(self["position.z)"],
        #     phi=float(self["position.phi"])
        # )

    @sp_property
    def poloidal_angle(self):
        """Angle of the sensor normal vector (n) with respect to horizontal plane (clockwise as in cocos=11 theta-like angle).
        Zero if sensor normal vector fully in the horizontal plane and oriented towards increasing major radius. Values in [0 , 2Pi]
        """
        return self["poloidal_angle"]

    @sp_property
    def toroidal_angle(self):
        """Angle of the projection of the sensor normal vector (n) in the horizontal plane with the increasing R direction (i.e. grad(R))
        (angle is counter-clockwise from above as in cocos=11 phi-like angle). Values should be taken modulo pi with values within (-pi/2,pi/2].
        Zero if projected sensor normal is parallel to grad(R), pi/2 if it is parallel to grad(phi). """
        return self["toroidal_angle"]

    @sp_property
    def indices_differential(self):
        """Indices (from the bpol_probe array of structure) of the two probes used to build the field difference field(second index) - field(first index).
        Use only if ../type/index = 6, leave empty otherwise {static}    INT_1D    1- 1...2"""
        return Dict(self["toroidal_angle"])

    @sp_property
    def bandwidth_3db(self):
        """3dB bandwith (first index : lower frequency bound, second index : upper frequency bound) {static} [Hz]    """
        return Dict(self._cache.bandwidth_3db)

    @sp_property
    def area(self):
        """Area of each turn of the sensor; becomes effective area when multiplied by the turns {static} [m^2]  """
        return self._cache.area

    @sp_property
    def length(self):
        """Length of the sensor along it's normal vector (n) {static} [m]  """
        return self._cache.length

    @sp_property
    def turns(self):
        """Turns in the coil, including sign {static}    INT_0D    """
        return self._cache.turns

    @sp_property
    def field(self):
        """Magnetic field component in direction of sensor normal axis (n) averaged over sensor volume defined by area and length,
        where n = cos(poloidal_angle)*cos(toroidal_angle)*grad(R) - sin(poloidal_angle)*grad(Z) + cos(poloidal_angle)*sin(toroidal_angle)*grad(Phi)/norm(grad(Phi)) [T].
        This quantity is COCOS-dependent, with the following transformation :"""
        return Signal(self.time, data=self._cache.field)

    @sp_property
    def voltage(self):
        """Voltage on the coil terminals [V]

           .data            : Data {dynamic} [as_parent]
           .validity_timed : Indicator of the validity of the data for each time slice.
                              * 0: valid from automated processing,
                              * 1: valid and certified by the diagnostic RO;
                              * -1: means problem identified in the data processing (request verification by the diagnostic RO),
                              * -2: invalid data, should not be used (values lower than -2 have a code-specific meaning detailing the origin of their invalidity) {dynamic}    INT_1D    1- b_field_pol_probe(i1)/voltage/time
           .validity        : Indicator of the validity of the data for the whole acquisition period. 0: valid from automated processing,
                              * 1: valid and certified by the diagnostic RO;
                              * -1: means problem identified in the data processing (request verification by the diagnostic RO),
                              * -2: invalid data, should not be used (values lower than -2 have a code-specific meaning detailing the origin of their invalidity) {constant}
           .time           : time(:)    Time {dynamic} [s]
        """
        return Signal(time=self.time,
                      data=self._cache.voltage.data,
                      validity_timed=self._cache.voltage.validity_timed,
                      validity=int(self._cache.voltage.validity)
                      )

    @sp_property
    def non_linear_response(self):
        """Non-linear response of the probe (typically in case of a Hall probe)"""
        return Dict(b_field_linear=self._cache.b_field_linear,
                    b_field_non_linear=Dict(b_field_linear=self._cache.b_field_non_linear))


class MagneticsRogowskiCoil(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)


class Magnetics(IDS):
    """Magnetic diagnostics for equilibrium identification and plasma shape control.

    """
    FluxLoop = MagneticsFluxLoop
    MagneticProbe = MagneticsMagneticProbe
    RogowskiCoil = MagneticsRogowskiCoil

    def __init__(self,  *args,   ** kwargs):
        super().__init__(*args, ** kwargs)

    @sp_property
    def flux_loop(self) -> List[FluxLoop]:
        """Flux loops; partial flux loops can be described   """
        return self["flux_loop"]

    @sp_property
    def b_field_pol_probe(self) -> List[MagneticProbe]:
        """Poloidal field probes    struct_array [max_size=200] """
        return self["b_field_pol_probe"]

    @sp_property
    def b_field_tor_probe(self) -> List[MagneticProbe]:
        """Toroidal field probes    struct_array [max_size=20] """
        return self["b_field_pol_probe"]

    @sp_property
    def rogowski_coil(self) -> List[RogowskiCoil]:
        """Set of Rogowski coils"""
        return self["rogowski_coil"]
