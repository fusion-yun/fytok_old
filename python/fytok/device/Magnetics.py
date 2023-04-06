
import matplotlib.pyplot as plt
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.Signal import Signal
from spdm.data.sp_property import sp_property

from ..common.IDS import IDS
from ..common.Misc import Identifier, RZTuple


class MagneticsFluxLoop(Dict):

    name: str = sp_property()
    """Name of the probe {static}  """

    identifier: str = sp_property()
    """ID of the probe {static}  """

    type: Identifier = sp_property()
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

    position: RZTuple = sp_property()
    """List of (R,Z,phi) points defining the position of the loop (see data structure documentation FLUXLOOPposition.pdf) {static}   """

    indices_differential: float = sp_property()
    """Indices (from the flux_loop array of structure) of the two flux loops used to build the flux difference flux(second index) - flux(first index).
         Use only if ../type/index = 6, leave empty otherwise {static}  """

    area: float = sp_property()
    """Effective area (ratio between flux and average magnetic field over the loop) {static} [m^2]    """

    gm9: float = sp_property()
    """Integral of 1/R over the loop area (ratio between flux and magnetic rigidity R0.B0). Use only if ../type/index = 3 to 6,
          leave empty otherwise. {static} [m]         """

    flux: Signal = sp_property()
    """Measured magnetic flux over loop in which Z component of normal to loop is directed downwards (negative grad Z direction) [Wb]."""

    voltage: Signal = sp_property()
    """Measured voltage between the loop terminals [V]"""


class MagneticsMagneticProbe(Dict):

    name: str = sp_property()
    """Name of the probe {static}  """

    identifier: str = sp_property()
    """ID of the probe {static}"""

    type: Identifier = sp_property()
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

    position: RZTuple = sp_property()
    """R, Z, Phi position of the coil centre    structure    """

    poloidal_angle: float = sp_property()
    """Angle of the sensor normal vector (n) with respect to horizontal plane (clockwise as in cocos=11 theta-like angle).
        Zero if sensor normal vector fully in the horizontal plane and oriented towards increasing major radius. Values in [0 , 2Pi]
        """

    toroidal_angle: float = sp_property()
    """Angle of the projection of the sensor normal vector (n) in the horizontal plane with the increasing R direction (i.e. grad(R))
        (angle is counter-clockwise from above as in cocos=11 phi-like angle). Values should be taken modulo pi with values within (-pi/2,pi/2].
        Zero if projected sensor normal is parallel to grad(R), pi/2 if it is parallel to grad(phi). """

    indices_differential: Dict = sp_property()
    """Indices (from the bpol_probe array of structure) of the two probes used to build the field difference field(second index) - field(first index).
        Use only if ../type/index = 6, leave empty otherwise {static}    INT_1D    1- 1...2"""

    bandwidth_3db: Dict = sp_property()
    """3dB bandwith (first index : lower frequency bound, second index : upper frequency bound) {static} [Hz]    """

    area: float = sp_property()
    """Area of each turn of the sensor; becomes effective area when multiplied by the turns {static} [m^2]  """

    length: float = sp_property()
    """Length of the sensor along it's normal vector (n) {static} [m]  """

    turns: float = sp_property()
    """Turns in the coil, including sign {static}    INT_0D    """

    field: Signal = sp_property()
    """Magnetic field component in direction of sensor normal axis(n) averaged over sensor volume defined by area and length,
        where n = cos(poloidal_angle)*cos(toroidal_angle)*grad(R) - sin(poloidal_angle)*grad(Z) + cos(poloidal_angle)*sin(toroidal_angle)*grad(Phi)/norm(grad(Phi))[T].
        This quantity is COCOS-dependent, with the following transformation: """

    voltage: Signal = sp_property()
    """Voltage on the coil terminals[V]

           .data: Data {dynamic}[as_parent]
           .validity_timed: Indicator of the validity of the data for each time slice.
                              * 0: valid from automated processing,
                              * 1: valid and certified by the diagnostic RO;
                              * -1: means problem identified in the data processing(request verification by the diagnostic RO),
                              * -2: invalid data, should not be used(values lower than - 2 have a code-specific meaning detailing the origin of their invalidity) {dynamic}    INT_1D    1 - b_field_pol_probe(i1)/voltage/time
           .validity: Indicator of the validity of the data for the whole acquisition period. 0: valid from automated processing,
                              * 1: valid and certified by the diagnostic RO;
                              * -1: means problem identified in the data processing(request verification by the diagnostic RO),
                              * -2: invalid data, should not be used(values lower than - 2 have a code-specific meaning detailing the origin of their invalidity) {constant}
           .time: time(:)    Time {dynamic} [s]
        """

    @sp_property
    def non_linear_response(self):
        """Non-linear response of the probe(typically in case of a Hall probe)"""
        return Dict(b_field_linear=self._cache.b_field_linear,
                    b_field_non_linear=Dict(b_field_linear=self._cache.b_field_non_linear))


class Magnetics(IDS):
    """Magnetic diagnostics for equilibrium identification and plasma shape control.
    """

    FluxLoop = MagneticsFluxLoop
    MagneticProbe = MagneticsMagneticProbe

    flux_loop: List[FluxLoop] = sp_property()
    """Flux loops; partial flux loops can be described   """

    b_field_pol_probe: List[MagneticProbe] = sp_property()
    """Poloidal field probes struct_array [max_size= 200] """

    def plot(self, axis=None, *args, with_circuit=False, **kwargs):

        if axis is None:
            axis = plt.gca()
        for idx, p_probe in enumerate(self.bpol_probe):
            pos = p_probe.position

            axis.add_patch(plt.Circle((pos.r, pos.z), 0.01))
            axis.text(pos.r, pos.z, idx,
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize='xx-small')

        for p in self.flux_loop:
            axis.add_patch(plt.Rectangle((p.position.r,  p.position.z), 0.01, 0.01))
            # axis.text(p.position.r, p.position.z, p.name,
            #           horizontalalignment='center',
            #           verticalalignment='center',
            #           fontsize='xx-small')
        return axis
