
import matplotlib.pyplot as plt
from _imas.magnetics import _T_magnetics



class Magnetics(_T_magnetics):
    """Magnetic diagnostics for equilibrium identification and plasma shape control.
    """

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
