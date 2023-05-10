
import collections

import matplotlib.pyplot as plt
from  fytok._imas.lastest.pf_active import _T_pf_active



class PFActive(_T_pf_active):

    def plot(self, axis=None, *args, with_circuit=False, **kwargs):

        if axis is None:
            axis = plt.gca()

        for coil in self.coil:
            rect = coil.element[0].geometry.rectangle

            axis.add_patch(plt.Rectangle((rect.r - rect.width / 2.0,  rect.z - rect.height / 2.0),
                                         rect.width,  rect.height,
                                         **collections.ChainMap(kwargs,  {"fill": False})))
            axis.text(rect.r, rect.z, coil.name,
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize='xx-small')

        return axis
