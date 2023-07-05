import collections.abc

import numpy as np
import scipy.constants
from spdm.data.Entry import Entry
from spdm.data.Function import Function
from spdm.data.HTree import Dict, Node
from spdm.data.sp_property import SpDict, sp_property
from spdm.utils.tree_utils import merge_tree_recursive
atoms = {
    "e": {
        "label": "e",
        "z": -1,
        "element": [{"a": scipy.constants.m_e/scipy.constants.m_p, "z_n": 1, "atoms_n": 1}],
    },
    "electron": {
        "label": "e",
        "z": -1,
        "element": [{"a": scipy.constants.m_e/scipy.constants.m_p, "z_n": 1, "atoms_n": 1}],
    },
    "n": {
        "label": "n",
        "z": 0,
        "element": [{"a": scipy.constants.m_n/scipy.constants.m_p, "z_n": 0, "atoms_n": 1}],
    },
    "p": {
        "label": "p",
        "z": 1,
        "element": [{"a": 1, "z_n": 1, "atoms_n": 1}],
    },

    "H": {
        "label": "H",
        "z": 1,
        "z_ion": 1,
        "element": [{"a": 1, "z_n": 1, "atoms_n": 1}],

    },
    "D": {
        "label": "D",
        "z": 1,
        "z_ion": 1,
        "element": [{"a": 2, "z_n": 1, "atoms_n": 1}],

    },
    "T": {
        "label": "T",
        "z": 1,
        "z_ion": 1,
        "element": [{"a": 3, "z_n": 1, "atoms_n": 1}],

    },
    "He": {
        "label": "He",
        "z": 2,
        "z_ion": 2,
        "element": [{"a": 4, "z_n": 1, "atoms_n": 1}],

    },
    "Be": {
        "label": "Be",
        "z": 4,
        "z_ion": 4,
        "element": [{"a": 9, "z_n": 1, "atoms_n":   1}],

    },
    "Ar": {
        "label": "Ar",
        "z": 18,
        "z_ion": 18,
        "element": [{"a": 40, "z_n": 1, "atoms_n":   1}],

    }
}


def get_species(species):
    
    if isinstance(species, str):
        return atoms.get(species, {"label": species})
    
    elif isinstance(species, collections.abc.Sequence):
        return [atoms.get(s, {"label": s}) for s in species]
    
    elif isinstance(species, collections.abc.Mapping):
        label = species.get("label", None)
        if label is None:
            raise ValueError(f"Species {species} must have a label")
        else:
            return merge_tree_recursive(species, atoms.get(label, {"label": label}))
    else:
        raise TypeError(f"Unknown species type: {type(species)}")


nuclear_reaction = {
    r"D(t,n)\alpha": {
        "reactants": ["D", "T"],
        "products": ["He", "n"],
        "energy": 13.5e6,  # eV
        "reactivities":
        (  # m^3/s
            np.array([1.254e-32, 7.292e-31, 9.344e-30, 5.697e-29, 2.253e-28, 6.740e-28, 1.662e-27, 6.857e-27, 2.546e-26, 3.174e-26, 6.923e-26,
                      1.539e-25, 1.773e-25, 2.977e-25, 8.425e-25, 1.867e-24, 5.974e-24, 1.366e-23, 2.554e-23, 6.222e-23, 1.136e-22, 1.747e-22,
                      2.740e-22, 4.330e-22, 6.681e-22, 7.998e-22, 8.649e-22, ]),
            # eV
            np.array([0.20e3, 0.30e3, 0.40e3, 0.50e3, 0.60e3, 0.70e3, 0.80e3, 1.00e3, 1.25e3, 1.30e3, 1.50e3, 1.75e3, 1.80e3,
                      2.00e3, 2.50e3, 3.00e3, 4.00e3, 5.00e3, 6.00e3, 8.00e3, 10.0e3, 12.0e3, 15.0e3, 20.0e3, 30.0e3, 40.0e3, 50.0e3, ]),

        )
    }
}

thermal_reactivities = np.array([
    # | $T_i \\ (eV)$ | $D(t,n)\alpha \\ (m^3/s)$ | $^3He(d,p)\alpha \\ (m^3)/s $ | $D(d,p)T \\ (m^3)/s $ | $D(d,p)^3He \\ (m^3)/s $ |
    [0.20e3, 1.254e-32, 1.414e-41, 4.640e-34, 4.482e-34, ],
    [0.30e3, 7.292e-31, 1.033e-38, 2.071e-32, 2.004e-32, ],
    [0.40e3, 9.344e-30, 6.537e-37, 2.237e-31, 2.168e-31, ],
    [0.50e3, 5.697e-29, 1.241e-35, 1.204e-30, 1.169e-30, ],
    [0.60e3, 2.253e-28, 1.166e-34, 4.321e-30, 4.200e-30, ],
    [0.70e3, 6.740e-28, 6.960e-34, 1.193e-29, 1.162e-29, ],
    [0.80e3, 1.662e-27, 3.032e-33, 2.751e-29, 2.681e-29, ],
    [1.00e3, 6.857e-27, 3.057e-32, 1.017e-28, 9.933e-29, ],
    [1.25e3, 2.546e-26, 2.590e-31, 3.387e-28, 3.319e-28, ],
    [1.30e3, 3.174e-26, 3.708e-31, 4.143e-28, 4.660e-28, ],
    [1.50e3, 6.923e-26, 1.317e-30, 8.431e-28, 8.284e-28, ],
    [1.75e3, 1.539e-25, 4.813e-30, 1.739e-27, 1.713e-27, ],
    [1.80e3, 1.773e-25, 6.053e-30, 1.976e-27, 1.948e-27, ],
    [2.00e3, 2.977e-25, 1.399e-29, 3.150e-27, 3.110e-27, ],
    [2.50e3, 8.425e-25, 7.477e-29, 7.969e-27, 7.905e-27, ],
    [3.00e3, 1.867e-24, 2.676e-28, 1.608e-26, 1.602e-26, ],
    [4.00e3, 5.974e-24, 1.710e-27, 4.428e-26, 4.447e-26, ],
    [5.00e3, 1.366e-23, 6.377e-27, 9.024e-26, 9.128e-26, ],
    [6.00e3, 2.554e-23, 1.739e-26, 1.545e-25, 1.573e-25, ],
    [8.00e3, 6.222e-23, 7.504e-26, 3.354e-25, 3.457e-25, ],
    [10.0e3, 1.136e-22, 2.126e-25, 5.781e-25, 6.023e-25, ],
    [12.0e3, 1.747e-22, 4.715e-25, 8.723e-25, 9.175e-25, ],
    [15.0e3, 2.740e-22, 1.175e-24, 1.390e-24, 1.481e-24, ],
    [20.0e3, 4.330e-22, 3.482e-24, 2.399e-24, 2.603e-24, ],
    [30.0e3, 6.681e-22, 1.363e-23, 4.728e-24, 5.271e-24, ],
    [40.0e3, 7.998e-22, 3.160e-23, 7.249e-24, 8.235e-24, ],
    [50.0e3, 8.649e-22, 5.554e-23, 9.838e-24, 1.133e-23, ],
])
