from spdm.numlib import constants

atoms = {
    "e": {
        "label": "e",
        "z_ion": -1,
        "element": [{"a": constants.m_e/constants.m_p, "z_n": 1, "atoms_n": 1}],
    },
    "H": {
        "label": "H",
        "z_ion": 1,
        "element": [{"a": 1, "z_n": 1, "atoms_n": 1}],

    },
    "D": {
        "label": "D",
        "z_ion": 1,
        "element": [{"a": 2, "z_n": 1, "atoms_n": 1}],

    },
    "He": {
        "label": "He",
        "z_ion": 2,
        "element": [{"a": 4, "z_n": 1, "atoms_n": 1}],

    },
    "Be": {
        "label": "Be",
        "z_ion": 4,
        "element": [{"a": 9, "z_n": 1, "atoms_n":   1}],

    },
    "Ar": {
        "label": "Ar",
        "z_ion": 18,
        "element": [{"a": 40, "z_n": 1, "atoms_n":   1}],

    }
}
