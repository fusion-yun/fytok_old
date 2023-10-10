# =============================================================================
# Authors:            Zhi YU
# Email:              yuzhi@ipp.ac.cn
# =============================================================================

import os
from .._schemas.imas_v3_38_1_dirty.__version__ import __version__ as imas_version
from .._schemas.imas_v3_38_1_dirty import equilibrium, core_profiles, core_sources,\
    ic_antennas, interferometer, lh_antennas, magnetics, nbi, pellets,\
    core_transport,  wall, pf_active, tf, transport_solver_numerics, utilities, ec_launchers, amns_data

__all__ = ["equilibrium", "core_profiles", "core_sources", "ec_launchers",
           "ic_antennas", "interferometer", "lh_antennas", "magnetics", "nbi", "pellets", "amns_data",
           "core_transport", "wall", "pf_active", "tf", "transport_solver_numerics", "utilities"]


IMAS_VERSION = os.environ.get("IMAS_VERSION", imas_version)

imas_version_major, *_ = IMAS_VERSION.split(".")

GLOBAL_SCHEMA = f"imas/{imas_version_major}"
