__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
import typing
from spdm.data.sp_property import AttributeTree
from ..utils.logger import logger


GLOBAL_ONTOLOGY = os.environ.get("FYTOK_ONTOLOGY",  f"imas/3")


try:
    from .imas_lastest.__version__ import __version__ as imas_version

    from .imas_lastest import equilibrium, core_profiles, core_sources,\
        ic_antennas, interferometer, lh_antennas, magnetics, nbi, pellets,\
        core_transport, pf_active, tf, transport_solver_numerics, utilities, \
        ec_launchers, amns_data, wall

    __all__ = ["equilibrium", "core_profiles", "core_sources", "ec_launchers",
               "ic_antennas", "interferometer", "lh_antennas", "magnetics", "nbi", "pellets", "amns_data",
               "core_transport", "wall", "pf_active", "tf", "transport_solver_numerics", "utilities"]


except ModuleNotFoundError as error:
    imas_version = None

# else:
    if (GLOBAL_ONTOLOGY != f"imas/{imas_version[1:].split('.')[0]}"):
        raise RuntimeError(f"Global ontology {GLOBAL_ONTOLOGY} is not compatible with IMAS version {imas_version}")


# class DummyModule:
#     def __init__(self, name):
#         self._module = name

#     def __str__(self) -> str:
#         return f"<dummy_module '{__package__}.dummy.{self._module}'>"

#     def __getattr__(self, __name: str) -> typing.Type[AttributeTree]:
#         cls = type(__name, (AttributeTree,), {})
#         cls.__module__ = f"{__package__}.dummy.{self._module}"
#         return cls


# def __getattr__(key: str) -> DummyModule:
#     return DummyModule(key)
