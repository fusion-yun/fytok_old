__path__ = __import__('pkgutil').extend_path(__path__, __name__)


import pathlib
import pprint
import sys

from .version import __version__


external_path = (pathlib.Path(__path__[0])/"../../external").resolve()

external_pkg_path = []

if external_path.exists():
    for d in external_path.iterdir():
        if not d.is_dir():
            continue
        elif (d/"python").is_dir():
            external_pkg_path.append(d/"python")
        else:
            external_pkg_path.append(d)

external_pkg_path = [p.as_posix() for p in external_pkg_path]

sys.path.extend(external_pkg_path)

phys_modules_path = [(pathlib.Path(__path__[0])/"../../fymodules/modules").resolve().as_posix()]

sys.path.extend(phys_modules_path)

from .modules.Scenario import Scenario
from .modules.Tokamak import Tokamak

# try:
#     from spdm.logger import logger
# except ModuleNotFoundError as error:
#     raise error
# else:

#     logger.info(f"""Using FyTok \t: {__version__}
#     EXTERNAL_PYTHON_PATH={':'.join(external_pkg_path)}
#     FY_MODULE_PATH={':'.join(phys_modules_path)}
# """)

# logger.info(f"FY_MODULE_PATH={':'.join(ext_mod_path)}")
