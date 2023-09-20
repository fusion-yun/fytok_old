__path__ = __import__('pkgutil').extend_path(__path__, __name__)

try:
    from .__version__ import __version__
except:
    # try:
    #     import subprocess
    #     __version__ = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')
    # except:
    __version__ = "0.0.0"

import os

from pathlib import Path

from fytok.utils.logger import logger

# logger.name = __package__[:__package__.find('.')]

mapping_path = Path(__file__).parent.resolve() / "_mapping"

if mapping_path.exists():

    SP_DATA_MAPPING_PATH = (":".join([mapping_path.as_posix(),
                                      os.environ.get("SP_DATA_MAPPING_PATH", '')])).strip(':')

    os.environ["SP_DATA_MAPPING_PATH"] = SP_DATA_MAPPING_PATH

    logger.info(f"FyTok Mapping path: {SP_DATA_MAPPING_PATH}")


# external_path = (pathlib.Path(__path__[0])/"../../external").resolve()

# external_pkg_path = []

# if external_path.exists():
#     for d in external_path.iterdir():
#         if not d.is_dir():
#             continue
#         elif (d/"python").is_dir():
#             external_pkg_path.append(d/"python")
#         else:
#             external_pkg_path.append(d)

# external_pkg_path = [p.as_posix() for p in external_pkg_path]

# sys.path.extend(external_pkg_path)

# phys_modules_path = [(pathlib.Path(__path__[0])/"../../fymodules").resolve().as_posix()]

# sys.path.extend(phys_modules_path)

# from .Scenario import Scenario
# from .Tokamak import Tokamak

# try:
#     from fytok.utils.logger import logger
# except ModuleNotFoundError as error:
#     raise error
# else:

#     logger.info(f"""Using FyTok \t: {__version__}
#     EXTERNAL_PYTHON_PATH={':'.join(external_pkg_path)}
#     FY_MODULE_PATH={':'.join(phys_modules_path)}
# """)

# logger.info(f"FY_MODULE_PATH={':'.join(ext_mod_path)}")
