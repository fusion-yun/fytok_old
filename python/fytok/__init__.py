__path__ = __import__('pkgutil').extend_path(__path__, __name__)


import sys
import pathlib
import pprint

mod_path = [(pathlib.Path(__path__[0])/"../../phys_modules").resolve()]
ext_path = (pathlib.Path(__path__[0])/"../../external").resolve()

if ext_path.exists():
    for d in ext_path.iterdir():
        if not d.is_dir():
            continue
        elif (d/"python").is_dir():
            mod_path.append(d/"python")
        else:
            mod_path.append(d)

mod_path = [p.as_posix() for p in mod_path]

sys.path.extend(mod_path)

try:
    from spdm.util.logger import logger
except Exception as error:
    pprint.pprint(sys.path)
    pprint.pprint(mod_path)
    pprint.pprint(f"Error: {error}")

else:

    logger.info(f"FY_MODULE_PATH={':'.join(mod_path)}")
