from spdm.utils import envs
import os
import getpass

try:
    from ..__version__ import version
except Exception:
    FY_VERSION = "develop"
else:
    FY_VERSION = version

try:
    from ..extension import tags as extension_tags
except ImportError:
    FY_EXT_VERSION = "n/a"
else:
    FY_EXT_VERSION = extension_tags


FY_DEBUG = os.environ.get("FY_DEBUG", True)

# os.environ["SP_DEBUG"] = str(FY_DEBUG)
# envs.SP_DEBUG = FY_DEBUG


FY_JOBID = f"fytok_{getpass.getuser().lower()}_{os.uname().nodename.lower()}_{os.getpid()}"

__all__ = ["FY_DEBUG", "FY_JOBID", "FY_VERSION", "FY_EXT_VERSION"]
