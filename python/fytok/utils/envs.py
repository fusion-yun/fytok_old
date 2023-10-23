from spdm.utils import envs
import os
import getpass
from ..__version__ import version
FY_DEBUG = os.environ.get("FY_DEBUG", True)
# os.environ["SP_DEBUG"] = str(FY_DEBUG)
# envs.SP_DEBUG = FY_DEBUG
FY_VERSION = version

FY_JOBID = f"fytok_{version.replace('.','')}/{getpass.getuser().lower()}_{os.getpid()}_{os.uname().nodename.lower()}"

__all__ = ["FY_DEBUG", "FY_JOBID", "FY_VERSION"]
