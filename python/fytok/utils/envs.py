from spdm.utils import envs
import os

FY_DEBUG = os.environ.get("FY_DEBUG", True)
# os.environ["SP_DEBUG"] = str(FY_DEBUG)
# envs.SP_DEBUG = FY_DEBUG

__all__ = ["FY_DEBUG"]
