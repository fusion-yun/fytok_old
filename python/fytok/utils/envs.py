import os

FY_DEBUG = os.environ.get("FY_DEBUG", True)
os.environ["SP_DEBUG"] = str(FY_DEBUG)
__all__ = ["FY_DEBUG"]
