import os

from pathlib import Path


mapping_path = Path(__file__).parent.resolve() / "_mapping"

if mapping_path.exists():
    SP_DATA_MAPPING_PATH = (
        ":".join([mapping_path.as_posix(), os.environ.get("SP_DATA_MAPPING_PATH", "")])
    ).strip(":")

    os.environ["SP_DATA_MAPPING_PATH"] = SP_DATA_MAPPING_PATH

    # logger.info(f"FyTok Mapping path: {SP_DATA_MAPPING_PATH}")

FY_DEBUG = os.environ.get("SP_DEBUG", "debug")

__all__ = ["FY_DEBUG"]
