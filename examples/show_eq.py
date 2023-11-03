import pathlib

from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.logger import logger
from spdm.view.View import display

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

# os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"

output_path = pathlib.Path(f"{WORKSPACE}/output/")

if __name__ == "__main__":

    # Example: gfile I/O,  equilibrium.dump()

    # eq0 = Equilibrium(f"file+GEQdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890#equilibrium")
    eq0 = Equilibrium("file+geqdsk:///home/salmon/workspace/fytok_tutorial/tutorial/data/g900003.00230_ITER_15MA_eqdsk16HR.txt#equilibrium")
    eq0.refresh()

    display(eq0, title=f"EQUILIBRIUM", output=output_path/"EQUILIBRIUM.svg")

    # logger.debug(eq0.time_slice.current.profiles_1d.grid.psi_norm)
    # logger.debug(eq0.time_slice.current.profiles_1d.grid.psi)
    # logger.debug(eq0.time_slice.current.profiles_1d.grid.rho_tor)

    # with File(f"{output_path}/EQUILIBRIUM.gfile", mode="w", format="GEQdsk") as oid:
    #     oid.write({"equilibrium": eq0.dump()})

    # eq0.refresh()

    # eq1 = Equilibrium(f"file+GEQdsk://{WORKSPACE}/output/EQUILIBRIUM.gfile#equilibrium")

    # logger.debug(eq0.time_slice.current.profiles_1d.q.__value__)

    # logger.debug(eq1.time_slice.current.profiles_1d.q.__value__)
