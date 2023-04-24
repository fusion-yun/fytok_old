from fytok.transport.Equilibrium import Equilibrium
from spdm.util.logger import logger

from spdm.data.File import File

if __name__ == '__main__':
    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="GEQdsk").read()

    # eq = Equilibrium({"time_slice": [{"global_quantities": {"beta_pol": 1.23450}, }]})
    eq = Equilibrium({"time_slice": [eqdsk_file]})

    logger.debug(eq.time_slice[0].profiles_1d.psi)
