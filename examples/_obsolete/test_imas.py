from  fytok._imas import _T_equilibrium

from spdm.utils.logger import logger
from spdm.data.sp_property import sp_property


class Equilibrium(_T_equilibrium):
    time_slice = sp_property()


if __name__ == "__main__":

    eq = Equilibrium({"time_slice": [{"global_quantities": {"beta_pol": 1.23450}, }]})

    logger.debug(_T_equilibrium.time_slice.__doc__)
    logger.debug(eq.time_slice[0].global_quantities.beta_pol)
