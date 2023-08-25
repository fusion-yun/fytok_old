from  fytok._imas.lastest import _T_equilibrium

from spdm.utils.logger import logger

if __name__ == "__main__":

    eq = _T_equilibrium({"time_slice": [{"global_quantities": {"beta_pol": 1.23450}, }],
                            "model": [{"code": {"name": "dummy"}}]})

    logger.debug(eq.time_slice[0].global_quantities.beta_pol)