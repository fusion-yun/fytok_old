from _imas import _T_core_transport

from spdm.util.logger import logger

if __name__ == "__main__":

    eq = _T_core_transport({"time_slice": [{"global_quantities": {"beta_pol": 1.23450}, }],
                            "model": [{"code": {"name": "dummy"}}]})

    logger.debug(eq.model[0].profiles_1d[0].electrons.particles.d)
