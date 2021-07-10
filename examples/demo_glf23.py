import numpy as np
from spdm.util.logger import logger
import sys
sys.path.append("/home/salmon/workspace/fytok/phys_modules/transport/core_transport/")

if __name__ == "__main__":

    from glf23 import glf23

    # from nclass import nclass
    glf23.glf.nmode=15

    glf23.glf.zevec_k_gf = np.zeros([20, 12, 12])
    logger.debug(glf23.glf2d(1))
    # logger.debug(dir(glf23.glf23_mod))
    # logger.debug(nclass.nclass_mod.__doc__)
