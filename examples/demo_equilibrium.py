from fytok.modules.Equilibrium import Equilibrium
from spdm.utils.logger import logger

from spdm.data.File import File

if __name__ == '__main__':
    # baseline
    device_desc = File("/home/salmon/workspace/fytok_data/mapping/ITER/imas/3/static/config.xml", format="XML").read()
    